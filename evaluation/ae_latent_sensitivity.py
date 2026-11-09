#!/usr/bin/env python3
"""
Decoder-sensitivity control for the FM round-trip comparison.

Motivation
----------
The round-trip protocol measures MS-STFT between x and decode(z_hat), so a
difference between two models can come from either
  (i)  the FM field being less invertible (larger ||z_hat - z0||), or
  (ii) the AE's decoder amplifying a given latent error more strongly.
Because each FM is paired with a *different* AE, (ii) is a confound: it is not a
property of the flow model at all.

This script isolates (ii).  For each AE it injects perturbations of calibrated
relative L2 magnitude into real latents, decodes, and measures MS-STFT -- giving
the transfer curve  latent relative error -> MS-STFT.  Evaluating that curve at
the FM round trip's *observed* latent error yields the MS-STFT predicted by
decoder sensitivity alone; comparing it against the measured round-trip MS-STFT
says how much of the gap is attributable to the flow models.

USAGE
    python evaluation/ae_latent_sensitivity.py \
        --fm_a .../fm_univ_best.pt --fm_b .../fm_softnorm_best.pt \
        --label_a univ --label_b softnorm --num_clips 32 --steps 250
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data import collect_validation_stems, load_stem_clip
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss
from evaluation.eval_fm_roundtrip_msstft import (
    load_fm, ae_encode, ae_decode, roundtrip, msstft)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_a", required=True)
    p.add_argument("--fm_b", required=True)
    p.add_argument("--label_a", default="univ")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--num_clips", type=int, default=32)
    p.add_argument("--num_dirs", type=int, default=3,
                   help="random perturbation directions averaged per clip per eps")
    p.add_argument("--eps", default="0.0,0.05,0.10,0.15,0.20,0.30,0.40",
                   help="relative latent perturbation magnitudes (fraction of ||z0||)")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="evaluation/eval_fm_roundtrip")
    args = p.parse_args()

    device = torch.device(args.device)
    sr, n = args.sample_rate, int(args.sample_rate * args.clip_seconds)
    eps_list = [float(e) for e in args.eps.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    M = {}
    for lab, path in [(args.label_a, args.fm_a), (args.label_b, args.fm_b)]:
        fm, ck = load_fm(path, device)
        M[lab] = {"fm": fm, "ae": AutoencoderWrapper(ck["ae_checkpoint"], device)}
        print(f"[Load] {lab:12s} AE {Path(ck['ae_checkpoint']).stem}")

    stems = collect_validation_stems(sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(stems), size=args.num_clips, replace=False)

    loss_fn = SAMELoss().to(device)
    curves = {lab: {e: [] for e in eps_list} for lab in M}
    actual = {lab: {"rel_err": [], "msstft": []} for lab in M}

    for j, i in enumerate(idx):
        a = load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy()
        x = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)

        for lab, d in M.items():
            z0 = ae_encode(d["ae"], x)                      # (1, D, T_lat)
            nz = z0.norm()

            # --- actual FM round trip, for the observed error ---
            z_hat, _ = roundtrip(d["fm"], z0, args.steps)
            x_rt = ae_decode(d["ae"], z_hat, n)
            actual[lab]["rel_err"].append(
                float((z_hat - z0).norm() / (nz + 1e-12)))
            actual[lab]["msstft"].append(msstft(loss_fn, x_rt, x).item())

            # --- calibrated random perturbation transfer curve ---
            for eps in eps_list:
                vals = []
                for k in range(args.num_dirs):
                    if eps == 0.0:
                        z_p = z0
                    else:
                        g = torch.randn_like(z0)
                        z_p = z0 + eps * nz * g / (g.norm() + 1e-12)
                    x_p = ae_decode(d["ae"], z_p, n)
                    vals.append(msstft(loss_fn, x_p, x).item())
                curves[lab][eps].append(float(np.mean(vals)))
        print(f"  [{j+1}/{len(idx)}]", flush=True)

    # ---- report ----
    summary = {}
    print(f"\n{'=' * 100}")
    print("Decoder sensitivity: MS-STFT caused by a latent error of given relative L2 size")
    print(f"{args.num_clips} clips, {args.num_dirs} random directions per point, "
          f"MS-STFT vs the unperturbed original")
    print(f"{'=' * 100}")
    hdr = "eps (rel L2):" + "".join(f"{e:>10.2f}" for e in eps_list)
    print(f"{'AE (model)':<18}{hdr}")
    print("-" * 100)
    for lab in M:
        row = "".join(f"{np.mean(curves[lab][e]):>10.3f}" for e in eps_list)
        print(f"{lab:<18}{row}")
        summary[lab] = {"curve": {str(e): float(np.mean(curves[lab][e])) for e in eps_list}}
    print("-" * 100)

    # local slope d(MS-STFT)/d(eps), taken around the observed round-trip error
    for lab in M:
        obs = float(np.mean(actual[lab]["rel_err"]))
        meas = float(np.mean(actual[lab]["msstft"]))
        # linear interpolation of the curve at the observed error
        xs = np.array(eps_list); ys = np.array([np.mean(curves[lab][e]) for e in eps_list])
        pred = float(np.interp(obs, xs, ys))
        # slope over the two bracketing points
        hi = int(np.searchsorted(xs, obs, side="right"))
        lo = max(hi - 1, 0); hi = min(hi, len(xs) - 1)
        slope = ((ys[hi] - ys[lo]) / (xs[hi] - xs[lo])) if xs[hi] != xs[lo] else float("nan")
        summary[lab].update({
            "observed_rel_err": obs, "measured_roundtrip_msstft": meas,
            "predicted_from_sensitivity": pred,
            "residual": meas - pred,
            "slope_msstft_per_unit_rel_err": float(slope),
            "zero_eps_msstft": float(np.mean(curves[lab][0.0])),
        })
        print(f"{lab:<10} observed latent err {obs:.4f} -> measured round-trip MS-STFT "
              f"{meas:.3f}")
        print(f"{'':10} decoder sensitivity alone predicts {pred:.3f} "
              f"(slope {slope:.2f} MS-STFT per unit rel err); unexplained residual "
              f"{meas - pred:+.3f}")

    # ---- variance decomposition ----
    A, B = args.label_a, args.label_b
    sa, sb = summary[A], summary[B]
    print(f"\nVariance decomposition of the observed gap "
          f"({sb['measured_roundtrip_msstft']:.3f} - {sa['measured_roundtrip_msstft']:.3f} "
          f"= {sb['measured_roundtrip_msstft'] - sa['measured_roundtrip_msstft']:+.3f}):")
    # what the gap would be if BOTH models had the *same* latent error
    # (use each AE's own sensitivity, at the mean observed error)
    print(f"  decoder sensitivity differs: {sb['slope_msstft_per_unit_rel_err']:.2f} "
          f"({B}) vs {sa['slope_msstft_per_unit_rel_err']:.2f} ({A}) MS-STFT per unit rel err")
    print(f"  observed latent error differs: {sa['observed_rel_err']:.4f} ({A}) vs "
          f"{sb['observed_rel_err']:.4f} ({B})")

    (out / "decoder_sensitivity.json").write_text(json.dumps(
        {"summary": summary, "protocol": {
            "num_clips": args.num_clips, "num_dirs": args.num_dirs,
            "eps": eps_list, "steps": args.steps,
            "note": "perturbations are random directions; the FM round-trip error is a "
                    "structured error of comparable norm"}, "raw_actual": actual},
        indent=2))
    print(f"\nJSON: {out / 'decoder_sensitivity.json'}")


if __name__ == "__main__":
    main()
