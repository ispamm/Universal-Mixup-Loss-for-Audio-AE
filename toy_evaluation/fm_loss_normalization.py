#!/usr/bin/env python3
"""
Is the RF validation loss comparable across two DIFFERENT latent spaces?

Motivation
----------
Experiment 4 compares `fm_univ` (val loss 0.238) against `fm_softnorm`
(val loss 0.326) and reads the difference as evidence that the cone-structured
latent makes flow matching easier to learn (M-FM1).  But the objective is

    L = E || f_theta(z_t, t) - (z_noise - z_data) ||^2

and its *scale* is set by the target's own energy
E ||z_noise - z_data||^2, which depends on the geometry of the latent space and
on how much the minibatch OT coupling manages to straighten the paths.  A latent
space with a smaller transport target energy will report a smaller raw MSE even
if the model fits it no better.  Raw MSE across two latent spaces is therefore
not an apples-to-apples comparison unless the target energy is reported too.

This script measures, for each latent space, on held-out latents and with the
same protocol as training (batch 32, OT coupling, t ~ U(0,1)):

  build_mse        the checkpoint's own RF loss (sanity check vs the logged value)
  target_energy    E ||z_noise - z_data||^2  (OT-coupled and uncoupled)
  var_target       Var of the target = the constant-predictor MSE
  relative_loss    build_mse / var_target   <- the comparable quantity
  r2               1 - build_mse / var_target

and also the same quantities without OT coupling, which directly measures how
much coupling straightens the transport in each latent space (M-FM1's premise).

USAGE
    python toy_evaluation/fm_loss_normalization.py \
        --fm_a .../fm_univ_best.pt --fm_b .../fm_softnorm_best.pt \
        --label_a univ --label_b softnorm --num_clips 2000
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.train_fm_pilot import ot_coupling
from toy_evaluation.eval_fm_roundtrip_msstft import load_fm, ae_encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_a", required=True)
    p.add_argument("--fm_b", required=True)
    p.add_argument("--label_a", default="univ")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--num_clips", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32,
                   help="must match the training/protocol batch size used for OT coupling")
    p.add_argument("--t_samples", type=int, default=20,
                   help="timestep draws per batch (averaged)")
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="toy_evaluation/eval_fm_roundtrip")
    args = p.parse_args()

    device = torch.device(args.device)
    sr, n = args.sample_rate, int(args.sample_rate * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    stems = collect_validation_stems(sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(stems), size=min(args.num_clips, len(stems)), replace=False)

    report = {}
    for lab, path in [(args.label_a, args.fm_a), (args.label_b, args.fm_b)]:
        fm, ck = load_fm(path, device)
        ae = ToyModelWrapper(ck["ae_checkpoint"], device)
        print(f"\n[{lab}] FM {Path(path).name} (logged val {ck.get('val_loss'):.4f}), "
              f"AE {Path(ck['ae_checkpoint']).stem}")

        # ---- encode held-out clips ----
        zs = []
        for k in range(0, len(idx), 8):
            b = idx[k:k + 8]
            x = torch.from_numpy(np.stack(
                [load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy()
                 for i in b])).unsqueeze(1).to(device)
            zs.append(ae_encode(ae, x).cpu())
        Z = torch.cat(zs)                                  # (N, D, T_lat)
        print(f"  encoded {len(Z)} val latents, shape {tuple(Z.shape[1:])}")

        gen = torch.Generator(device=device).manual_seed(args.seed)
        acc = {k: [] for k in ("mse_ot", "energy_ot", "mse_raw", "energy_raw")}
        v_ot_all, v_raw_all, zt_all, t_all = [], [], [], []

        with torch.no_grad():
            for k in range(0, len(Z) - args.batch_size + 1, args.batch_size):
                z_d = Z[k:k + args.batch_size].to(device)
                B = z_d.shape[0]
                noise = torch.randn(z_d.shape, device=device, generator=gen)
                noise_ot = ot_coupling(z_d, noise)

                for tag, nz in (("ot", noise_ot), ("raw", noise)):
                    v = (nz - z_d).reshape(B, -1)
                    acc[f"energy_{tag}"].append(float(v.pow(2).sum(1).mean()))
                    (v_ot_all if tag == "ot" else v_raw_all).append(v.cpu())

                # model loss, averaged over timestep draws
                ms, en = [], []
                for _ in range(args.t_samples):
                    t = torch.rand(B, device=device, generator=gen)
                    z_t = (1 - t).view(B, 1, 1) * z_d + t.view(B, 1, 1) * noise_ot
                    v_t = noise_ot - z_d
                    ms.append(float((fm(z_t, t) - v_t).pow(2).mean()))
                    en.append(float(v_t.reshape(B, -1).pow(2).sum(1).mean()))
                    # keep (z_t, v, t) for the linear-oracle reference
                    zt_all.append(z_t.reshape(B, -1).cpu())
                    t_all.append(t.cpu())
                acc["mse_ot"].append(float(np.mean(ms)))
                acc["energy_ot"][-1] = float(np.mean(en))

        v_ot = torch.cat(v_ot_all).double()
        v_raw = torch.cat(v_raw_all).double()
        n_elem = v_ot.shape[1]                       # D * T_lat
        # NOTE: F.mse_loss averages over elements, so the target must be expressed
        # per element too -- comparing a mean-MSE against a SUMMED variance is a
        # units error that inflates R^2 to 1.
        energy_ot = float(np.mean(acc["energy_ot"])) / n_elem      # per element
        energy_raw = float(np.mean(acc["energy_raw"])) / n_elem
        var_ot = float(v_ot.var(dim=0, unbiased=True).sum()) / n_elem
        var_raw = float(v_raw.var(dim=0, unbiased=True).sum()) / n_elem
        mse = float(np.mean(acc["mse_ot"]))

        # ---- reference floors on the same batches ----
        # zero predictor: MSE of predicting v = 0
        zero_mse = energy_ot
        # linear oracle: best scalar per timestep bin predicting v from z_t.
        # Zt holds t_samples copies of each batch, so expand v_ot to match.
        B = args.batch_size
        n_batches = len(v_ot) // B
        v_aligned = torch.cat([v_ot[k * B:(k + 1) * B].repeat(args.t_samples, 1)
                               for k in range(n_batches)])
        Zt = torch.cat(zt_all).double()[:len(v_aligned)]
        Tt = torch.cat(t_all).double()[:len(v_aligned)]
        v_aligned = v_aligned[:len(Zt)]
        oracle, bins = 0.0, torch.linspace(0, 1, 11)
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (Tt >= lo) & ((Tt < hi) if hi < 1 else (Tt <= hi))
            if int(m.sum()) < 100:
                continue
            zt_b, v_b = Zt[m], v_aligned[m]
            a = float((v_b * zt_b).sum() / (zt_b * zt_b).sum())     # LS scalar per bin
            oracle += float((v_b - a * zt_b).pow(2).mean()) * float(m.double().mean())

        report[lab] = {
            "fm_checkpoint": str(path),
            "ae_checkpoint": ck["ae_checkpoint"],
            "logged_val_loss": float(ck.get("val_loss")),
            "replicated_mse": mse,
            "target_energy_ot": energy_ot,
            "target_energy_uncoupled": energy_raw,
            "ot_energy_reduction_pct": 100.0 * (1 - energy_ot / energy_raw),
            "var_target_ot": var_ot,
            "var_target_uncoupled": var_raw,
            "zero_predictor_mse": zero_mse,
            "linear_oracle_mse": oracle,
            "relative_loss_ot": mse / var_ot,
            "r2_ot": 1.0 - mse / var_ot,
            "r2_vs_oracle": 1.0 - mse / oracle if oracle else None,
            "n_latents": int(len(Z)),
            "n_elements_per_sample": int(n_elem),
        }
        print(f"  replicated RF MSE       : {mse:.4f}  (logged {ck.get('val_loss'):.4f})")
        print(f"  target mean energy/elem : {energy_ot:.4f}   (uncoupled {energy_raw:.4f})  "
              f"-> OT reduces it by {report[lab]['ot_energy_reduction_pct']:.2f}%")
        print(f"  target variance/elem    : {var_ot:.4f}   zero-predictor MSE = {zero_mse:.4f}")
        print(f"  linear oracle MSE       : {oracle:.4f}  (best per-timestep scalar on z_t)")
        print(f"  RELATIVE loss (MSE/var) : {mse / var_ot:.4f}   R^2 = {1 - mse / var_ot:.4f}   "
              f"R^2 vs oracle = {1 - mse / oracle:.4f}")

    A, B = args.label_a, args.label_b
    ra, rb = report[A], report[B]
    print(f"\n{'=' * 100}")
    print("Is the raw RF val loss comparable across these two latent spaces?")
    print(f"{'=' * 100}")
    print(f"{'quantity':<34}{A:>20}{B:>20}{'ratio A/B':>16}")
    print("-" * 100)
    for key, name in [("logged_val_loss", "logged val loss"),
                      ("replicated_mse", "replicated RF MSE"),
                      ("target_energy_ot", "target energy/elem (OT)"),
                      ("target_energy_uncoupled", "target energy/elem (no OT)"),
                      ("ot_energy_reduction_pct", "OT energy reduction (%)"),
                      ("var_target_ot", "target variance/elem"),
                      ("zero_predictor_mse", "zero-predictor MSE"),
                      ("linear_oracle_mse", "linear-oracle MSE"),
                      ("relative_loss_ot", "RELATIVE loss MSE/var"),
                      ("r2_ot", "R^2 of velocity field"),
                      ("r2_vs_oracle", "R^2 vs linear oracle")]:
        va, vb = ra[key], rb[key]
        ratio = f"{va / vb:.3f}" if vb else "--"
        print(f"{name:<34}{va:>20.4f}{vb:>20.4f}{ratio:>16}")
    print("-" * 100)
    if ra["relative_loss_ot"] < rb["relative_loss_ot"]:
        print(f"Relative loss: {A} is BETTER ({ra['relative_loss_ot']:.4f} vs "
              f"{rb['relative_loss_ot']:.4f}) - same direction as the raw MSE.")
    else:
        print(f"Relative loss: {B} is BETTER ({rb['relative_loss_ot']:.4f} vs "
              f"{ra['relative_loss_ot']:.4f}) - the OPPOSITE direction to the raw MSE.")
    eng_ratio = ra["target_energy_ot"] / rb["target_energy_ot"]
    if abs(eng_ratio - 1.0) < 0.02:
        print(f"Target energy per element is equal to within {abs(eng_ratio-1)*100:.2f}%, so the "
              "raw MSE comparison IS apples-to-apples.")
    else:
        print(f"Target energy differs by {abs(eng_ratio-1)*100:.1f}% between latent spaces, so the "
              "raw MSE comparison is NOT apples-to-apples; use the relative loss.")

    (out / "loss_normalization.json").write_text(json.dumps(
        {"models": report, "protocol": {
            "num_clips": int(min(args.num_clips, len(stems))),
            "batch_size": args.batch_size, "t_samples": args.t_samples,
            "ot_coupling": "toy_training.train_fm_pilot.ot_coupling (Sinkhorn, 20 iters)",
            "seed": args.seed}}, indent=2))
    print(f"\nJSON: {out / 'loss_normalization.json'}")


if __name__ == "__main__":
    main()
