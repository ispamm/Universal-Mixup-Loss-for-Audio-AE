#!/usr/bin/env python3
"""
Controls for the FM round-trip reconstruction eval.

Purpose: decide whether the round-trip audio is *meaningful reconstruction* or
garbage/noise.  MS-STFT alone cannot answer that -- a low MS-STFT could in
principle come from decoding something meaningless.  So the round-trip score is
calibrated against four references, all on the same clips and the same metric:

  1. AE ceiling            ae.decode(ae.encode(x))            best possible
  2. ROUND TRIP (ours)     encode -> FM 0->1 -> FM 1->0 -> decode
  3. WRONG CLIP            MS-STFT(x_i, recon(x_j)), j != i    does the recon carry
                                                               clip identity?
  4. RANDOM LATENT DECODE  z ~ N(0, I) decoded                 "noise" floor
  5. FM UNCONDITIONAL      noise -> FM 1->0 -> decode          what
                                                               compare_fm_latents.py
                                                               produced (generation,
                                                               no conditioning)

Interpretation:
  * meaningful recon  <=>  round trip << wrong clip, and round trip << random latent.
  * if round trip ~= wrong clip ~= random latent, the recon is not clip-specific
    and must be judged meaningless regardless of the absolute MS-STFT value.

Also writes spectrogram PNGs (orig | univ recon | softnorm recon) for eyeballing,
and reports waveform correlation as a supporting (non-MS-STFT) diagnostic.

USAGE
    python toy_evaluation/fm_roundtrip_controls.py \
        --fm_a .../fm_univ_best.pt --fm_b .../fm_softnorm_best.pt \
        --label_a univ --label_b softnorm --num_clips 64 --steps 250
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss
from toy_evaluation.eval_fm_roundtrip_msstft import (
    load_fm, ae_encode, ae_decode, roundtrip, msstft, gain_match)


@torch.no_grad()
def uncond_sample(fm, shape, steps, device):
    """Noise (t=1) -> data (t=0), the compare_fm_latents.py / sample_fm.py path."""
    z = torch.randn(shape, device=device, generator=torch.Generator(device).manual_seed(0))
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        z = z + fm(z, ts[i].expand(shape[0])) * (ts[i + 1] - ts[i])
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_a", required=True)
    p.add_argument("--fm_b", required=True)
    p.add_argument("--label_a", default="univ")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--num_clips", type=int, default=64)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="toy_evaluation/eval_fm_roundtrip/controls")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    n = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    labels = [args.label_a, args.label_b]
    M = {}
    for lab, path in [(args.label_a, args.fm_a), (args.label_b, args.fm_b)]:
        fm, ck = load_fm(path, device)
        M[lab] = {"fm": fm, "ck": ck, "ae": ToyModelWrapper(ck["ae_checkpoint"], device)}
        print(f"[Load] {lab:12s} FM {Path(path).name}  AE {Path(ck['ae_checkpoint']).stem}")

    stems = collect_validation_stems(sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(stems), size=args.num_clips, replace=False)
    print(f"[Data] {args.num_clips} held-out clips, seed {args.seed}, {args.clip_seconds}s")

    loss_fn = SAMELoss().to(device)
    scores = {lab: {k: [] for k in
                    ("own", "wrong", "random_latent", "ceiling", "uncond", "corr")}
              for lab in labels}
    spectro = {}

    for start in range(0, len(idx), args.batch_size):
        bidx = idx[start:start + args.batch_size]
        B = len(bidx)
        x = torch.from_numpy(np.stack(
            [load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy() for i in bidx])
        ).unsqueeze(1).to(device)                                  # (B, 1, T)

        # wrong-clip partner: shift by half the pool (never equal to itself)
        shift = max(1, len(idx) // 2)
        widx = [idx[(start + j + shift) % len(idx)] for j in range(B)]
        x_wrong = torch.from_numpy(np.stack(
            [load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy() for i in widx])
        ).unsqueeze(1).to(device)

        for lab in labels:
            fm, ae = M[lab]["fm"], M[lab]["ae"]
            z0 = ae_encode(ae, x)
            z_hat, _ = roundtrip(fm, z0, args.steps)
            x_hat = ae_decode(ae, z_hat, n)
            x_ceil = ae_decode(ae, z0, n)

            z_rand = torch.randn_like(z0)
            x_rand = ae_decode(ae, z_rand, n)

            z_u = uncond_sample(fm, (B,) + tuple(z0.shape[1:]), args.steps, device)
            x_u = ae_decode(ae, z_u, n)

            scores[lab]["own"] += msstft(loss_fn, x_hat, x).cpu().tolist()
            scores[lab]["wrong"] += msstft(loss_fn, x_hat, x_wrong).cpu().tolist()
            scores[lab]["random_latent"] += msstft(loss_fn, x_rand, x).cpu().tolist()
            scores[lab]["ceiling"] += msstft(loss_fn, x_ceil, x).cpu().tolist()
            scores[lab]["uncond"] += msstft(loss_fn, x_u, x).cpu().tolist()

            # supporting diagnostic: waveform Pearson r on the 3s centre
            c = n // 2
            sl = slice(c, c + 3 * sr)
            xs = x[:, 0, sl].cpu().numpy()
            hs = x_hat[:, 0, sl].cpu().numpy()
            r = [float(np.corrcoef(xs[j], hs[j])[0, 1]) for j in range(B)]
            scores[lab]["corr"] += r

            if start == 0:
                spectro[lab] = {"x": x[:, 0].cpu().numpy(),
                                "x_hat": x_hat[:, 0].cpu().numpy()} 

        print(f"  [{min(start + B, len(idx))}/{len(idx)}]", flush=True)

    # ---- summary ----
    def stats(v):
        v = np.array(v, np.float64)
        return float(v.mean()), float(v.std()), float(np.median(v))

    summary = {}
    print(f"\n{'=' * 100}")
    print("Is the round-trip reconstruction meaningful?  (MS-STFT, lower = better)")
    print(f"{'=' * 100}")
    print(f"{'model':<10}{'AE ceiling':>12}{'ROUND TRIP':>13}{'wrong clip':>13}"
          f"{'random lat':>13}{'uncond gen':>13}{'wav r':>9}")
    print("-" * 100)
    for lab in labels:
        s = scores[lab]
        row = {k: stats(v) for k, v in s.items() if k != "corr"}
        summary[lab] = {f"{k}_mean": row[k][0] for k in row}
        summary[lab].update({f"{k}_median": row[k][2] for k in row})
        summary[lab]["corr_mean"] = stats(s["corr"])[0]
        summary[lab]["n_clips"] = len(s["own"])
        print(f"{lab:<10}{row['ceiling'][0]:>12.3f}{row['own'][0]:>13.3f}"
              f"{row['wrong'][0]:>13.3f}{row['random_latent'][0]:>13.3f}"
              f"{row['uncond'][0]:>13.3f}{summary[lab]['corr_mean']:>9.3f}")

    print("-" * 100)
    for lab in labels:
        s = scores[lab]
        own = np.array(s["own"]); wrong = np.array(s["wrong"])
        rand = np.array(s["random_latent"]); ceil_ = np.array(s["ceiling"])
        print(f"  {lab}:")
        print(f"    own recon vs wrong-clip gap : {wrong.mean() - own.mean():+.3f} "
              f"({wrong.mean() / own.mean():.2f}x)   own<wrong on "
              f"{int((own < wrong).sum())}/{len(own)} clips")
        print(f"    own recon vs random-latent   : {rand.mean() - own.mean():+.3f} "
              f"({rand.mean() / own.mean():.2f}x)")
        print(f"    own recon above AE ceiling   : {own.mean() - ceil_.mean():+.3f} "
              f"({own.mean() / ceil_.mean():.2f}x)")
        print(f"    mean waveform r (3s centre)  : {np.mean(s['corr']):+.3f}")

    # ---- spectrograms ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for lab in labels:
            for j in range(min(3, args.num_clips)):
                fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
                for ax, sig, ttl in [
                        (axes[0], spectro[lab]["x"][j], "original"),
                        (axes[1], spectro[lab]["x_hat"][j], f"{lab} round-trip recon")]:
                    ax.specgram(sig, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
                    ax.set_title(ttl, fontsize=9); ax.set_ylabel("Hz", fontsize=8)
                axes[2].specgram(spectro[labels[0]]["x_hat"][j] if lab != labels[0]
                                 else spectro[labels[1]]["x_hat"][j],
                                 NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
                axes[2].set_title("other model's recon", fontsize=9)
                axes[2].set_ylabel("Hz", fontsize=8); axes[2].set_xlabel("s")
                fig.suptitle(f"clip{j:02d} ({lab} comparison)", fontsize=10)
                fig.tight_layout()
                fig.savefig(out / f"spec_clip{j:02d}_{lab}.png", dpi=110)
                plt.close(fig)
        print(f"\nSpectrograms: {out}/spec_clip*.png")
    except Exception as e:                                     # pragma: no cover
        print(f"[warn] spectrogram PNGs skipped: {e}")

    (out / "controls.json").write_text(json.dumps(
        {"summary": summary,
         "per_clip": {lab: {k: [float(x) for x in v] for k, v in scores[lab].items()}
                      for lab in labels},
         "protocol": {"metric": "MS-STFT (SAMELoss magnitude, use_phase=False)",
                      "steps": args.steps, "num_clips": args.num_clips,
                      "clip_seconds": args.clip_seconds, "seed": args.seed,
                      "wrong_clip_partner": "index shifted by num_clips//2",
                      "uncond_sample_seed": 0}}, indent=2))
    print(f"JSON : {out / 'controls.json'}")


if __name__ == "__main__":
    main()
