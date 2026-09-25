#!/usr/bin/env python3
"""
Latent distribution + reconstruction comparison for the encoder-alignment study.

Measures the FM-compatibility properties (latent variance, pair cosine,
norm ratio) + reconstruction (SI-SDR, MS-STFT, LSD) on 10-second clips, for
the three encoder-alignment configurations:
  - univ_warm (decoder-only, reference, FM-compatible)
  - cycle (cycle-consistency, best encoder homomorphism)
  - enc_anneal (annealed encoder cone + boosted KL)

Usage:
    python toy_evaluation/eval_enc_alignment_dist.py \
        --checkpoints ckpt1 ckpt2 ckpt3 \
        --num_clips 500 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss


def si_sdr_db(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    alpha = float(np.dot(ref, est) / (np.dot(ref, ref) + 1e-8))
    target = alpha * ref; noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", nargs="+",
                   default=["/data/wav48_silence_trimmed",
                            "/data/fsd50k/FSD50K.dev_audio",
                            "/data/fma_small"])
    p.add_argument("--num_clips", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--output_dir", default="toy_evaluation/eval_enc_alignment_dist")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    clip_samples = sr * 10  # 10-second clips
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] {args.num_clips} clips, seed {args.seed}, sr {sr}, {clip_samples/sr:.0f}s clips", flush=True)
    stems = collect_validation_stems(args.data_dirs, sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(stems), size=min(args.num_clips, len(stems)), replace=False)
    print(f"[Setup] {len(stems)} stems, using {len(indices)} clips", flush=True)

    loss_fn = SAMELoss().to(device)
    wrappers = {}
    for ck_path in args.checkpoints:
        wrappers[Path(ck_path).stem] = ToyModelWrapper(ck_path, device)
    print(f"[Setup] Loaded {len(wrappers)} models", flush=True)

    results = {name: {"sisdr": [], "msstft": [], "lsd": [], "var": [], "cos": [], "norm_ratio": []}
               for name in wrappers}

    for i, idx in enumerate(indices):
        # Load clip A
        pt_path_a, _, _ = stems[idx]
        audio_a = load_stem_clip(pt_path_a, clip_samples, centre=True).squeeze().numpy()
        # Load clip B (next index, for pair cosine and norm ratio)
        j = (idx + 1) % len(stems)
        pt_path_b, _, _ = stems[j]
        audio_b = load_stem_clip(pt_path_b, clip_samples, centre=True).squeeze().numpy()

        for name, w in wrappers.items():
            z_a = w.encode(audio_a)
            z_b = w.encode(audio_b)
            # reconstruction
            recon_np = w.decode(z_a, clip_samples)
            recon = torch.from_numpy(recon_np).unsqueeze(0).unsqueeze(0).to(device)
            x = torch.from_numpy(audio_a).unsqueeze(0).unsqueeze(0).to(device)
            msstft = loss_fn(recon, x, use_phase=False).item()
            sisdr = si_sdr_db(audio_a, recon_np)
            # LSD
            X = torch.stft(x.flatten(), 2048, 512, return_complex=True, window=torch.hann_window(2048, device=device))
            R = torch.stft(recon.flatten(), 2048, 512, return_complex=True, window=torch.hann_window(2048, device=device))
            lsd = (20 * (torch.log10(X.abs() + 1e-8) - torch.log10(R.abs() + 1e-8)).pow(2).mean().sqrt()).item()
            # latent distribution
            var = z_a.var().item()
            cos = torch.dot(z_a.flatten(), z_b.flatten()).item() / (torch.linalg.norm(z_a).item() * torch.linalg.norm(z_b).item() + 1e-8)
            # norm ratio: ||trigMix(z_a, z_b)|| / ||z_a|| at alpha=0.5
            theta = math.pi / 4
            z_mix = math.cos(theta) * z_a + math.sin(theta) * z_b
            norm_ratio = torch.linalg.norm(z_mix).item() / (torch.linalg.norm(z_a).item() + 1e-8)
            results[name]["sisdr"].append(sisdr)
            results[name]["msstft"].append(msstft)
            results[name]["lsd"].append(lsd)
            results[name]["var"].append(var)
            results[name]["cos"].append(abs(cos))
            results[name]["norm_ratio"].append(norm_ratio)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(indices)}] done", flush=True)

    # ---- Summary ----
    print(f"\n{'='*100}", flush=True)
    print(f"Encoder alignment: reconstruction + latent distribution ({len(indices)} clips, {clip_samples/sr:.0f}s, seed {args.seed})", flush=True)
    print(f"{'='*100}", flush=True)
    hdr = f"{'Checkpoint':<30} {'SI-SDR':<14} {'MS-STFT':<14} {'LSD':<14} {'Lat Var':<10} {'|cos|':<10} {'NormR':<10}"
    print(hdr, flush=True)
    print("-"*100, flush=True)
    summary = {}
    for name in wrappers:
        s = {k: np.array(v) for k, v in results[name].items()}
        summary[name] = {
            "sisdr": (s["sisdr"].mean(), s["sisdr"].std()),
            "msstft": (s["msstft"].mean(), s["msstft"].std()),
            "lsd": (s["lsd"].mean(), s["lsd"].std()),
            "var": (s["var"].mean(), s["var"].std()),
            "cos": (s["cos"].mean(), s["cos"].std()),
            "norm_ratio": (s["norm_ratio"].mean(), s["norm_ratio"].std()),
            "n_clips": len(indices),
        }
        print(f"{name:<30} {s['sisdr'].mean():.2f}±{s['sisdr'].std():.2f}  "
              f"{s['msstft'].mean():.4f}±{s['msstft'].std():.4f}  "
              f"{s['lsd'].mean():.2f}±{s['lsd'].std():.2f}  "
              f"{s['var'].mean():.3f}   {s['cos'].mean():.3f}   {s['norm_ratio'].mean():.3f}",
              flush=True)

    # ---- Paired t-tests ----
    names = list(wrappers.keys())
    if len(names) >= 2:
        from scipy import stats
        print(f"\nPaired t-tests (n={len(indices)}):", flush=True)
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                na, nb = names[i], names[j]
                for metric in ["sisdr", "msstft", "var", "cos", "norm_ratio"]:
                    a = np.array(results[na][metric]); b = np.array(results[nb][metric])
                    t, pval = stats.ttest_rel(a, b)
                    if pval < 0.05:
                        print(f"  {na} vs {nb} [{metric}]: Δ={b.mean()-a.mean():+.4f}, t={t:.2f}, p={pval:.2e} *", flush=True)

    out_file = out / "enc_alignment_dist_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
