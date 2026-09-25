#!/usr/bin/env python3
"""
SI-SDR + MS-STFT reconstruction comparison between toy autoencoder checkpoints.

Computes per-clip SI-SDR (scale-invariant signal-to-distortion ratio) and
MS-STFT reconstruction loss, then reports mean ± std and a paired t-test.

Usage:
    python toy_evaluation/eval_recon_sisdr.py \
        --checkpoints \
            toy_training/checkpoints/transformer_univ_warm_v2_best.pt \
            toy_training/checkpoints/univ_cycle_enc_best.pt \
        --num_clips 500 --device cuda:3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper, detect_model_type
from toy_training.losses import SAMELoss


def si_sdr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """SI-SDR in dB. ref, est: 1D float arrays of equal length. Higher = better."""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(ref, est) / (np.dot(ref, ref) + 1e-8))
    target = alpha * ref
    noise = est - target
    val = 10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8))
    return float(val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", nargs="+",
                   default=["/data/wav48_silence_trimmed",
                            "/data/fsd50k/FSD50K.dev_audio",
                            "/data/fma_small"])
    p.add_argument("--num_clips", type=int, default=500)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--output_dir", default="toy_evaluation/eval_recon_sisdr")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    clip_samples = sr * 10  # 10-second clips (standard eval length)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Loading {args.num_clips} validation clips (seed {args.seed}, sr {sr})")
    stems = collect_validation_stems(args.data_dirs, sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(stems), size=min(args.num_clips, len(stems)), replace=False)
    print(f"[Setup] {len(stems)} validation stems, using {len(indices)} clips")

    # Load models once
    wrappers = {}
    loss_fn = SAMELoss().to(device)
    for ck_path in args.checkpoints:
        name = Path(ck_path).stem
        wrappers[name] = ToyModelWrapper(ck_path, device)
    print(f"[Setup] Loaded {len(wrappers)} models")

    results = {name: {"sisdr": [], "msstft": [], "lsd": []} for name in wrappers}

    for i, idx in enumerate(indices):
        pt_path, stem_name, track_id = stems[idx]
        audio_np = load_stem_clip(pt_path, clip_samples, centre=True).squeeze().numpy()
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)

        for name, w in wrappers.items():
            z = w.encode(audio_np)  # (1, D, T_lat)
            recon_np = w.decode(z, clip_samples)  # (T,)
            recon = torch.from_numpy(recon_np).unsqueeze(0).unsqueeze(0).to(device)

            msstft = loss_fn(recon, x, use_phase=False).item()
            sisdr = si_sdr_db(audio_np, recon_np)

            # LSD (2048 FFT, 512 hop, Hann)
            X = torch.stft(x.flatten(), 2048, 512, return_complex=True,
                           window=torch.hann_window(2048, device=device))
            R = torch.stft(recon.flatten(), 2048, 512, return_complex=True,
                           window=torch.hann_window(2048, device=device))
            lsd = (20 * (torch.log10(X.abs() + 1e-8) - torch.log10(R.abs() + 1e-8)
                        ).pow(2).mean().sqrt()).item()

            results[name]["sisdr"].append(sisdr)
            results[name]["msstft"].append(msstft)
            results[name]["lsd"].append(lsd)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(indices)}] done")

    # ---- summary ----
    print(f"\n{'='*90}")
    print(f"Reconstruction quality ({len(indices)} clips, seed {args.seed}, sr {sr})")
    print(f"{'='*90}")
    print(f"{'Checkpoint':<45} {'SI-SDR (dB)':<20} {'MS-STFT':<20} {'LSD (dB)':<20}")
    print("-"*105)
    summary = {}
    for name in wrappers:
        s = np.array(results[name]["sisdr"])
        m = np.array(results[name]["msstft"])
        l = np.array(results[name]["lsd"])
        summary[name] = {
            "sisdr_mean": float(s.mean()), "sisdr_std": float(s.std()),
            "msstft_mean": float(m.mean()), "msstft_std": float(m.std()),
            "lsd_mean": float(l.mean()), "lsd_std": float(l.std()),
            "n_clips": len(indices),
        }
        print(f"{name:<45} {s.mean():.3f}±{s.std():.3f}      {m.mean():.4f}±{m.std():.4f}      {l.mean():.3f}±{l.std():.3f}")

    # ---- paired t-test (all pairs if >2 models, else just the pair) ----
    names = list(wrappers.keys())
    if len(names) >= 2:
        from scipy import stats
        print(f"\nPaired t-tests (n={len(indices)}):")
        summary["paired_tests"] = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                na, nb = names[i], names[j]
                sa, sb = np.array(results[na]["sisdr"]), np.array(results[nb]["sisdr"])
                t_sisdr, p_sisdr = stats.ttest_rel(sa, sb)
                ma, mb = np.array(results[na]["msstft"]), np.array(results[nb]["msstft"])
                t_msstft, p_msstft = stats.ttest_rel(ma, mb)
                print(f"  {na} vs {nb}:")
                print(f"    SI-SDR:  Δ={sb.mean()-sa.mean():+.3f} dB, t={t_sisdr:.2f}, p={p_sisdr:.2e}")
                print(f"    MS-STFT: Δ={mb.mean()-ma.mean():+.4f}, t={t_msstft:.2f}, p={p_msstft:.2e}")
                summary["paired_tests"].append({
                    "pair": f"{na} vs {nb}",
                    "si_sdr_delta": float(sb.mean()-sa.mean()), "si_sdr_t": float(t_sisdr), "si_sdr_p": float(p_sisdr),
                    "msstft_delta": float(mb.mean()-ma.mean()), "msstft_t": float(t_msstft), "msstft_p": float(p_msstft),
                })

    out_file = out / "recon_sisdr_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
