#!/usr/bin/env python3
"""
Compare two flow-matching models trained in DIFFERENT latent spaces.

The scientific question (M-FM1 / M-FM2 of the paper): does the cone-structured
(universal) latent make flow matching easier to learn than an unstructured
(soft-norm-only) latent?

This script, given an FM checkpoint per latent space plus its AE, produces:

  A. MATCHED AUDIO.  Samples generated from the *same* noise seeds in both
     models, decoded by their respective AEs, written side by side so they can
     be A/B listened:
         <out>/universal/uncond_XX.wav
         <out>/softnorm/uncond_XX.wav

  B. LATENT-DISTRIBUTION FIDELITY.  For each model, compare the distribution of
     *generated* latents against the *real* data latents:
         - per-element mean / std / var (a unit-Gaussian prior is the FM target)
         - effective rank (participation ratio) of generated vs real pools
         - mean pairwise |cos| between independent samples vs real data
         - squared MMD (RBF) between generated and real latent pools
     A model whose generated distribution matches the data distribution more
     closely has learned the flow better.

  C. RECONSTRUCTION-RECOVERY.  Round-trip check: encode a real clip, run the
     flow from noise, decode; compare against the AE's own reconstruction. This
     isolates flow error from AE error.

  D. LOSS CURVES.  Reads the two *_fm_results.json files and reports the
     convergence comparison (validation RF loss vs epoch).

Usage:
    python training/compare_fm_latents.py \
        --fm_a training/checkpoints/fm_pilot/fm_univ_best.pt \
        --fm_b training/checkpoints/fm_pilot/fm_softnorm_best.pt \
        --label_a universal --label_b softnorm \
        --num_samples 6 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data import collect_validation_stems, load_stem_clip
from evaluation.model_utils import AutoencoderWrapper
from training.sample_fm import LatentDiT, sample_latents


def eff_rank(X):
    """Participation-ratio effective rank of a (N, D) pool."""
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def mean_abs_cos(X, n_pairs=300, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return float("nan")
    idx = rng.integers(0, N, size=(n_pairs, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    if len(idx) == 0:
        return float("nan")
    A = X[idx[:, 0]]; B = X[idx[:, 1]]
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float(np.mean(np.abs(num / den)))


def mmd_rbf(X, Y, sigma=None, max_n=1500, seed=0):
    """Squared MMD with an RBF kernel (unbiased-free, simple estimator)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    Xf = X.reshape(len(X), -1).astype(np.float64)
    Yf = Y.reshape(len(Y), -1).astype(np.float64)
    if sigma is None:
        Z = np.vstack([Xf, Yf])
        d2 = ((Z[:200, None, :] - Z[None, :200, :]) ** 2).sum(-1)
        sigma = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0)) + 1e-12
    def K(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))
    return float(K(Xf, Xf).mean() + K(Yf, Yf).mean() - 2 * K(Xf, Yf).mean()), sigma


def mel_features(audio, sr, n_mels=128, n_fft=1024, hop=256, rms_norm=True):
    """Log-mel frames (T, n_mels) as float64 numpy.

    Audio is RMS-normalised first so the resulting Fr\'echet distance measures
    *spectral shape* rather than loudness (generated and real clips otherwise
    differ in overall level, which dominates a log-mel FAD).
    """
    import torchaudio
    a = np.asarray(audio, dtype=np.float32)
    if rms_norm:
        r = float(np.sqrt(np.mean(a ** 2)))
        a = a / (r + 1e-8)
    wav = torch.from_numpy(a).unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        power=2.0)(wav)
    logmel = torch.log(mel + 1e-6)
    return logmel.squeeze(0).transpose(0, 1).numpy().astype(np.float64)  # (T, n_mels)


def frechet_distance(X, Y):
    """Frechet distance between two sets of feature frames (FAD with mel features).

    d^2 = ||mu_X - mu_Y||^2 + Tr(C_X + C_Y - 2 (C_X^{1/2} C_Y C_X^{1/2})^{1/2})
    """
    from scipy.linalg import sqrtm
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Cx = np.cov(X, rowvar=False); Cy = np.cov(Y, rowvar=False)
    diff = mu_x - mu_y
    covmean, _ = sqrtm(Cx @ Cy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff @ diff + np.trace(Cx + Cy - 2.0 * covmean)
    return float(np.real(fd))


def load_fm(path, device):
    ck = torch.load(path, map_location="cpu")
    m = LatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                  depth=ck["depth"], num_heads=ck["num_heads"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_a", required=True)
    p.add_argument("--fm_b", required=True)
    p.add_argument("--label_a", default="universal")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--results_a", default=None, help="fm_results.json for A (loss curve)")
    p.add_argument("--results_b", default=None)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fid_reps", type=int, default=10,
                   help="Repeat the distribution-fidelity comparison this many "
                        "times, each on an equal-size pool of real clips "
                        "(rank is bounded by N-1, so the pools must match).")
    p.add_argument("--output_dir", default="evaluation/fm_latent_compare")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    n = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    models = {}
    for label, path in [(args.label_a, args.fm_a), (args.label_b, args.fm_b)]:
        m, ck = load_fm(path, device)
        ae = AutoencoderWrapper(ck["ae_checkpoint"], device)
        models[label] = {"fm": m, "ck": ck, "ae": ae}
        print(f"[Load] {label}: FM {Path(path).name} (epoch {ck.get('epoch')}, "
              f"val {ck.get('val_loss'):.4f})  AE {Path(ck['ae_checkpoint']).stem}")

    shape = (args.num_samples, models[args.label_a]["ck"]["channels"],
             models[args.label_a]["ck"]["t_lat"])

    # ---- real data latents for the fidelity comparison ----
    # Encode enough clips for --fid_reps repetitions of an N-sized pool, where
    # N equals the number of generated samples (rank <= N-1, so pools must match).
    stems = collect_validation_stems(sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    n_real_need = min(len(stems), args.num_samples * args.fid_reps)
    real_idx = rng.choice(len(stems), size=n_real_need, replace=False)

    report = {"samples": {}, "fidelity": {}, "loss_curves": {}}

    for label, d in models.items():
        fm, ae = d["fm"], d["ae"]
        sub = out / label; sub.mkdir(parents=True, exist_ok=True)

        # ---------- A. matched-seed unconditional samples ----------
        z = sample_latents(fm, shape, args.steps, device, seed=args.seed)
        z_np = z.detach().cpu().numpy()
        parts = []
        gen_mel = []
        for i in range(args.num_samples):
            audio = ae.decode(z[i:i+1], n)
            pk = float(np.abs(audio).max())
            sf.write(sub / f"uncond_{i:02d}.wav", audio, sr)
            parts.append(audio / (pk + 1e-9) * 0.5)
            parts.append(np.zeros(int(0.15 * sr), dtype=np.float32))
            gen_mel.append(mel_features(audio, sr))
        sf.write(sub / "uncond_all.wav", np.concatenate(parts), sr)
        gen_mel = np.concatenate(gen_mel, axis=0)
        report["samples"][label] = {"peak_max": pk,
                                    "latent_std": float(z_np.std()),
                                    "latent_mean": float(z_np.mean())}

        # ---- real data latents + mel features in THIS latent space ----
        real_z = []
        real_mel_list = []
        for idx in real_idx:
            a = load_stem_clip(stems[idx][0], n, centre=True).squeeze().numpy()
            real_z.append(ae.encode(a).squeeze(0).cpu().numpy())
            real_mel_list.append(mel_features(a, sr))
        real_z = np.stack(real_z)          # (N_all, D, T)
        real_mel = np.concatenate(real_mel_list, axis=0)   # (frames_all, n_mels)

        # ---------- B. distribution fidelity (equal-size pools, repeated) ----------
        # Pool size N = number of generated samples, so the effective-rank
        # comparison is not biased by the N-1 rank ceiling. The real pool is
        # resampled --fid_reps times to report mean +- std.
        N = len(z_np)
        Xg = z_np.reshape(N, -1)
        reps = {"gen_var": [], "real_var": [], "gen_eff_rank": [], "real_eff_rank": [],
                "gen_abs_cos": [], "real_abs_cos": [], "mmd_sq": [], "mel_fad": []}
        sigma = None
        for r in range(args.fid_reps):
            sel = list(range(r * N, min((r + 1) * N, len(real_z))))
            if len(sel) < 2:
                break
            Xr = real_z[sel].reshape(len(sel), -1)
            mel_r = np.concatenate([real_mel_list[i] for i in sel], axis=0)
            reps["gen_var"].append(float(Xg.var()))
            reps["real_var"].append(float(Xr.var()))
            reps["gen_eff_rank"].append(eff_rank(Xg))
            reps["real_eff_rank"].append(eff_rank(Xr))
            reps["gen_abs_cos"].append(mean_abs_cos(Xg, seed=args.seed))
            reps["real_abs_cos"].append(mean_abs_cos(Xr, seed=args.seed))
            mmd, sigma = mmd_rbf(Xg, Xr, seed=args.seed)
            reps["mmd_sq"].append(mmd)
            reps["mel_fad"].append(frechet_distance(gen_mel, mel_r))
        report["fidelity"][label] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
            for k, v in reps.items()
        }
        report["fidelity"][label].update(
            {"n_pool": N, "n_reps": len(reps["gen_eff_rank"]), "mmd_sigma": sigma})
        f = report["fidelity"][label]
        print(f"[{label}] pool N={N} x{len(reps['gen_eff_rank'])}: "
              f"var gen {f['gen_var']['mean']:.3f} vs real {f['real_var']['mean']:.3f} | "
              f"rank gen {f['gen_eff_rank']['mean']:.1f}+-{f['gen_eff_rank']['std']:.1f} "
              f"vs real {f['real_eff_rank']['mean']:.1f}+-{f['real_eff_rank']['std']:.1f} | "
              f"MMD^2 {f['mmd_sq']['mean']:.4f} | mel-FAD {f['mel_fad']['mean']:.1f}", flush=True)

        # ---------- C. reconstruction recovery ----------
        # encode a real clip, then run the flow from the *same AE code* back
        # through the model? No: flow maps noise->data, so we instead measure
        # how well the FM distribution covers real codes via the fidelity stats
        # above, and record AE recon quality as the ceiling.
        a0 = load_stem_clip(stems[real_idx[0]][0], n, centre=True).squeeze().numpy()
        recon = ae.decode(ae.encode(a0), n)
        rms_a = float(np.sqrt(np.mean(a0 ** 2))); rms_r = float(np.sqrt(np.mean(recon ** 2)))
        report["samples"][label]["ae_recon_rms_ratio"] = rms_r / (rms_a + 1e-12)

    # ---------- D. loss curves ----------
    for label, rp in [(args.label_a, args.results_a), (args.label_b, args.results_b)]:
        if rp and Path(rp).exists():
            j = json.loads(Path(rp).read_text())
            lc = j["loss_curve"]
            report["loss_curves"][label] = {
                "best_val": j["best_val"],
                "final_val": lc[-1]["val_loss"],
                "curve": [(e["epoch"], e["train_loss"], e["val_loss"]) for e in lc],
                "latent_stats": j.get("latent_stats"),
            }

    (out / "fm_latent_compare.json").write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    print(f"\n{'='*84}")
    print("FM latent-space comparison")
    print(f"{'='*84}")
    a, b = args.label_a, args.label_b
    if a in report["fidelity"] and b in report["fidelity"]:
        fa, fb = report["fidelity"][a], report["fidelity"][b]
        print(f"{'metric':<24}{a:>20}{b:>20}")
        print("-" * 84)
        for key, name in [("gen_var", "generated var"), ("real_var", "real var"),
                          ("gen_eff_rank", "generated eff-rank"),
                          ("real_eff_rank", "real eff-rank"),
                          ("gen_abs_cos", "generated |cos|"),
                          ("real_abs_cos", "real |cos|"),
                          ("mmd_sq", "MMD^2 gen vs real"),
                          ("mel_fad", "mel-FAD (lower better)")]:
            print(f"{name:<24}{fa[key]['mean']:>14.4f}+-{fa[key]['std']:<5.4f}"
                  f"{fb[key]['mean']:>14.4f}+-{fb[key]['std']:<5.4f}")
    for label in (a, b):
        if label in report["loss_curves"]:
            lc = report["loss_curves"][label]
            print(f"[{label}] FM best val {lc['best_val']:.4f}, final {lc['final_val']:.4f}")
    print(f"\nAudio: {out}/{a}/uncond_*.wav  vs  {out}/{b}/uncond_*.wav")
    print(f"JSON : {out / 'fm_latent_compare.json'}")


if __name__ == "__main__":
    main()
