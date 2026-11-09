#!/usr/bin/env python3
"""
44.1 kHz AE reconstruction + interpolation comparison
=========================================================

Compares reconstruction quality of two 44.1 kHz checkpoints:

  * ``univ_dt2048_441_best.pt``     — patch_size=2048 (aggressive downsampling)
  * ``transformer_univ_441_best.pt`` — patch_size=256  (normal resolution)

Metrics (per model)
------------------
  **Single-audio reconstruction** (1000 clips, recon vs original):
    - MS-STFT  : training-matched ``SAMELoss`` magnitude loss (spectral
                 contrast + adaptive log-magnitude, summed over the 7 FFT
                 resolutions 32→2048).  Lower = better.
    - SI-SDR   : scale-invariant SDR in dB (Le Roux et al. 2019).  Higher = better.

  **Total interpolation path** (1000 random pairs, seed 42):
    For each pair (A, B) and each α ∈ np.linspace(0, 1, 9):
      latent path   z_interp = interp(Z_A, Z_B, α)   [linear / trigonometric]
      decoded audio recon = decode(z_interp)
      ideal audio mix    mix = ideal(A, B, α)        [lerp / constant-power]
      metric(α) = MS-STFT(mix, recon)  and  SI-SDR(mix, recon)
    The **path mean** = mean over the 9 α points (one number per path),
    then averaged over the 1000 pairs (reported as mean ± std).

Data
----
Uses the shared ``evaluation.data`` layer at ``sample_rate=44100`` so the
validation stems are exactly the held-out tracks of the training split (VCTK +
FSD50K + FMA-small), loaded from the preprocessed ``sr44100`` cache.

Audio saving
------------
A small number of reconstructed clips (and a couple of interpolation paths) are
written as 44.1 kHz mono .wav per model under ``--audio_out`` so the
reconstruction quality can be auditioned.

Usage
-----
    # one model per process (run the two in parallel)
    python evaluation/eval_recon_interp_441.py \
        --ckpt training/checkpoints/univ_dt2048_441_best.pt \
        --num_clips 1000 --num_pairs 1000 \
        --out evaluation/results_441/dt2048.json \
        --audio_out evaluation/audio_441/dt2048 \
        --label dt2048
    python evaluation/eval_recon_interp_441.py \
        --ckpt training/checkpoints/transformer_univ_441_best.pt \
        --num_clips 1000 --num_pairs 1000 \
        --out evaluation/results_441/normal.json \
        --audio_out evaluation/audio_441/normal \
        --label normal

Then aggregate:
    python evaluation/eval_recon_interp_441.py --aggregate \
        evaluation/results_441/dt2048.json \
        evaluation/results_441/normal.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import (
    CLIP_DURATION,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss

# 44.1 kHz constants
SR44 = 44100
CLIP_SAMPLES = int(CLIP_DURATION * SR44)  # 88200
ALPHAS = np.linspace(0.0, 1.0, 9).tolist()  # full path incl. endpoints


# =============================================================================
#  Metrics
# =============================================================================

def ms_stft_loss(
    pred_np: np.ndarray,
    tgt_np: np.ndarray,
    loss_fn: SAMELoss,
    device: torch.device,
) -> float:
    """Training-matched multi-resolution STFT magnitude loss (SAMELoss mag)."""
    pred = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    tgt = torch.from_numpy(tgt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, mag = loss_fn(pred, tgt, use_phase=False, return_components=True)
    return float(mag.item())


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019). Higher = better."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = np.dot(ref, ref)
    if denom < 1e-12:
        return float("nan")
    alpha = np.dot(est, ref) / denom
    e_target = alpha * ref
    e_noise = est - e_target
    snr = np.dot(e_target, e_target) / (np.dot(e_noise, e_noise) + 1e-12)
    return float(10.0 * np.log10(snr))


# =============================================================================
#  Interpolation
# =============================================================================

def lerp(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear latent interpolation."""
    return (1.0 - alpha) * z_a + alpha * z_b


def trig_mix(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Trigonometric (constant-power) latent interpolation."""
    theta = alpha * (np.pi / 2.0)
    return math.cos(theta) * z_a + math.sin(theta) * z_b


def ideal_lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def ideal_trig(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    theta = alpha * (math.pi / 2.0)
    return math.cos(theta) * a + math.sin(theta) * b


# =============================================================================
#  Audio saving
# =============================================================================

def save_wav(path: Path, audio: np.ndarray, sr: int = SR44) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(audio, -1.0, 1.0) * 32767.0
    wavfile.write(str(path), sr, arr.astype(np.int16))


def _align(recon: np.ndarray, target_len: int) -> np.ndarray:
    """Crop / pad reconstructed audio to the target length."""
    if len(recon) > target_len:
        return recon[:target_len]
    elif len(recon) < target_len:
        return np.pad(recon, (0, target_len - len(recon)))
    return recon


# =============================================================================
#  Per-model evaluation
# =============================================================================

def evaluate_model(
    ckpt_path: str,
    device: torch.device,
    *,
    num_clips: int,
    num_pairs: int,
    seed: int,
    audio_out: Optional[Path],
    n_save_recon: int,
    n_save_pairs: int,
    label: Optional[str] = None,
    skip_recon: bool = False,
) -> Dict:
    """Run reconstruction + interpolation battery for ONE checkpoint.

    Returns a dict with mean/std MS-STFT and SI-SDR for single reconstruction
    and for the linear / trigonometric interpolation paths (path-mean over the
    9 α points, then mean over pairs).

    If ``skip_recon`` is True, the single-audio reconstruction phase is skipped
    (useful when reconstruction has already been measured separately and only
    the interpolation path is wanted).
    """
    t_start = time.time()
    name = label or Path(ckpt_path).stem
    print(f"\n{'='*72}\n=== {name}  ({ckpt_path})\n{'='*72}")

    stems = collect_validation_stems(sample_rate=SR44)
    print(f"  validation stems @ {SR44} Hz: {len(stems)}")

    model = AutoencoderWrapper(ckpt_path, device)
    print(f"  model: {model.name}  type={model.model_type}  device={device}")

    loss_fn = SAMELoss().to(device)

    # ---- deterministic clip selection & pairing (identical for both models) ----
    import random
    rng = random.Random(seed)
    n_avail = len(stems)
    n_clips = min(num_clips, n_avail)
    clip_indices = rng.sample(range(n_avail), n_clips)
    # pairs drawn from the same seed (generate_pairs uses its own Random(seed))
    pairs = generate_pairs(stems, num_pairs, seed=seed, mode="random")
    print(f"  clips={n_clips}  pairs={len(pairs)}  seed={seed}")

    # -------------------------------------------------------------------------
    #  1. Single-audio reconstruction  (skippable via --skip-recon)
    # -------------------------------------------------------------------------
    msstft_recon: List[float] = []
    sisdr_recon: List[float] = []
    if skip_recon:
        print("  [1/2] Single-audio reconstruction ... SKIPPED (--skip-recon)")
    else:
        print("  [1/2] Single-audio reconstruction ...")
        saved_recon = 0
        for idx in tqdm(clip_indices, desc=f"recon[{name}]", unit="clip"):
            path, stem, track = stems[idx]
            try:
                audio = stem_to_numpy(path, CLIP_SAMPLES, centre=True)
            except Exception as exc:
                print(f"    ⚠ skip {path}: {exc}")
                continue
            n = len(audio)
            z = model.encode(audio)
            recon = _align(model.decode(z, n), n)
            msstft_recon.append(ms_stft_loss(recon, audio, loss_fn, device))
            sisdr_recon.append(si_sdr(audio, recon))

            if audio_out is not None and saved_recon < n_save_recon:
                dom = track.split("::")[0]
                base = f"recon_{saved_recon:03d}_{dom}_{Path(path).stem}"
                save_wav(audio_out / f"{base}_orig.wav", audio)
                save_wav(audio_out / f"{base}_recon.wav", recon)
                saved_recon += 1

    if skip_recon:
        recon_summary = {
            "n": 0,
            "msstft_mean": float("nan"),
            "msstft_std": float("nan"),
            "sisdr_mean": float("nan"),
            "sisdr_std": float("nan"),
            "skipped": True,
        }
    else:
        recon_summary = {
            "n": len(msstft_recon),
            "msstft_mean": float(np.mean(msstft_recon)) if msstft_recon else float("nan"),
            "msstft_std": float(np.std(msstft_recon)) if msstft_recon else float("nan"),
            "sisdr_mean": float(np.nanmean(sisdr_recon)) if sisdr_recon else float("nan"),
            "sisdr_std": float(np.nanstd(sisdr_recon)) if sisdr_recon else float("nan"),
        }
        print(f"  recon: MS-STFT={recon_summary['msstft_mean']:.6f} ± {recon_summary['msstft_std']:.6f} | "
              f"SI-SDR={recon_summary['sisdr_mean']:.2f} ± {recon_summary['sisdr_std']:.2f} dB "
              f"(n={recon_summary['n']})")

    # -------------------------------------------------------------------------
    #  2. Interpolation paths (linear + trigonometric)
    # -------------------------------------------------------------------------
    print("  [2/2] Interpolation paths (linear + trigonometric) ...")
    # per-(pair, α) arrays
    lerp_msstft = np.full((len(pairs), len(ALPHAS)), np.nan)
    lerp_sisdr = np.full((len(pairs), len(ALPHAS)), np.nan)
    trig_msstft = np.full((len(pairs), len(ALPHAS)), np.nan)
    trig_sisdr = np.full((len(pairs), len(ALPHAS)), np.nan)
    saved_pairs = 0

    for p_idx, (i, j) in enumerate(tqdm(pairs, desc=f"interp[{name}]", unit="pair")):
        try:
            a = stem_to_numpy(stems[i][0], CLIP_SAMPLES, centre=True)
            b = stem_to_numpy(stems[j][0], CLIP_SAMPLES, centre=True)
        except Exception as exc:
            print(f"    ⚠ skip pair {p_idx}: {exc}")
            continue
        n = len(a)
        z_a = model.encode(a)
        z_b = model.encode(b)
        if z_a.shape[-1] != z_b.shape[-1]:
            mlen = min(z_a.shape[-1], z_b.shape[-1])
            z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]

        save_this = (audio_out is not None and saved_pairs < n_save_pairs)
        for a_idx, alpha in enumerate(ALPHAS):
            # ---- linear path ----
            zl = lerp(z_a, z_b, alpha)
            rl = _align(model.decode(zl, n), n)
            il = ideal_lerp(a, b, alpha)
            lerp_msstft[p_idx, a_idx] = ms_stft_loss(rl, il, loss_fn, device)
            lerp_sisdr[p_idx, a_idx] = si_sdr(il, rl)

            # ---- trigonometric path ----
            zt = trig_mix(z_a, z_b, alpha)
            rt = _align(model.decode(zt, n), n)
            it = ideal_trig(a, b, alpha)
            trig_msstft[p_idx, a_idx] = ms_stft_loss(rt, it, loss_fn, device)
            trig_sisdr[p_idx, a_idx] = si_sdr(it, rt)

            if save_this and a_idx in (0, 4, 8):  # α = 0.0, 0.5, 1.0
                tag = f"interp_{saved_pairs:03d}_a{int(round(alpha*4))}"
                if a_idx == 4:  # midpoint only: save lerp + trig
                    save_wav(audio_out / f"{tag}_lerp_ideal.wav", il)
                    save_wav(audio_out / f"{tag}_lerp_recon.wav", rl)
                    save_wav(audio_out / f"{tag}_trig_ideal.wav", it)
                    save_wav(audio_out / f"{tag}_trig_recon.wav", rt)
                if a_idx == 0:  # endpoints = reconstructions
                    save_wav(audio_out / f"{tag}_A_orig.wav", a)
                    save_wav(audio_out / f"{tag}_A_recon.wav", model.decode(z_a, n) if False else rl)
                    save_wav(audio_out / f"{tag}_B_orig.wav", b)
                    # B recon at alpha=1:
                    zl_b = lerp(z_a, z_b, 1.0)
                    save_wav(audio_out / f"{tag}_B_recon.wav", _align(model.decode(zl_b, n), n))
        if save_this:
            saved_pairs += 1

    # path-mean over the 9 α points, then mean/std over pairs
    def _path_stats(arr: np.ndarray) -> Dict[str, float]:
        per_pair_mean = np.nanmean(arr, axis=1)  # mean over α per pair
        return {
            "n_pairs": int(np.sum(~np.isnan(per_pair_mean))),
            "path_mean_of_mean": float(np.nanmean(per_pair_mean)),
            "path_std_of_mean": float(np.nanstd(per_pair_mean)),
            # also a flat mean over all (pair, α) cells for reference
            "global_mean": float(np.nanmean(arr)),
            "global_std": float(np.nanstd(arr)),
            # per-α mean curve (mean over pairs)
            "per_alpha_mean": np.nanmean(arr, axis=0).tolist(),
        }

    interp_summary = {
        "alphas": ALPHAS,
        "linear": {
            "msstft": _path_stats(lerp_msstft),
            "sisdr": _path_stats(lerp_sisdr),
        },
        "trigonometric": {
            "msstft": _path_stats(trig_msstft),
            "sisdr": _path_stats(trig_sisdr),
        },
    }
    for path_name in ("linear", "trigonometric"):
        s = interp_summary[path_name]
        print(f"  {path_name:14s} path: MS-STFT(path-mean)={s['msstft']['path_mean_of_mean']:.6f} "
              f"± {s['msstft']['path_std_of_mean']:.6f} | "
              f"SI-SDR(path-mean)={s['sisdr']['path_mean_of_mean']:.2f} "
              f"± {s['sisdr']['path_std_of_mean']:.2f} dB "
              f"(n={s['msstft']['n_pairs']})")

    elapsed = time.time() - t_start
    result = {
        "label": name,
        "checkpoint": ckpt_path,
        "model_type": model.model_type,
        "device": str(device),
        "sample_rate": SR44,
        "clip_duration_s": CLIP_DURATION,
        "clip_samples": CLIP_SAMPLES,
        "alphas": ALPHAS,
        "num_clips_requested": n_clips,
        "num_pairs_requested": len(pairs),
        "seed": seed,
        "reconstruction": recon_summary,
        "interpolation": interp_summary,
        "elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    print(f"  done in {elapsed:.1f}s")
    return result


# =============================================================================
#  CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="44.1 kHz AE reconstruction + interpolation comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Single checkpoint to evaluate (one-model mode).")
    ap.add_argument("--label", type=str, default=None,
                    help="Display label for the checkpoint.")
    ap.add_argument("--num_clips", type=int, default=1000,
                    help="Number of clips for single reconstruction (default 1000).")
    ap.add_argument("--num_pairs", type=int, default=1000,
                    help="Number of random pairs for interpolation (default 1000).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device (default cpu; cuda if available).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (one-model mode).")
    ap.add_argument("--audio_out", type=Path, default=None,
                    help="Directory to save reconstructed .wav for listening.")
    ap.add_argument("--n_save_recon", type=int, default=8,
                    help="Number of reconstructed clips to save (default 8).")
    ap.add_argument("--n_save_pairs", type=int, default=2,
                    help="Number of interpolation paths to save (default 2).")
    ap.add_argument("--skip_recon", action="store_true",
                    help="Skip the single-audio reconstruction phase (only run interpolation).")
    ap.add_argument("--aggregate", nargs="+", default=None,
                    help="Aggregate mode: pass per-model JSON paths to build a comparison table.")
    args = ap.parse_args()

    # ---- aggregate mode ----
    if args.aggregate:
        aggregate(args.aggregate)
        return

    if not args.ckpt or not args.out:
        ap.error("--ckpt and --out are required in one-model mode "
                 "(or use --aggregate <json...>).")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    result = evaluate_model(
        args.ckpt,
        device,
        num_clips=args.num_clips,
        num_pairs=args.num_pairs,
        seed=args.seed,
        audio_out=args.audio_out,
        n_save_recon=args.n_save_recon,
        n_save_pairs=args.n_save_pairs,
        label=args.label,
        skip_recon=args.skip_recon,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj):
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    args.out.write_text(json.dumps(_sanitize(result), indent=2))
    print(f"\n📄 JSON -> {args.out}")


# =============================================================================
#  Aggregation / comparison table
# =============================================================================

def aggregate(json_paths: List[str]) -> None:
    models = []
    for p in json_paths:
        with open(p) as fh:
            models.append(json.load(fh))
    if not models:
        print("No model JSONs found.")
        return

    print("\n" + "=" * 86)
    print("44.1 kHz reconstruction + interpolation — comparison")
    print("=" * 86)
    labels = [m.get("label", Path(m["checkpoint"]).stem) for m in models]
    hdr = f"{'Metric':<40}" + "".join(f"{l[:18]:>20}" for l in labels)
    print(hdr)
    print("-" * 86)

    def _row(label, getter, fmt="{:>20.6f}"):
        r = f"{label:<40}"
        for m in models:
            v = getter(m)
            r += fmt.format(v) if isinstance(v, (int, float)) else f"{'—':>20}"
        print(r)

    # --- single reconstruction ---
    _row("Recon MS-STFT (mean)",            lambda m: m["reconstruction"]["msstft_mean"])
    _row("Recon MS-STFT (std)",             lambda m: m["reconstruction"]["msstft_std"])
    _row("Recon SI-SDR (mean, dB)",         lambda m: m["reconstruction"]["sisdr_mean"], "{:>20.2f}")
    _row("Recon SI-SDR (std, dB)",          lambda m: m["reconstruction"]["sisdr_std"],  "{:>20.2f}")
    print("-" * 86)
    # --- interpolation paths (path-mean over 9 α, then mean over pairs) ---
    for path in ("linear", "trigonometric"):
        _row(f"{path} path MS-STFT (path-mean)",
             lambda m, p=path: m["interpolation"][p]["msstft"]["path_mean_of_mean"])
        _row(f"{path} path MS-STFT (std)",
             lambda m, p=path: m["interpolation"][p]["msstft"]["path_std_of_mean"])
        _row(f"{path} path SI-SDR (path-mean, dB)",
             lambda m, p=path: m["interpolation"][p]["sisdr"]["path_mean_of_mean"], "{:>20.2f}")
        _row(f"{path} path SI-SDR (std, dB)",
             lambda m, p=path: m["interpolation"][p]["sisdr"]["path_std_of_mean"],  "{:>20.2f}")
        print("-" * 86)

    # --- n ---
    n_recon = [m["reconstruction"]["n"] for m in models]
    n_pairs = [m["interpolation"]["linear"]["msstft"]["n_pairs"] for m in models]
    print(f"{'n clips (recon)':<40}" + "".join(f"{n:>20}" for n in n_recon))
    print(f"{'n pairs (interp)':<40}" + "".join(f"{n:>20}" for n in n_pairs))
    print("=" * 86)

    # --- relative difference (model[1] vs model[0]) if exactly 2 ---
    if len(models) == 2:
        a, b = models[0], models[1]
        print(f"\nRelative: [{labels[1]}] vs [{labels[0]}]  (positive = {labels[1]} is better for SI-SDR, worse for MS-STFT)")
        def _delta(label, get_a, get_b, fmt="{:+.4f}", higher_better=True):
            va, vb = get_a(a), get_b(b)
            if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))):
                return
            d = vb - va
            print(f"  {label:<40} {fmt.format(d)}")
        _delta("Recon MS-STFT Δ",
               lambda m: m["reconstruction"]["msstft_mean"],
               lambda m: m["reconstruction"]["msstft_mean"], "{:+.6f}", higher_better=False)
        _delta("Recon SI-SDR Δ (dB)",
               lambda m: m["reconstruction"]["sisdr_mean"],
               lambda m: m["reconstruction"]["sisdr_mean"], "{:+.2f}")
        for path in ("linear", "trigonometric"):
            _delta(f"{path} path MS-STFT Δ",
                   lambda m, p=path: m["interpolation"][p]["msstft"]["path_mean_of_mean"],
                   lambda m, p=path: m["interpolation"][p]["msstft"]["path_mean_of_mean"], "{:+.6f}", higher_better=False)
            _delta(f"{path} path SI-SDR Δ (dB)",
                   lambda m, p=path: m["interpolation"][p]["sisdr"]["path_mean_of_mean"],
                   lambda m, p=path: m["interpolation"][p]["sisdr"]["path_mean_of_mean"], "{:+.2f}")
        print()


if __name__ == "__main__":
    main()
