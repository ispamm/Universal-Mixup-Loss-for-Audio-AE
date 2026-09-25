#!/usr/bin/env python3
"""
Toy AE — Reconstruction Quality Evaluation
===========================================

Loads toy autoencoder checkpoints and measures basic reconstruction fidelity
on held-out validation clips.

Metrics (per clip, per model)
-----------------------------
  **Multi-resolution STFT reconstruction loss (MS-STFT)** — the exact
  training-time spectral objective, ``SAMELoss``'s magnitude component
  (spectral contrast + adaptive log-magnitude) summed over the 7 STFT
  resolutions (FFT 32 → 2048, 75 % overlap, Hann window).

  **Log Spectral Distance (LSD)** — measures the spectral distance between
  original and reconstructed audio in dB. Lower values indicate better
  spectral match. Computed using STFT with Hanning window, 2048 FFT size,
  512 hop length, on peak-normalized audio. Matches the implementation in
  lims_test_toy.py.

  **Decomposed LSD (sparse-audio safe)** — the plain mean LSD is dominated
  by silent frames of sparse references (drums, isolated events, speech
  pauses): reference frames at the ``+1e-12`` floor (≈ −120 dB) against a
  low-level recon noise floor contribute ~50–90 dB of error per frame,
  swamping the ~3 dB error of the actual content.  ``compute_lsd_decomposed``
  additionally reports ``lsd_active`` (mean over reference frames ≥
  −60 dBFS), ``lsd_silent``, ``lsd_median``, ``lsd_energy_weighted`` and
  ``sparsity`` (fraction of silent frames), plus per-domain aggregates
  (VCTK / FSD50K / FMA).  For sparse audio, ``lsd_active`` and
  ``lsd_energy_weighted`` are the meaningful fidelity numbers; MS-STFT
  remains the primary reconstruction objective metric.

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``toy_evaluation.data``):
validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.
All randomness is seeded so runs are reproducible.

Usage
-----
    # Evaluate 4 checkpoints
    python toy_evaluation/eval_reconstruction.py \
        --checkpoints \
            toy_training/checkpoints/transformer_best_full_best.pt \
            toy_training/checkpoints/transformer_norm_full_best.pt \
            toy_training/checkpoints/transformer_softMax_full_best.pt \
            toy_training/checkpoints/transformer_minimal_full_best.pt \
        --num_clips 100

    # Or evaluate 2 checkpoints (legacy mode)
    python toy_evaluation/eval_reconstruction.py \
        --ckpt_baseline toy_training/checkpoints/baseline.pt \
        --ckpt_phase toy_training/checkpoints/with_mixup.pt \
        --num_clips 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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

from toy_evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    stem_to_numpy,
)
from toy_evaluation.model_utils import detect_model_type, load_model
from toy_training.losses import SAMELoss


# =============================================================================
#  Metrics
# =============================================================================

def ms_stft_loss(
    orig: np.ndarray,
    recon: np.ndarray,
    device: torch.device,
    loss_fn: Optional[SAMELoss] = None,
) -> float:
    """Multi-resolution STFT reconstruction loss (SAME mag-loss).

    Wraps ``SAMELoss(use_phase=False, return_components=True)`` — the
    magnitude-only component = spectral contrast + adaptive log-magnitude,
    summed over the 7 resolutions the toy models were trained with.  Lower =
    closer spectral envelope to the original.
    """
    if loss_fn is None:
        loss_fn = SAMELoss()
    pred = torch.from_numpy(recon.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    target = torch.from_numpy(orig.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, mag_loss = loss_fn(pred, target, use_phase=False, return_components=True)
    return float(mag_loss.item())


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1] range."""
    peak = np.max(np.abs(audio.astype(np.float64)))
    if peak > 1e-12:
        return audio.astype(np.float64) / peak
    return audio.astype(np.float64)


def compute_lsd(
    audio_ref: np.ndarray,
    audio_est: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """Log-Spectral Distance (dB).  Both inputs peak-normalised.
    
    This matches the implementation in lims_test_toy.py which is used
    throughout the analysis.
    
    Parameters
    ----------
    audio_ref : np.ndarray
        Reference audio waveform.
    audio_est : np.ndarray
        Estimated/reconstructed audio waveform.
    n_fft : int
        FFT window size (default: 2048).
    hop : int
        Hop length between STFT frames (default: 512).

    Returns
    -------
    float
        LSD in dB (mean of absolute log-spectral differences).
    """
    audio_ref = peak_normalize(audio_ref)
    audio_est = peak_normalize(audio_est)
    
    if audio_ref.ndim == 0:
        return 0.0

    window = np.hanning(n_fft)
    
    def _spec_db(sig):
        n_frames = (len(sig) - n_fft) // hop + 1
        n_bins = n_fft // 2 + 1
        spec = np.zeros((n_bins, n_frames))
        for t in range(n_frames):
            start = t * hop
            spec[:, t] = np.abs(np.fft.rfft(sig[start:start + n_fft] * window))
        return 10.0 * np.log10(spec + 1e-12)

    db_ref = _spec_db(audio_ref)
    db_est = _spec_db(audio_est)
    min_f = min(db_ref.shape[1], db_est.shape[1])
    return float(np.mean(np.abs(db_ref[:, :min_f] - db_est[:, :min_f])))


def compute_lsd_decomposed(
    audio_ref: np.ndarray,
    audio_est: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    silence_db: float = 60.0,
) -> Dict[str, float]:
    """LSD decomposed into active / silent reference frames (sparse-audio safe).

    The plain mean LSD is dominated by silent frames of sparse references
    (drum hits, isolated events, speech pauses): a reference frame at the
    ``+1e-12`` floor (≈ −120 dB) compared against a low-level reconstruction
    noise floor contributes ~50–90 dB of error per frame, swamping the
    ~3 dB error of the actual content.  This routine reports the standard
    ``lsd`` alongside:

    - ``lsd_active`` — mean |Δlog-spec| over frames whose *reference* peak
      spectral magnitude is ≥ −``silence_db`` dBFS (relative to the clip peak;
      both inputs are peak-normalised).  This is the meaningful fidelity
      number for sparse audio.
    - ``lsd_silent`` — mean over the complementary frames (diagnostic).
    - ``lsd_median`` — median of per-frame LSD (robust to a few bad frames).
    - ``lsd_energy_weighted`` — per-frame LSD weighted by reference frame
      energy (silence automatically down-weighted).
    - ``sparsity`` — fraction of silent reference frames (0 = dense,
      1 = fully silent).
    """
    audio_ref = peak_normalize(audio_ref)
    audio_est = peak_normalize(audio_est)

    if audio_ref.ndim == 0 or len(audio_ref) < n_fft:
        return {
            "lsd": 0.0,
            "lsd_active": float("nan"),
            "lsd_silent": float("nan"),
            "lsd_median": 0.0,
            "lsd_energy_weighted": 0.0,
            "sparsity": 0.0,
        }

    window = np.hanning(n_fft)

    def _spec_db(sig):
        n_frames = (len(sig) - n_fft) // hop + 1
        n_bins = n_fft // 2 + 1
        spec = np.zeros((n_bins, n_frames))
        for t in range(n_frames):
            start = t * hop
            spec[:, t] = np.abs(np.fft.rfft(sig[start:start + n_fft] * window))
        return 10.0 * np.log10(spec + 1e-12)

    db_ref = _spec_db(audio_ref)
    db_est = _spec_db(audio_est)
    min_f = min(db_ref.shape[1], db_est.shape[1])
    per_frame = np.abs(db_ref[:, :min_f] - db_est[:, :min_f])  # (n_bins, n_frames)
    per_frame_lsd = per_frame.mean(axis=0)                      # (n_frames,)

    # Reference frame loudness: peak spectral magnitude (dBFS, ref is peak-norm'ed).
    ref_lin = 10.0 ** (db_ref[:, :min_f] / 10.0)
    frame_peak_db = 10.0 * np.log10(ref_lin.max(axis=0) + 1e-12)
    silent = frame_peak_db < -silence_db

    lsd = float(per_frame_lsd.mean())
    lsd_active = float(per_frame_lsd[~silent].mean()) if (~silent).any() else float("nan")
    lsd_silent = float(per_frame_lsd[silent].mean()) if silent.any() else float("nan")
    lsd_median = float(np.median(per_frame_lsd))
    ref_frame_energy = ref_lin.mean(axis=0)
    w = ref_frame_energy / (ref_frame_energy.sum() + 1e-12)
    lsd_energy_weighted = float(np.sum(per_frame_lsd * w))

    return {
        "lsd": lsd,
        "lsd_active": lsd_active,
        "lsd_silent": lsd_silent,
        "lsd_median": lsd_median,
        "lsd_energy_weighted": lsd_energy_weighted,
        "sparsity": float(silent.mean()),
    }


# =============================================================================
#  Single-checkpoint core
# =============================================================================

def evaluate_reconstruction_checkpoint(
    ckpt_path: str,
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_clips: int = 100,
    seed: int = 42,
    model_type: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    clip_duration: float = CLIP_DURATION,
    clip_indices: Optional[List[int]] = None,
    save_wav_dir: Optional[Path] = None,
    include_lsd: bool = True,
) -> Dict:
    """Reconstruction fidelity for ONE checkpoint on validation stems.

    Parameters
    ----------
    stems : list[(pt_path, stem_name, track_id)] — validation stems.
    clip_indices : optional pre-selected stem indices — pass the SAME index
        list when comparing multiple checkpoints.
    include_lsd : bool
        If True, compute Log Spectral Distance in addition to MS-STFT.

    Returns dict with mean/std MS-STFT reconstruction loss, LSD, and per-clip values.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_samples = int(clip_duration * sample_rate)

    if model_type is None:
        model_type = detect_model_type(ckpt_path, device)
        print(f"  [Auto-detect] Model type: {model_type}")
    model = load_model(ckpt_path, device, model_type)
    print(f"  Model: {Path(ckpt_path).stem}  ({model_type})")

    n_available = len(stems)
    n_clips = min(num_clips, n_available)
    if clip_indices is None:
        rng = random.Random(seed)
        clip_indices = rng.sample(range(n_available), n_clips)

    # Reuse one SAMELoss instance across all clips (windows are pre-created).
    loss_fn = SAMELoss()
    msstft_losses: List[float] = []
    lsd_losses: List[float] = []
    lsd_active_losses: List[float] = []
    lsd_silent_losses: List[float] = []
    lsd_median_losses: List[float] = []
    lsd_energy_weighted_losses: List[float] = []
    sparsities: List[float] = []
    domains: List[str] = []
    per_clip: List[Dict] = []

    for idx in tqdm(clip_indices, desc="Clips", unit="clip"):
        path, stem, track = stems[idx]
        try:
            audio = stem_to_numpy(path, clip_samples, centre=True)
        except Exception as exc:
            print(f"  ⚠ Skip {path}: {exc}")
            continue

        audio_t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon_t = model(audio_t)
        recon = recon_t.squeeze().cpu().numpy()
        if len(recon) > len(audio):
            recon = recon[: len(audio)]
        elif len(recon) < len(audio):
            recon = np.pad(recon, (0, len(audio) - len(recon)))

        msstft_losses.append(ms_stft_loss(audio, recon, device, loss_fn=loss_fn))

        if include_lsd:
            lsd_d = compute_lsd_decomposed(audio, recon)
            lsd_losses.append(lsd_d["lsd"])
            lsd_active_losses.append(lsd_d["lsd_active"])
            lsd_silent_losses.append(lsd_d["lsd_silent"])
            lsd_median_losses.append(lsd_d["lsd_median"])
            lsd_energy_weighted_losses.append(lsd_d["lsd_energy_weighted"])
            sparsities.append(lsd_d["sparsity"])
            domains.append(track.split("::")[0])
            per_clip.append(
                {
                    "stem": stem,
                    "track": track,
                    "domain": track.split("::")[0],
                    "msstft": msstft_losses[-1],
                    "lsd": lsd_d["lsd"],
                    "lsd_active": lsd_d["lsd_active"],
                    "lsd_silent": lsd_d["lsd_silent"],
                    "sparsity": lsd_d["sparsity"],
                }
            )

        if save_wav_dir is not None:
            clip_dir = save_wav_dir / f"{track}_{stem}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            def _save(name: str, arr: np.ndarray) -> None:
                arr_int16 = np.clip(arr, -1.0, 1.0) * 32767.0
                wavfile.write(str(clip_dir / name), sample_rate, arr_int16.astype(np.int16))

            _save("original.wav", audio)
            _save("recon.wav", recon)

    result = {
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "num_clips_requested": len(clip_indices),
        "num_clips_valid": len(msstft_losses),
        "mean_msstft": float(np.mean(msstft_losses)) if msstft_losses else float("nan"),
        "std_msstft": float(np.std(msstft_losses)) if msstft_losses else float("nan"),
    }

    if include_lsd and lsd_losses:
        result["mean_lsd"] = float(np.mean(lsd_losses))
        result["std_lsd"] = float(np.std(lsd_losses))
        result["median_lsd"] = float(np.median(lsd_losses))
        result["mean_lsd_active"] = float(np.nanmean(lsd_active_losses))
        result["std_lsd_active"] = float(np.nanstd(lsd_active_losses))
        result["mean_lsd_silent"] = float(np.nanmean(lsd_silent_losses))
        result["mean_lsd_median"] = float(np.mean(lsd_median_losses))
        result["mean_lsd_energy_weighted"] = float(np.mean(lsd_energy_weighted_losses))
        result["mean_sparsity"] = float(np.mean(sparsities))
        # Per-domain aggregates (sparse FSD50K vs dense VCTK/FMA).
        by_domain: Dict[str, Dict[str, float]] = {}
        for dom in sorted(set(domains)):
            idxs = [i for i, d in enumerate(domains) if d == dom]
            by_domain[dom] = {
                "n": len(idxs),
                "mean_lsd": float(np.mean([lsd_losses[i] for i in idxs])),
                "mean_lsd_active": float(np.nanmean([lsd_active_losses[i] for i in idxs])),
                "mean_sparsity": float(np.mean([sparsities[i] for i in idxs])),
            }
        result["by_domain"] = by_domain
        result["per_clip"] = per_clip

    return result


# =============================================================================
#  Multi-checkpoint evaluation
# =============================================================================

def evaluate_multiple_checkpoints(
    ckpt_paths: List[str],
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_clips: int = 100,
    seed: int = 42,
    sample_rate: int = MODEL_SAMPLE_RATE,
    clip_duration: float = CLIP_DURATION,
    save_wav_dir: Optional[Path] = None,
    include_lsd: bool = True,
) -> Dict:
    """Evaluate reconstruction quality for multiple checkpoints.

    Parameters
    ----------
    ckpt_paths : list of str
        Paths to checkpoint files to evaluate.
    stems : list[(pt_path, stem_name, track_id)]
        Validation stems.
    num_clips : int
        Number of clips to evaluate per checkpoint.
    seed : int
        Random seed for reproducibility.
    sample_rate : int
        Sample rate in Hz.
    clip_duration : float
        Duration of each clip in seconds.
    save_wav_dir : Path or None
        Directory to save reconstructed audio (optional).
    include_lsd : bool
        If True, compute LSD in addition to MS-STFT.

    Returns
    -------
    dict
        Dictionary with results for all checkpoints.
    """
    # Generate common clip indices for all checkpoints
    n_available = len(stems)
    n_clips = min(num_clips, n_available)
    rng = random.Random(seed)
    clip_indices = rng.sample(range(n_available), n_clips)

    results = {}
    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).stem
        print(f"\nEvaluating {ckpt_name}...")
        results[ckpt_name] = evaluate_reconstruction_checkpoint(
            ckpt_path,
            stems,
            device,
            num_clips=n_clips,
            seed=seed,
            sample_rate=sample_rate,
            clip_duration=clip_duration,
            clip_indices=clip_indices,
            save_wav_dir=save_wav_dir,
            include_lsd=include_lsd,
        )

    return results


# =============================================================================
#  CLI
# =============================================================================

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sample_rate = args.sample_rate
    print(f"[Device] {device}")

    print("[1/3] Collecting validation stems (multi-dataset training split) ...")
    stems = collect_validation_stems(args.data_dirs, args.cache_dir, sample_rate=sample_rate)
    print(f"  Found {len(stems)} validation stems")

    # Determine output directory
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Check if we're using the new multi-checkpoint mode or legacy mode
    if args.checkpoints:
        # Multi-checkpoint mode
        ckpt_paths = args.checkpoints
        print(f"[2/3] Evaluating {len(ckpt_paths)} checkpoints ...")

        results = evaluate_multiple_checkpoints(
            ckpt_paths,
            stems,
            device,
            num_clips=args.num_clips,
            seed=args.seed,
            sample_rate=sample_rate,
            clip_duration=CLIP_DURATION,
            save_wav_dir=out_dir,
            include_lsd=True,  # Always compute LSD in multi-checkpoint mode
        )

        # Print results table
        print("\n[3/3] Results")
        print("=" * 80)

        # Get all checkpoint names
        ckpt_names = list(results.keys())

        # Print header
        header = f"{'Metric':<25}"
        for name in ckpt_names:
            header += f"{name[:15]:>15}"
        print(header)
        print("-" * 80)

        # Print MS-STFT row
        row = f"{'MS-STFT loss':<25}"
        for name in ckpt_names:
            mean_msstft = results[name]["mean_msstft"]
            row += f"{mean_msstft:>15.6f}"
        print(row)

        # Print LSD rows (if available)
        if "mean_lsd" in results[ckpt_names[0]]:
            rows = [
                ("LSD (dB)", "mean_lsd", "{:>15.4f}"),
                ("LSD active (dB)", "mean_lsd_active", "{:>15.4f}"),
                ("LSD silent (dB)", "mean_lsd_silent", "{:>15.4f}"),
                ("LSD median (dB)", "mean_lsd_median", "{:>15.4f}"),
                ("LSD energy-wtd (dB)", "mean_lsd_energy_weighted", "{:>15.4f}"),
                ("sparsity (frac silent)", "mean_sparsity", "{:>15.4f}"),
            ]
            for label, key, fmt in rows:
                row = f"{label:<25}"
                for name in ckpt_names:
                    row += fmt.format(results[name].get(key, float("nan")))
                print(row)

            # Per-domain LSD table (sparse vs dense datasets).
            print("-" * 80)
            print("LSD by domain (full | active | sparsity):")
            header = f"{'domain':<24}"
            for name in ckpt_names:
                header += f"{name[:15]:>22}"
            print(header)
            all_domains = sorted({d for n in ckpt_names for d in results[n].get("by_domain", {})})
            for dom in all_domains:
                row = f"{dom[:24]:<24}"
                for name in ckpt_names:
                    bd = results[name].get("by_domain", {}).get(dom, {})
                    row += f"{bd.get('mean_lsd', float('nan')):>8.2f}/{bd.get('mean_lsd_active', float('nan')):>6.2f}/{bd.get('mean_sparsity', float('nan')):>5.2f}"
                print(row)

        print("=" * 80)

    else:
        # Legacy two-checkpoint mode
        clip_samples = int(CLIP_DURATION * sample_rate)
        rng = random.Random(args.seed)
        n_clips = min(args.num_clips, len(stems))
        clip_indices = rng.sample(range(len(stems)), n_clips)

        print("[2/3] Evaluating baseline ...")
        res_base = evaluate_reconstruction_checkpoint(
            args.ckpt_baseline,
            stems,
            device,
            num_clips=n_clips,
            seed=args.seed,
            model_type=args.model_type,
            sample_rate=sample_rate,
            clip_duration=CLIP_DURATION,
            clip_indices=clip_indices,
            save_wav_dir=out_dir,
            include_lsd=True,
        )
        print("[3/3] Evaluating phase/compare ...")
        res_phase = evaluate_reconstruction_checkpoint(
            args.ckpt_phase,
            stems,
            device,
            num_clips=n_clips,
            seed=args.seed,
            model_type=args.model_type,
            sample_rate=sample_rate,
            clip_duration=CLIP_DURATION,
            clip_indices=clip_indices,
            save_wav_dir=out_dir,
            include_lsd=True,
        )

        print("\n[4/4] Results")
        print("=" * 72)
        print(f"{'Metric':<30} {'Baseline':>16} {'Phase':>16}")
        print("-" * 72)
        for label, key in [("MS-STFT loss", "msstft"), ("LSD (dB)", "lsd")]:
            base_val = res_base.get(f"mean_{key}", float("nan"))
            phase_val = res_phase.get(f"mean_{key}", float("nan"))
            print(f"{label:<30} {base_val:>16.8f} {phase_val:>16.8f}")
        print("=" * 72)

        results = {
            "baseline": res_base,
            "phase": res_phase,
        }

    # Save JSON summary
    summary = {
        "description": "Toy AE Reconstruction Evaluation",
        "timestamp": datetime.now().isoformat(),
        "clip_duration_s": CLIP_DURATION,
        "num_clips": args.num_clips if args.checkpoints else n_clips,
        "results": results,
    }

    json_path = _EVAL_DIR / "reconstruction_summary.json"

    def _sanitize(obj):
        """Recursively convert non-finite floats (NaN/Inf) to None for strict JSON."""
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    with open(json_path, "w") as fp:
        json.dump(_sanitize(summary), fp, indent=2, default=str, allow_nan=False)
    print(f"  📄 JSON → {json_path}")
    print("✓ Evaluation complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy AE Reconstruction Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Multi-checkpoint mode (new)
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Paths to checkpoint files to evaluate (supports multiple). "
        "If provided, evaluates all checkpoints. Otherwise, uses --ckpt_baseline and --ckpt_phase.",
    )

    # Legacy two-checkpoint mode
    p.add_argument("--ckpt_baseline", type=str, default=None,
                   help="Path to first checkpoint (legacy mode).")
    p.add_argument("--ckpt_phase", type=str, default=None,
                   help="Path to second checkpoint (legacy mode).")

    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                   nargs="+", default=None,
                   help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                        "training set.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Preprocessed cache base dir.")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["conv", "transformer"],
                   help="Model architecture. Auto-detected from checkpoint if omitted.")
    p.add_argument("--num_clips", type=int, default=100,
                   help="Number of random test clips (default: 100).")
    p.add_argument("--output_dir", default=None,
                   help="Directory to save reconstructed audio (optional).")
    p.add_argument("--device", default="cuda",
                   help="Device (default: cuda).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=MODEL_SAMPLE_RATE,
                   help=f"Target sample rate in Hz (default: {MODEL_SAMPLE_RATE}).")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
