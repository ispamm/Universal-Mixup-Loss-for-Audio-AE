#!/usr/bin/env python3
"""
LSD / sparsity decomposition analysis
======================================

Verifies the hypothesis that the reported reconstruction LSD is inflated by
silent frames in sparse audio (drums, isolated events, speech pauses).

For every saved (original.wav, recon.wav) pair under
``toy_evaluation/reconstructions/`` it computes:

  * the standard LSD (exactly as ``eval_reconstruction.compute_lsd``),
  * a per-frame decomposition of the absolute log-spectral difference into
    ACTIVE and SILENT frames of the REFERENCE (original) signal,
  * the clip sparsity (fraction of silent frames), and
  * domain-specific aggregates (VCTK speech / FSD50K sound events / FMA music).

Thresholds: a reference frame is "silent" when its peak spectral magnitude is
below ``--silence_db`` dBFS (relative to the clip peak, since both signals are
peak-normalised by the LSD implementation).

Usage:
    python toy_evaluation/analyze_lsd_sparsity.py [--recon-dir ...] \
        [--silence-db 60] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

_N_FFT = 2048
_HOP = 512


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio.astype(np.float64)))
    if peak > 1e-12:
        return audio.astype(np.float64) / peak
    return audio.astype(np.float64)


def frame_spec_db(sig: np.ndarray) -> np.ndarray:
    """(n_bins, n_frames) log-magnitude STFT, identical to eval_reconstruction."""
    window = np.hanning(_N_FFT)
    n_frames = (len(sig) - _N_FFT) // _HOP + 1
    spec = np.zeros((_N_FFT // 2 + 1, n_frames))
    for t in range(n_frames):
        start = t * _HOP
        spec[:, t] = np.abs(np.fft.rfft(sig[start : start + _N_FFT] * window))
    return 10.0 * np.log10(spec + 1e-12)


def analyze_pair(ref_wav: Path, est_wav: Path, silence_db: float) -> dict:
    sr_r, ref = wavfile.read(str(ref_wav))
    sr_e, est = wavfile.read(str(est_wav))
    assert sr_r == sr_e, (sr_r, sr_e)
    ref = peak_normalize(ref.astype(np.float64))
    est = peak_normalize(est.astype(np.float64))
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]

    db_ref = frame_spec_db(ref)
    db_est = frame_spec_db(est)
    per_frame = np.abs(db_ref - db_est)          # (n_bins, n_frames)
    lsd_full = float(per_frame.mean())

    # Reference frame "loudness": peak spectral magnitude in dB (ref is peak-norm'ed).
    ref_lin = 10.0 ** (db_ref / 10.0)
    frame_peak_db = 10.0 * np.log10(ref_lin.max(axis=0) + 1e-12)  # (n_frames,)
    silent = frame_peak_db < -silence_db
    n_frames = len(silent)
    sparsity = float(silent.mean())

    lsd_active = float(per_frame[:, ~silent].mean()) if (~silent).any() else float("nan")
    lsd_silent = float(per_frame[:, silent].mean()) if silent.any() else float("nan")

    # Time-weighted version: fraction of total |Δ| mass attributable to silent frames.
    silent_mass_frac = float(
        per_frame[:, silent].sum() / per_frame.sum()
    ) if silent.any() else 0.0

    # Energy-weighted LSD (weight frames by reference frame energy).
    ref_frame_energy = (10.0 ** (db_ref / 10.0)).mean(axis=0)
    w = ref_frame_energy / (ref_frame_energy.sum() + 1e-12)
    lsd_energy_weighted = float(np.sum(per_frame.mean(axis=0) * w))

    # Median (robust) LSD over frames.
    lsd_median = float(np.median(per_frame.mean(axis=0)))

    return {
        "lsd_full": lsd_full,
        "lsd_median": lsd_median,
        "lsd_energy_weighted": lsd_energy_weighted,
        "lsd_active": lsd_active,
        "lsd_silent": lsd_silent,
        "sparsity": sparsity,
        "silent_mass_frac": silent_mass_frac,
        "n_frames": n_frames,
    }


def domain_of(dirname: str) -> str:
    if "vctk" in dirname or "p3" in dirname or dirname.startswith("p"):
        return "vctk"
    if "fsd50k" in dirname.lower():
        return "fsd50k"
    if "fma" in dirname.lower():
        return "fma"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", type=Path,
                    default=Path(__file__).resolve().parent / "reconstructions")
    ap.add_argument("--silence-db", type=float, default=60.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    pairs = sorted(args.recon_dir.glob("*/original.wav"))
    rows = []
    for orig in pairs:
        est = orig.with_name("recon.wav")
        if not est.exists():
            continue
        clip_dir = orig.parent.name
        r = analyze_pair(orig, est, args.silence_db)
        r["clip"] = clip_dir
        r["domain"] = domain_of(clip_dir)
        rows.append(r)

    lsd_full = np.array([r["lsd_full"] for r in rows])
    lsd_active = np.array([r["lsd_active"] for r in rows])
    lsd_silent = np.array([r["lsd_silent"] for r in rows])
    lsd_median = np.array([r["lsd_median"] for r in rows])
    lsd_ew = np.array([r["lsd_energy_weighted"] for r in rows])
    sparsity = np.array([r["sparsity"] for r in rows])
    silent_mass = np.array([r["silent_mass_frac"] for r in rows])

    print(f"pairs: {len(rows)}   silence threshold: -{args.silence_db:.0f} dBFS")
    print("=" * 78)
    print(f"{'metric':<24}{'mean':>10}{'median':>10}{'std':>10}")
    for name, arr in [
        ("LSD full (dB)", lsd_full),
        ("LSD active (dB)", lsd_active),
        ("LSD silent (dB)", lsd_silent),
        ("LSD median-frame (dB)", lsd_median),
        ("LSD energy-wtd (dB)", lsd_ew),
        ("sparsity (frac silent)", sparsity),
        ("silent mass frac", silent_mass),
    ]:
        print(f"{name:<24}{np.nanmean(arr):>10.3f}{np.nanmedian(arr):>10.3f}{np.nanstd(arr):>10.3f}")

    # Correlation between sparsity and LSD.
    valid = ~(np.isnan(lsd_full) | np.isnan(sparsity))
    if valid.sum() > 2:
        rho = np.corrcoef(lsd_full[valid], sparsity[valid])[0, 1]
        print(f"\ncorr(LSD_full, sparsity) = {rho:+.3f}  (n={valid.sum()})")
    valid_a = ~(np.isnan(lsd_active) | np.isnan(sparsity))
    if valid_a.sum() > 2:
        rho_a = np.corrcoef(lsd_active[valid_a], sparsity[valid_a])[0, 1]
        print(f"corr(LSD_active, sparsity) = {rho_a:+.3f}  (n={valid_a.sum()})")

    print("\nby domain:")
    print(f"{'domain':<10}{'n':>5}{'LSD full':>10}{'LSD act':>10}{'LSD sil':>10}{'sparsity':>10}{'sil mass':>10}")
    for dom in ["vctk", "fsd50k", "fma", "other"]:
        sub = [r for r in rows if r["domain"] == dom]
        if not sub:
            continue
        print(
            f"{dom:<10}{len(sub):>5}"
            f"{np.mean([r['lsd_full'] for r in sub]):>10.3f}"
            f"{np.nanmean([r['lsd_active'] for r in sub]):>10.3f}"
            f"{np.nanmean([r['lsd_silent'] for r in sub]):>10.3f}"
            f"{np.mean([r['sparsity'] for r in sub]):>10.3f}"
            f"{np.mean([r['silent_mass_frac'] for r in sub]):>10.3f}"
        )

    # Top-10 sparsest clips, and their full vs active LSD.
    print("\nsparsest clips (full vs active LSD):")
    for r in sorted(rows, key=lambda r: -r["sparsity"])[:10]:
        print(
            f"  {r['clip'][:46]:<48} sp={r['sparsity']:.2f} "
            f"LSD={r['lsd_full']:6.2f} act={r['lsd_active']:6.2f} "
            f"sil={r['lsd_silent']:6.2f}"
        )

    if args.json:
        summary = {
            "silence_db_threshold": args.silence_db,
            "num_pairs": len(rows),
            "aggregates": {
                "lsd_full": {"mean": float(np.nanmean(lsd_full)), "median": float(np.nanmedian(lsd_full)), "std": float(np.nanstd(lsd_full))},
                "lsd_active": {"mean": float(np.nanmean(lsd_active)), "median": float(np.nanmedian(lsd_active)), "std": float(np.nanstd(lsd_active))},
                "lsd_silent": {"mean": float(np.nanmean(lsd_silent)), "median": float(np.nanmedian(lsd_silent)), "std": float(np.nanstd(lsd_silent))},
                "lsd_median_frame": {"mean": float(np.nanmean(lsd_median)), "median": float(np.nanmedian(lsd_median))},
                "lsd_energy_weighted": {"mean": float(np.nanmean(lsd_ew)), "median": float(np.nanmedian(lsd_ew))},
                "sparsity": {"mean": float(np.nanmean(sparsity)), "median": float(np.nanmedian(sparsity))},
                "silent_mass_frac": {"mean": float(np.nanmean(silent_mass))},
                "corr_lsd_sparsity": float(rho) if valid.sum() > 2 else float("nan"),
            },
            "per_clip": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\n📄 JSON -> {args.json}")


if __name__ == "__main__":
    main()
