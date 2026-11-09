#!/usr/bin/env python3
"""
AE — LIMS (Latent Interpolation & Manifold Smoothness) Test
================================================================

Evaluates the **decoder's topological additivity** for autoencoder
checkpoints (Transformer or CNN).

Core idea
---------
For a pair of audio clips (A, B):
  1. Encode  →  Z_A, Z_B
  2. For each α ∈ [0, 1]:
     a. Latent interpolation:  Z_interp = interp(Z_A, Z_B, α)
        [Lerp / Slerp / Trig (Constant-Power)]
     b. Decode:  audio_dec = decode(Z_interp)
     c. Ideal audio mix:  audio_mix_lerp = (1-α)·A + α·B  (linear)
        or  audio_mix_trig = cos(απ/2)·A + sin(απ/2)·B  (constant-power)
     d. Compare audio_dec vs audio_mix  (Log-Spectral Distance)
     e. Measure "Energy Drop": normalised RMS(reconstructed) / mean_RMS(A, B)

If the decoder is topologically additive, latent-space interpolation should
produce audio that closely matches the ideal mix of the originals.

In addition to the per-α LSD / energy curves, three trigMix-specific
curve-level metrics are reported:

  - ``lsd_curve_smoothness_trig``   — variance of the mean trig-LSD curve
    across α (lower = interpolation quality is consistent along the whole
    path, not just at the endpoints).
  - ``energy_curve_flatness_trig``  — std of the mean trig-energy curve
    across α (lower = constant-power mixing is preserved at every α).
  - ``energy_curve_gap_trig``       — mean |latent-energy − ideal-trig-energy|
    across α (closeness of the decoded trigMix energy to the ideal
    constant-power curve).

To remove the old scheme/target ambiguity, **every interpolation scheme is now
scored against BOTH ideal mixes** (Lerp target and Trig target) and both are
stored; the scheme's *primary* target is the lerp mix for Lerp and Slerp
(Slerp interpolates magnitudes linearly) and the trig mix for Trig.

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``evaluation.data``):
the validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.

Outputs
-------
  evaluation/lims_summary.json   — aggregated per-alpha metrics (LSD,
                                       Energy curves + ground-truth + curve
                                       smoothness/flatness for trigMix)
  evaluation/lims_plot.png       — LSD U-curve comparison
  evaluation/lims_energy_plot.png— Energy Drop comparison
  evaluation/lims_outputs/       — per-pair per-model interpolated audio

Usage
-----
    python evaluation/lims_test.py \\
        --ckpt_baseline training/checkpoints/baseline.pt \\
        --ckpt_phase training/checkpoints/with_mixup.pt \\
        --num_pairs 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent            # evaluation/
_REPO_ROOT = _EVAL_DIR.parents[0]        # repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import (
    CLIP_DURATION,
    DEFAULT_DATA_DIRS,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from evaluation.model_utils import AutoencoderWrapper

# =============================================================================
#  Constants
# =============================================================================

INTERP_STEPS = 11  # α ∈ {0.0, 0.1, …, 1.0}


# =============================================================================
#  Interpolation
# =============================================================================

def lerp(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear interpolation."""
    return (1.0 - alpha) * z_a + alpha * z_b


def slerp(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation with (linearly interpolated) magnitude.

    Note: magnitudes are lerped by design — this avoids collapsing to the OOD
    origin and matches how the real-codec LIMS scripts define slerp.  It is
    NOT a constant-norm slerp.
    """
    a_flat = z_a.reshape(-1)
    b_flat = z_b.reshape(-1)
    mag_a = torch.norm(a_flat)
    mag_b = torch.norm(b_flat)

    eps = 1e-12
    u_a = a_flat / (mag_a + eps)
    u_b = b_flat / (mag_b + eps)

    cos_theta = torch.dot(u_a, u_b).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
    theta = torch.acos(cos_theta)

    if theta < 1e-8:
        return lerp(z_a, z_b, alpha)

    sin_theta = torch.sin(theta)
    w_a = torch.sin((1.0 - alpha) * theta) / sin_theta
    w_b = torch.sin(alpha * theta) / sin_theta
    unit_interp = w_a * u_a + w_b * u_b
    interp_mag = (1.0 - alpha) * mag_a + alpha * mag_b

    return (unit_interp * interp_mag).reshape(z_a.shape)


def trig_mix(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Trigonometric (constant-power) interpolation.

    ``cos(α·π/2)·A + sin(α·π/2)·B`` — preserves RMS energy when A ⟂ B
    and have equal power.
    """
    theta = alpha * (np.pi / 2.0)
    return np.cos(theta) * z_a + np.sin(theta) * z_b


# =============================================================================
#  Metrics
# =============================================================================

def rms_energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio.astype(np.float64)))
    if peak > 1e-12:
        return audio.astype(np.float64) / peak
    return audio.astype(np.float64)


def compute_lsd(audio_ref: np.ndarray, audio_est: np.ndarray,
                n_fft: int = 2048, hop: int = 512) -> float:
    """Log-Spectral Distance (dB).  Both inputs peak-normalised."""
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


def compute_lsd_decomposed(audio_ref: np.ndarray, audio_est: np.ndarray,
                           n_fft: int = 2048, hop: int = 512,
                           silence_db: float = 60.0) -> Dict[str, float]:
    """LSD decomposed into active / silent reference frames (sparse-audio safe).

    The plain mean LSD is dominated by silent frames of sparse references
    (drum hits, isolated events, speech pauses): reference frames at the
    ``+1e-12`` floor (≈ −120 dB) against a low-level noise floor contribute
    ~50–90 dB of error per frame.  Returns ``lsd`` (standard), ``lsd_active``
    (mean over reference frames ≥ −``silence_db`` dBFS), ``lsd_silent``,
    ``lsd_median``, ``lsd_energy_weighted`` and ``sparsity`` (fraction of
    silent reference frames).  Identical convention to
    ``eval_reconstruction.compute_lsd_decomposed``.
    """
    audio_ref = peak_normalize(audio_ref)
    audio_est = peak_normalize(audio_est)
    if audio_ref.ndim == 0 or len(audio_ref) < n_fft:
        return {"lsd": 0.0, "lsd_active": float("nan"), "lsd_silent": float("nan"),
                "lsd_median": 0.0, "lsd_energy_weighted": 0.0, "sparsity": 0.0}

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
    per_frame = np.abs(db_ref[:, :min_f] - db_est[:, :min_f])
    per_frame_lsd = per_frame.mean(axis=0)

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
#  Single-checkpoint core (used by evaluate_all.py and the CLI)
# =============================================================================

def evaluate_lims_checkpoint(
    ckpt_path: str,
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_pairs: int = 100,
    seed: int = 42,
    pair_mode: str = "random",
    model_type: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    clip_duration: float = CLIP_DURATION,
    interp_steps: int = INTERP_STEPS,
    pairs: Optional[List[Tuple[int, int]]] = None,
    save_wav_dir: Optional[Path] = None,
    msstft_loss_fn: Optional[object] = None,
) -> Dict:
    """Run the decoder-additivity LIMS test for ONE checkpoint.

    Parameters
    ----------
    stems : list[(pt_path, stem_name, track_id)] — validation stems
        (see ``evaluation.data.collect_validation_stems``).
    pairs : optional pre-generated pair list — pass the SAME pair list when
        comparing multiple checkpoints so every model sees identical pairs.
    msstft_loss_fn : optional SAMELoss instance — when given, also accumulates
        the training-matched MS-STFT (magnitude-only) between the decoded
        trigMix and the ideal mix at every α (``mean_trig_msstft``), as a
        spectral-metric cross-check on the LSD numbers.

    Returns the aggregated summary dict (per-α mean/std of LSD and energy
    curves for Lerp / Slerp / Trig, each scored against both ideal mixes
    where applicable, plus trigMix curve-level smoothness / flatness).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_samples = int(clip_duration * sample_rate)
    alphas = np.linspace(0.0, 1.0, interp_steps).tolist()

    if model_type is None:
        from evaluation.model_utils import detect_model_type
        model_type = detect_model_type(ckpt_path, device)
        print(f"  [Auto-detect] Model type: {model_type}")

    model = AutoencoderWrapper(ckpt_path, device, model_type)

    if pairs is None:
        pairs = generate_pairs(stems, num_pairs, seed=seed, mode=pair_mode)
    print(f"  {model.name} ({model_type}): {len(pairs)} pairs")

    # ---- Accumulators ----
    def _make_accum():
        acc = {
            "lerp_lsd_lerp":   np.full((len(pairs), interp_steps), np.nan),
            "lerp_lsd_trig":   np.full((len(pairs), interp_steps), np.nan),
            "slerp_lsd_lerp":  np.full((len(pairs), interp_steps), np.nan),
            "slerp_lsd_trig":  np.full((len(pairs), interp_steps), np.nan),
            "trig_lsd_lerp":   np.full((len(pairs), interp_steps), np.nan),
            "trig_lsd_trig":   np.full((len(pairs), interp_steps), np.nan),
            # Sparse-audio-robust variants for the primary trig comparison:
            # active-region LSD and reference-mix sparsity per (pair, α).
            "trig_lsd_active_trig": np.full((len(pairs), interp_steps), np.nan),
            "trig_lsd_silent_trig": np.full((len(pairs), interp_steps), np.nan),
            "trig_lsd_energy_trig": np.full((len(pairs), interp_steps), np.nan),
            "trig_ref_sparsity":    np.full((len(pairs), interp_steps), np.nan),
            "lerp_energy":     np.full((len(pairs), interp_steps), np.nan),
            "slerp_energy":    np.full((len(pairs), interp_steps), np.nan),
            "trig_energy":     np.full((len(pairs), interp_steps), np.nan),
        }
        if msstft_loss_fn is not None:  # training-matched MS-STFT on the trig mix
            acc["trig_msstft_trig"] = np.full((len(pairs), interp_steps), np.nan)
        return acc

    acc = _make_accum()
    gt_lerp_energy = np.full((len(pairs), interp_steps), np.nan)
    gt_trig_energy = np.full((len(pairs), interp_steps), np.nan)

    def _save_wav(subdir: Path, name: str, arr: np.ndarray) -> None:
        subdir.mkdir(parents=True, exist_ok=True)
        arr_int16 = np.clip(arr, -1.0, 1.0) * 32767.0
        wavfile.write(str(subdir / name), sample_rate, arr_int16.astype(np.int16))

    for pair_idx, (i, j) in enumerate(tqdm(pairs, desc="Pairs", unit="pair")):
        path_a, stem_a, track_a = stems[i]
        path_b, stem_b, track_b = stems[j]

        try:
            audio_a = stem_to_numpy(path_a, clip_samples, centre=True)
            audio_b = stem_to_numpy(path_b, clip_samples, centre=True)
        except Exception as exc:
            print(f"  ⚠ Skip pair {pair_idx}: load error – {exc}")
            continue

        orig_len = len(audio_a)
        rms_a = rms_energy(audio_a)
        rms_b = rms_energy(audio_b)
        rms_mean = (rms_a + rms_b) / 2.0

        try:
            z_a = model.encode(audio_a)
            z_b = model.encode(audio_b)
        except Exception as exc:
            print(f"  ⚠ Skip pair {pair_idx} [{model.name}]: encode – {exc}")
            continue

        if z_a.shape[-1] != z_b.shape[-1]:
            mlen = min(z_a.shape[-1], z_b.shape[-1])
            z_a = z_a[..., :mlen]
            z_b = z_b[..., :mlen]

        model_pair_dir = None
        if save_wav_dir is not None:
            model_pair_dir = save_wav_dir / f"pair_{pair_idx:04d}" / model.name

        for a_idx, alpha in enumerate(alphas):
            # ---- Ideal audio mixes (ground-truth) ----
            audio_mix_lerp = (1.0 - alpha) * audio_a + alpha * audio_b
            audio_mix_trig = trig_mix(
                torch.from_numpy(audio_a), torch.from_numpy(audio_b), alpha
            ).numpy()

            gt_lerp_energy[pair_idx, a_idx] = rms_energy(audio_mix_lerp) / (rms_mean + 1e-12)
            gt_trig_energy[pair_idx, a_idx] = rms_energy(audio_mix_trig) / (rms_mean + 1e-12)

            # ---- Lerp ----
            z_lerp = lerp(z_a, z_b, alpha)
            audio_lerp = model.decode(z_lerp, orig_len)
            acc["lerp_lsd_lerp"][pair_idx, a_idx] = compute_lsd(audio_mix_lerp, audio_lerp)
            acc["lerp_lsd_trig"][pair_idx, a_idx] = compute_lsd(audio_mix_trig, audio_lerp)
            acc["lerp_energy"][pair_idx, a_idx] = rms_energy(audio_lerp) / (rms_mean + 1e-12)
            if model_pair_dir is not None:
                _save_wav(model_pair_dir / "lerp", f"alpha_{alpha:.2f}.wav", audio_lerp)

            # ---- Slerp ----
            z_slerp = slerp(z_a, z_b, alpha)
            audio_slerp = model.decode(z_slerp, orig_len)
            acc["slerp_lsd_lerp"][pair_idx, a_idx] = compute_lsd(audio_mix_lerp, audio_slerp)
            acc["slerp_lsd_trig"][pair_idx, a_idx] = compute_lsd(audio_mix_trig, audio_slerp)
            acc["slerp_energy"][pair_idx, a_idx] = rms_energy(audio_slerp) / (rms_mean + 1e-12)
            if model_pair_dir is not None:
                _save_wav(model_pair_dir / "slerp", f"alpha_{alpha:.2f}.wav", audio_slerp)

            # ---- Trig (Constant-Power) ----
            z_trig = trig_mix(z_a, z_b, alpha)
            audio_trig = model.decode(z_trig, orig_len)
            acc["trig_lsd_lerp"][pair_idx, a_idx] = compute_lsd(audio_mix_lerp, audio_trig)
            acc["trig_lsd_trig"][pair_idx, a_idx] = compute_lsd(audio_mix_trig, audio_trig)
            lsd_d = compute_lsd_decomposed(audio_mix_trig, audio_trig)
            acc["trig_lsd_active_trig"][pair_idx, a_idx] = lsd_d["lsd_active"]
            acc["trig_lsd_silent_trig"][pair_idx, a_idx] = lsd_d["lsd_silent"]
            acc["trig_lsd_energy_trig"][pair_idx, a_idx] = lsd_d["lsd_energy_weighted"]
            acc["trig_ref_sparsity"][pair_idx, a_idx] = lsd_d["sparsity"]
            acc["trig_energy"][pair_idx, a_idx] = rms_energy(audio_trig) / (rms_mean + 1e-12)
            if msstft_loss_fn is not None:
                with torch.no_grad():
                    pred_t = torch.from_numpy(audio_trig.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    tgt_t = torch.from_numpy(audio_mix_trig.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    _, mag_loss = msstft_loss_fn(pred_t, tgt_t, use_phase=False, return_components=True)
                acc["trig_msstft_trig"][pair_idx, a_idx] = float(mag_loss.item())
            if model_pair_dir is not None:
                _save_wav(model_pair_dir / "trig", f"alpha_{alpha:.2f}.wav", audio_trig)

        if model_pair_dir is not None:
            lerp_a = model_pair_dir / "lerp" / "alpha_0.00.wav"
            lerp_b = model_pair_dir / "lerp" / "alpha_1.00.wav"
            if lerp_a.exists():
                shutil.copy2(str(lerp_a), str(model_pair_dir / "reconstructed_A.wav"))
            if lerp_b.exists():
                shutil.copy2(str(lerp_b), str(model_pair_dir / "reconstructed_B.wav"))

    # ---- Aggregate ----
    def _drop_nan(*arrays):
        mask = ~np.isnan(arrays[0]).any(axis=1)
        for a in arrays[1:]:
            mask &= ~np.isnan(a).any(axis=1)
        return [a[mask] for a in arrays]

    acc_keys = list(acc.keys())
    # Drop pairs only on the PRIMARY keys; the sparse-audio diagnostic keys
    # legitimately contain NaN (e.g. lsd_silent when the reference has no
    # silent frames) and must not invalidate a pair.
    _PRIMARY = {k for k in acc_keys if k not in ("trig_lsd_silent_trig",)}
    mask = ~np.isnan(acc["lerp_lsd_lerp"]).any(axis=1)
    for k in sorted(_PRIMARY - {"lerp_lsd_lerp"}):
        mask &= ~np.isnan(acc[k]).any(axis=1)
    valid = [acc[k][mask] for k in acc_keys]
    valid_map = {k: v for k, v in zip(acc_keys, valid)}
    n_valid = valid[0].shape[0]

    def _mean_std(key: str) -> Tuple[List[float], List[float]]:
        arr = valid_map[key]
        return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

    def _nanmean_std(key: str) -> Tuple[List[float], List[float]]:
        """NaN-safe mean/std (lsd_silent is NaN for pairs with no silent frames)."""
        arr = valid_map[key].astype(np.float64)
        with np.errstate(invalid="ignore"):
            return (
                np.nanmean(arr, axis=0).tolist(),
                np.nanstd(arr, axis=0).tolist(),
            )

    gt_valid = _drop_nan(gt_lerp_energy, gt_trig_energy)

    trig_lsd_curve = np.asarray(_mean_std("trig_lsd_trig")[0])
    trig_energy_curve = np.asarray(_mean_std("trig_energy")[0])
    gt_trig_energy_curve = np.asarray(gt_valid[1].mean(axis=0))

    def _curve_stat(values: np.ndarray, fn) -> float:
        return float(fn(values)) if values.size else float("nan")

    return {
        "label": model.name,
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "valid_pairs": n_valid,
        "total_pairs": len(pairs),
        "alphas": alphas,
        "mean_lerp_lsd": _mean_std("lerp_lsd_lerp")[0],       # primary target (lerp)
        "std_lerp_lsd": _mean_std("lerp_lsd_lerp")[1],
        "mean_lerp_lsd_vs_trig": _mean_std("lerp_lsd_trig")[0],
        "std_lerp_lsd_vs_trig": _mean_std("lerp_lsd_trig")[1],
        "mean_slerp_lsd": _mean_std("slerp_lsd_lerp")[0],     # primary target (lerp, magnitude-lerp)
        "std_slerp_lsd": _mean_std("slerp_lsd_lerp")[1],
        "mean_slerp_lsd_vs_trig": _mean_std("slerp_lsd_trig")[0],
        "std_slerp_lsd_vs_trig": _mean_std("slerp_lsd_trig")[1],
        "mean_trig_lsd": _mean_std("trig_lsd_trig")[0],       # primary target (trig)
        "std_trig_lsd": _mean_std("trig_lsd_trig")[1],
        "mean_trig_lsd_vs_lerp": _mean_std("trig_lsd_lerp")[0],
        "std_trig_lsd_vs_lerp": _mean_std("trig_lsd_lerp")[1],
        # ---- Sparse-audio-robust variants (primary trig target) ----
        "mean_trig_lsd_active": _nanmean_std("trig_lsd_active_trig")[0],   # active-region LSD
        "std_trig_lsd_active": _nanmean_std("trig_lsd_active_trig")[1],
        "mean_trig_lsd_silent": _nanmean_std("trig_lsd_silent_trig")[0],   # silence-only LSD
        "mean_trig_lsd_energy_weighted": _nanmean_std("trig_lsd_energy_trig")[0],
        "mean_trig_ref_sparsity": _mean_std("trig_ref_sparsity")[0],    # ref-mix sparsity
        # ---- Training-matched MS-STFT cross-check (when requested) ----
        **(
            {"mean_trig_msstft": _mean_std("trig_msstft_trig")[0]}
            if msstft_loss_fn is not None else {}
        ),
        "mean_lerp_energy": _mean_std("lerp_energy")[0],
        "std_lerp_energy": _mean_std("lerp_energy")[1],
        "mean_slerp_energy": _mean_std("slerp_energy")[0],
        "std_slerp_energy": _mean_std("slerp_energy")[1],
        "mean_trig_energy": _mean_std("trig_energy")[0],
        "std_trig_energy": _mean_std("trig_energy")[1],
        "ground_truth_lerp_energy_mean": gt_valid[0].mean(axis=0).tolist(),
        "ground_truth_trig_energy_mean": gt_valid[1].mean(axis=0).tolist(),
        # ---- trigMix curve-level metrics (consistency along the whole path) ----
        "lsd_curve_smoothness_trig": _curve_stat(trig_lsd_curve, np.var),
        "lsd_curve_range_trig": _curve_stat(trig_lsd_curve,
                                            lambda v: np.max(v) - np.min(v)),
        "energy_curve_flatness_trig": _curve_stat(trig_energy_curve, np.std),
        "energy_curve_gap_trig": _curve_stat(
            np.abs(trig_energy_curve - gt_trig_energy_curve), np.mean),
    }


# =============================================================================
#  Legacy two-model pipeline (kept for CLI backward compatibility)
# =============================================================================

def run_lims(
    ckpt_baseline: str,
    ckpt_phase: str,
    data_dirs=None,
    cache_dir: Optional[str] = None,
    device: torch.device | None = None,
    output_dir: str = "evaluation/lims_outputs",
    num_pairs: int = 100,
    seed: int = 42,
    pair_mode: str = "random",
    model_type: str | None = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    save_audio: bool = True,
) -> None:
    """Two-checkpoint comparison wrapper around evaluate_lims_checkpoint.

    Both checkpoints are scored on the SAME deterministic validation pair set,
    then a comparison JSON + plots are written to the standard paths.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_samples = int(CLIP_DURATION * sample_rate)
    alphas = np.linspace(0.0, 1.0, INTERP_STEPS).tolist()

    # ---- Shared validation stems + identical pairs for both models ----
    print("[1/5] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(data_dirs, cache_dir, sample_rate=sample_rate)
    print(f"  Found {len(stems)} validation stems")
    pairs = generate_pairs(stems, num_pairs, seed=seed, mode=pair_mode)
    print(f"  Generated {len(pairs)} pairs (mode={pair_mode})")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[2/5] Running LIMS on baseline …")
    agg_base = evaluate_lims_checkpoint(
        ckpt_baseline, stems, device,
        num_pairs=num_pairs, seed=seed, pair_mode=pair_mode,
        model_type=model_type, sample_rate=sample_rate,
        clip_duration=CLIP_DURATION, interp_steps=INTERP_STEPS,
        pairs=pairs,
        save_wav_dir=out_root if save_audio else None,
    )
    print("[3/5] Running LIMS on phase/compare …")
    agg_phase = evaluate_lims_checkpoint(
        ckpt_phase, stems, device,
        num_pairs=num_pairs, seed=seed, pair_mode=pair_mode,
        model_type=model_type, sample_rate=sample_rate,
        clip_duration=CLIP_DURATION, interp_steps=INTERP_STEPS,
        pairs=pairs,
        save_wav_dir=out_root if save_audio else None,
    )

    # ---- Save JSON ----
    print("[4/5] Saving summary …")
    summary = {
        "description": "AE LIMS test — decoder topological additivity "
                       "(Energy Drop + Constant-Power interpolation)",
        "timestamp": datetime.now().isoformat(),
        "clip_duration_s": CLIP_DURATION,
        "interp_steps": INTERP_STEPS,
        "pair_mode": pair_mode,
        "data_dirs": list(data_dirs) if data_dirs is not None else DEFAULT_DATA_DIRS,
        "baseline": agg_base,
        "phase": agg_phase,
        "ground_truth": {
            "description": "Ideal audio-space mixing energy curves "
                           "(computed directly from raw waveforms)",
            "alphas": alphas,
            "audio_lerp_mean_energy": agg_base["ground_truth_lerp_energy_mean"],
            "audio_trig_mean_energy": agg_base["ground_truth_trig_energy_mean"],
        },
    }
    json_path = _EVAL_DIR / "lims_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  📄 JSON → {json_path}")

    # ---- Plots ----
    print("[5/5] Plotting LSD & Energy comparison …")
    _plot_lsd_comparison(agg_base, agg_phase, _EVAL_DIR / "lims_plot.png")
    _plot_energy_comparison(agg_base, agg_phase, _EVAL_DIR / "lims_energy_plot.png")
    print("✓ LIMS test complete.")


# =============================================================================
#  Plotting
# =============================================================================

def _plot_lsd_comparison(base: Dict, phase: Dict, out_path: Path) -> None:
    alphas = np.array(base["alphas"])
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    colors = {"Baseline": "#e74c3c", "Phase": "#2980b9"}
    interp_styles = {
        "lerp":  ("Lerp",  "o-"),
        "slerp": ("Slerp", "s--"),
        "trig":  ("Trig",  "^-."),
    }
    for ax_idx, (agg, label) in enumerate([(base, "Baseline"), (phase, "Phase")]):
        ax = axes[ax_idx]
        c = colors[label]
        for key, (disp_name, style) in interp_styles.items():
            mean = np.array(agg[f"mean_{key}_lsd"])
            std = np.array(agg[f"std_{key}_lsd"])
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.10, color=c)
            ax.plot(alphas, mean, style, color=c, linewidth=2, markersize=5,
                    label=disp_name)
        ax.set_title(f"{label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Interpolation α", fontsize=12)
        ax.set_ylabel("Log-Spectral Distance (dB)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
    fig.suptitle("AE — LSD: decoded(interp(Z_A, Z_B, α)) vs ideal mix",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 LSD Plot → {out_path}")


def _plot_energy_comparison(base: Dict, phase: Dict, out_path: Path) -> None:
    alphas = np.array(base["alphas"])
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    colors = {"Baseline": "#e74c3c", "Phase": "#2980b9"}
    interp_styles = {
        "lerp":  ("Latent Lerp",  "o-"),
        "slerp": ("Latent Slerp", "s--"),
        "trig":  ("Latent Trig",  "^-."),
    }
    for ax_idx, (agg, label) in enumerate([(base, "Baseline"), (phase, "Phase")]):
        ax = axes[ax_idx]
        c = colors[label]
        for key, (disp_name, style) in interp_styles.items():
            mean = np.array(agg[f"mean_{key}_energy"])
            std = np.array(agg[f"std_{key}_energy"])
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.10, color=c)
            ax.plot(alphas, mean, style, color=c, linewidth=2, markersize=5,
                    label=disp_name)
        ax.plot(alphas, base["ground_truth_lerp_energy_mean"], "k--", linewidth=2,
                label="Audio LERP (ideal)", alpha=0.7)
        ax.plot(alphas, base["ground_truth_trig_energy_mean"], "k:", linewidth=2,
                label="Audio TRIG (ideal)", alpha=0.7)
        ax.set_title(f"{label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Interpolation α", fontsize=12)
        ax.set_ylabel("Normalised RMS Energy", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    fig.suptitle("AE — Energy Drop: decoded(interp(Z_A, Z_B, α)) vs ideal mixes",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Energy Plot → {out_path}")


# =============================================================================
#  CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AE LIMS Test — Decoder Topological Additivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt_baseline", required=True,
                   help="First checkpoint (e.g. baseline without mixup).")
    p.add_argument("--ckpt_phase", required=True,
                   help="Second checkpoint (e.g. mixup-trained).")
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                   nargs="+", default=None,
                   help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                        "training set.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Preprocessed cache base dir "
                        "(default: /data/preprocessed_cache).")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["conv", "transformer"],
                   help="Model architecture. Auto-detected from checkpoint if omitted.")
    p.add_argument("--num_pairs", type=int, default=100,
                   help="Number of stem pairs (default: 100).")
    p.add_argument("--pair_mode", default="random",
                   choices=["random", "intra", "inter"],
                   help="Pair mode (default: random).")
    p.add_argument("--output_dir", default="evaluation/lims_outputs",
                   help="Root output directory for interpolated audio.")
    p.add_argument("--no-save-audio", dest="save_audio", action="store_false",
                   default=True, help="Skip saving interpolated .wav files.")
    p.add_argument("--device", default="cuda",
                   help="Device (default: cuda).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=MODEL_SAMPLE_RATE,
                   help=f"Target sample rate in Hz (default: {MODEL_SAMPLE_RATE}).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    run_lims(
        ckpt_baseline=args.ckpt_baseline,
        ckpt_phase=args.ckpt_phase,
        data_dirs=args.data_dirs,
        cache_dir=args.cache_dir,
        device=device,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        seed=args.seed,
        pair_mode=args.pair_mode,
        model_type=args.model_type,
        sample_rate=args.sample_rate,
        save_audio=args.save_audio,
    )
