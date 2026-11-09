#!/usr/bin/env python3
"""
Comparative interpolation-path analysis for two checkpoints over 1000 pairs.

Metrics (replacing the original LSD protocol):
  * Path-following — SI-SDR (dB, higher = decoded interpolation matches the
    ideal audio mix more closely) and MS-STFT (magnitude-only SAMELoss,
    lower = closer to the ideal mix). Both are computed for Lerp (linear)
    and Trig (constant-power) at every α ∈ {0.0, 0.1, ..., 1.0}, so the
    WHOLE path is scored, not only α=0.5.
    SI-SDR and MS-STFT conventions are identical to
    evaluation/generalized_interpolation_eval.py.

  * Energy similarity — normalised RMS(decoded(interp(α))) vs the
    normalised RMS of the real audio ideal mix at every α, for both Lerp
    and Trig. Smaller |decoded_energy − ideal_energy| = the model
    preserves the mix energy closer to the real audio mix energy along
    the whole path. (This metric is independent of the spectral metric
    used for path-following.)

Both checkpoints are scored on the SAME 1000 deterministic pairs (seed 42)
so the comparison is paired and fair.

Outputs:
  evaluation/interp_path_compare_1000/
      best_full_vs_implicit_v2_1000_sisdr_msstft.json
      REPORT_sisdr_msstft.md
      path_compare_1000_sisdr_msstft.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss

# =============================================================================
#  Defaults
# =============================================================================

CKPT_A = "training/checkpoints/transformer_best_full_best.pt"
CKPT_B = "training/checkpoints/transformer_implicit_full_v2_best.pt"
LABEL_A = "best_full"
LABEL_B = "implicit_v2"
NUM_PAIRS = 1000
SEED = 42
INTERP_STEPS = 11
OUT_DIR = Path("evaluation/interp_path_compare_1000")


# =============================================================================
#  Interpolation schemes
# =============================================================================

def lerp_z(z_a, z_b, alpha):
    return (1.0 - alpha) * z_a + alpha * z_b


def trig_z(z_a, z_b, alpha):
    theta = alpha * (np.pi / 2.0)
    return np.cos(theta) * z_a + np.sin(theta) * z_b


def lerp_audio(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b


def trig_audio(a, b, alpha):
    theta = alpha * (np.pi / 2.0)
    return np.cos(theta) * a + np.sin(theta) * b


# =============================================================================
#  Metrics (identical conventions to generalized_interpolation_eval.py)
# =============================================================================

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


def ms_stft_of(pred_np, tgt_np, loss_fn, device) -> float:
    """Magnitude-only MS-STFT (SAMELoss) between two mono numpy arrays."""
    pred_t = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, mag = loss_fn(pred_t, tgt_t, use_phase=False, return_components=True)
    return float(mag.item())


def rms_energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


# =============================================================================
#  Per-checkpoint evaluation
# =============================================================================

def evaluate_checkpoint(ckpt_path, stems, pairs, device, loss_fn, interp_steps=INTERP_STEPS):
    """Run the SI-SDR / MS-STFT / energy interpolation-path battery for ONE checkpoint."""
    model = AutoencoderWrapper(ckpt_path, device)
    name = Path(ckpt_path).stem
    alphas = np.linspace(0.0, 1.0, interp_steps).tolist()
    n = len(pairs)
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)

    # accumulators: scheme -> per-(pair, alpha) arrays
    acc = {
        "lerp_sisdr":  np.full((n, interp_steps), np.nan),
        "lerp_msstft": np.full((n, interp_steps), np.nan),
        "trig_sisdr":  np.full((n, interp_steps), np.nan),
        "trig_msstft": np.full((n, interp_steps), np.nan),
        "lerp_energy": np.full((n, interp_steps), np.nan),
        "trig_energy": np.full((n, interp_steps), np.nan),
    }
    gt_lerp_energy = np.full((n, interp_steps), np.nan)
    gt_trig_energy = np.full((n, interp_steps), np.nan)
    recon_sisdr = np.full((n,), np.nan)
    recon_msstft = np.full((n,), np.nan)

    print(f"  {name} ({model.model_type}): {n} pairs on {device}")

    for p_idx, (i, j) in enumerate(pairs):
        try:
            a = stem_to_numpy(stems[i][0], clip_samples, centre=True)
            b = stem_to_numpy(stems[j][0], clip_samples, centre=True)
        except Exception as exc:
            print(f"  ⚠ Skip pair {p_idx}: load error – {exc}")
            continue
        orig_len = len(a)
        rms_a = rms_energy(a)
        rms_b = rms_energy(b)
        rms_mean = (rms_a + rms_b) / 2.0

        try:
            z_a = model.encode(a)
            z_b = model.encode(b)
        except Exception as exc:
            print(f"  ⚠ Skip pair {p_idx} [{name}]: encode – {exc}")
            continue
        if z_a.shape[-1] != z_b.shape[-1]:
            mlen = min(z_a.shape[-1], z_b.shape[-1])
            z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]

        # endpoint reconstruction fidelity (same convention as generalized_interpolation_eval)
        ra = model.decode(z_a, orig_len)
        rb = model.decode(z_b, orig_len)
        recon_sisdr[p_idx] = float(np.nanmean([si_sdr(a, ra), si_sdr(b, rb)]))
        recon_msstft[p_idx] = float(np.nanmean([
            ms_stft_of(ra, a, loss_fn, device), ms_stft_of(rb, b, loss_fn, device)
        ]))

        for a_idx, alpha in enumerate(alphas):
            # ideal audio mixes
            mix_lerp = lerp_audio(a, b, alpha)
            mix_trig = trig_audio(a, b, alpha)
            gt_lerp_energy[p_idx, a_idx] = rms_energy(mix_lerp) / (rms_mean + 1e-12)
            gt_trig_energy[p_idx, a_idx] = rms_energy(mix_trig) / (rms_mean + 1e-12)

            # Lerp decode
            z_lerp = lerp_z(z_a, z_b, alpha)
            audio_lerp = model.decode(z_lerp, orig_len)
            acc["lerp_sisdr"][p_idx, a_idx] = si_sdr(mix_lerp, audio_lerp)
            acc["lerp_msstft"][p_idx, a_idx] = ms_stft_of(audio_lerp, mix_lerp, loss_fn, device)
            acc["lerp_energy"][p_idx, a_idx] = rms_energy(audio_lerp) / (rms_mean + 1e-12)

            # Trig decode
            z_trig = trig_z(z_a, z_b, alpha)
            audio_trig = model.decode(z_trig, orig_len)
            acc["trig_sisdr"][p_idx, a_idx] = si_sdr(mix_trig, audio_trig)
            acc["trig_msstft"][p_idx, a_idx] = ms_stft_of(audio_trig, mix_trig, loss_fn, device)
            acc["trig_energy"][p_idx, a_idx] = rms_energy(audio_trig) / (rms_mean + 1e-12)

        if (p_idx + 1) % 100 == 0 or p_idx == n - 1:
            print(f"    pair {p_idx + 1}/{n}  recon-SI-SDR {np.nanmean(recon_sisdr[:p_idx + 1]):.2f} dB")

    # ---- aggregate ----
    # keep only pairs with all-finite primary keys
    primary = np.stack([acc["lerp_sisdr"], acc["trig_sisdr"]], axis=0)
    mask = ~np.isnan(primary).any(axis=(0, 2))
    if mask.sum() == 0:
        raise RuntimeError(f"No valid pairs for {name}")

    def mean_std(key):
        arr = acc[key][mask]
        return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

    def nanmean(key):  # for recon arrays already 1-D
        return float(np.nanmean(acc[key][mask])) if np.isfinite(np.nanmean(acc[key][mask])) else float("nan")

    gt_lerp_v = gt_lerp_energy[mask]
    gt_trig_v = gt_trig_energy[mask]
    recon_sisdr_v = recon_sisdr[mask]
    recon_msstft_v = recon_msstft[mask]

    lerp_sisdr_m, lerp_sisdr_s = mean_std("lerp_sisdr")
    trig_sisdr_m, trig_sisdr_s = mean_std("trig_sisdr")
    lerp_msstft_m, lerp_msstft_s = mean_std("lerp_msstft")
    trig_msstft_m, trig_msstft_s = mean_std("trig_msstft")
    lerp_energy_m, lerp_energy_s = mean_std("lerp_energy")
    trig_energy_m, trig_energy_s = mean_std("trig_energy")

    arr_lerp_sisdr = np.asarray(lerp_sisdr_m)
    arr_trig_sisdr = np.asarray(trig_sisdr_m)
    arr_lerp_ms = np.asarray(lerp_msstft_m)
    arr_trig_ms = np.asarray(trig_msstft_m)
    dec_lerp_e = np.asarray(lerp_energy_m)
    dec_trig_e = np.asarray(trig_energy_m)
    gt_lerp_e = gt_lerp_v.mean(axis=0)
    gt_trig_e = gt_trig_v.mean(axis=0)

    interior = slice(1, -1)  # α ∈ {0.1..0.9}

    return {
        "label": name,
        "checkpoint": ckpt_path,
        "model_type": model.model_type,
        "valid_pairs": int(mask.sum()),
        "total_pairs": n,
        "alphas": alphas,
        "recon_sisdr_mean": float(np.nanmean(recon_sisdr_v)),
        "recon_msstft_mean": float(np.nanmean(recon_msstft_v)),
        # SI-SDR (higher better)
        "lerp_sisdr_mean": lerp_sisdr_m, "lerp_sisdr_std": lerp_sisdr_s,
        "trig_sisdr_mean": trig_sisdr_m, "trig_sisdr_std": trig_sisdr_s,
        "lerp_sisdr_interior_mean": float(arr_lerp_sisdr[interior].mean()),
        "trig_sisdr_interior_mean": float(arr_trig_sisdr[interior].mean()),
        "lerp_sisdr_whole_mean": float(arr_lerp_sisdr.mean()),
        "trig_sisdr_whole_mean": float(arr_trig_sisdr.mean()),
        # MS-STFT (lower better); ratio mix/recon (<1 = mixing as good as recon)
        "lerp_msstft_mean": lerp_msstft_m, "lerp_msstft_std": lerp_msstft_s,
        "trig_msstft_mean": trig_msstft_m, "trig_msstft_std": trig_msstft_s,
        "lerp_msstft_interior_mean": float(arr_lerp_ms[interior].mean()),
        "trig_msstft_interior_mean": float(arr_trig_ms[interior].mean()),
        "lerp_msstft_whole_mean": float(arr_lerp_ms.mean()),
        "trig_msstft_whole_mean": float(arr_trig_ms.mean()),
        "lerp_msstft_ratio_of_means": float(np.nanmean(acc["lerp_msstft"][mask].mean()) / np.nanmean(recon_msstft_v)),
        "trig_msstft_ratio_of_means": float(np.nanmean(acc["trig_msstft"][mask].mean()) / np.nanmean(recon_msstft_v)),
        # Energy (gap, lower better)
        "lerp_energy_mean": lerp_energy_m, "trig_energy_mean": trig_energy_m,
        "lerp_energy_gap": np.abs(dec_lerp_e - gt_lerp_e).tolist(),
        "trig_energy_gap": np.abs(dec_trig_e - gt_trig_e).tolist(),
        "lerp_energy_interior_gap": float(np.abs(dec_lerp_e - gt_lerp_e)[interior].mean()),
        "trig_energy_interior_gap": float(np.abs(dec_trig_e - gt_trig_e)[interior].mean()),
        "lerp_energy_whole_gap": float(np.abs(dec_lerp_e - gt_lerp_e).mean()),
        "trig_energy_whole_gap": float(np.abs(dec_trig_e - gt_trig_e).mean()),
        "ground_truth_lerp_energy_mean": gt_lerp_e.tolist(),
        "ground_truth_trig_energy_mean": gt_trig_e.tolist(),
    }


# =============================================================================
#  Report
# =============================================================================

def _fmt(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a"
    return f"{x:.{n}f}"


def write_report(A, B, meta, out_md):
    """Write a Markdown comparative analysis."""
    L = [
        "# Interpolation-path comparative analysis (1000 audio pairs)",
        "",
        f"- **Generated:** {meta['timestamp']}",
        f"- **Pairs:** {meta['num_pairs']} (seed {meta['seed']}, {A['valid_pairs']} valid)",
        f"- **Interpolation steps:** {meta['interp_steps']} α ∈ "
        f"{[round(a,1) for a in meta['alphas']]}",
        f"- **Metrics:** SI-SDR (dB, ↑ better), MS-STFT (magnitude-only SAMELoss, "
        f"↓ better), energy gap (normalised RMS, ↓ better)",
        f"- **Model A:** `{meta['ckpt_a']}`  (recon SI-SDR {_fmt(A['recon_sisdr_mean'],2)} dB)",
        f"- **Model B:** `{meta['ckpt_b']}`  (recon SI-SDR {_fmt(B['recon_sisdr_mean'],2)} dB)",
        "",
        "## 1. Path-following — SI-SDR vs ideal mix along the whole path",
        "",
        "SI-SDR (dB, higher = decoded interpolation matches the ideal audio mix "
        "more closely). *Interior* = α ∈ {0.1…0.9}; *whole-path* also includes "
        "the endpoint reconstructions (α=0,1).",
        "",
        "| Metric | Model A (best_full) | Model B (implicit_v2) | Better |",
        "|---|---|---|---|",
    ]

    def row(label, va, vb, higher_better=True):
        if va is None or vb is None:
            return f"| {label} | {_fmt(va)} | {_fmt(vb)} | n/a |"
        diff = vb - va
        if abs(diff) < 1e-9:
            winner = "tie"
        elif (diff > 0) == higher_better:
            winner = "**B**"
        else:
            winner = "**A**"
        return f"| {label} | {_fmt(va)} | {_fmt(vb)} | {winner} |"

    L += [
        row("Lerp — interior mean SI-SDR (↑)", A["lerp_sisdr_interior_mean"], B["lerp_sisdr_interior_mean"]),
        row("Lerp — whole-path mean SI-SDR (↑)", A["lerp_sisdr_whole_mean"], B["lerp_sisdr_whole_mean"]),
        row("Trig — interior mean SI-SDR (↑)", A["trig_sisdr_interior_mean"], B["trig_sisdr_interior_mean"]),
        row("Trig — whole-path mean SI-SDR (↑)", A["trig_sisdr_whole_mean"], B["trig_sisdr_whole_mean"]),
        "",
        "### 1a. Per-α SI-SDR curves (dB, ↑)",
        "",
        "| α | Lerp A | Lerp B | Trig A | Trig B |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(meta["alphas"]):
        L.append(f"| {a:.1f} | {_fmt(A['lerp_sisdr_mean'][i])} | "
                 f"{_fmt(B['lerp_sisdr_mean'][i])} | "
                 f"{_fmt(A['trig_sisdr_mean'][i])} | "
                 f"{_fmt(B['trig_sisdr_mean'][i])} |")

    L += [
        "",
        "## 2. Path-following — MS-STFT vs ideal mix along the whole path",
        "",
        "MS-STFT = magnitude-only multi-res STFT loss (lower = decoded "
        "interpolation closer to the ideal mix). Ratio = mean(mix MS-STFT) / "
        "mean(recon MS-STFT); <1 means mixing is as good as reconstruction.",
        "",
        "| Metric | Model A (best_full) | Model B (implicit_v2) | Better |",
        "|---|---|---|---|",
    ]
    L += [
        row("Lerp — interior mean MS-STFT (↓)", A["lerp_msstft_interior_mean"], B["lerp_msstft_interior_mean"], higher_better=False),
        row("Lerp — whole-path mean MS-STFT (↓)", A["lerp_msstft_whole_mean"], B["lerp_msstft_whole_mean"], higher_better=False),
        row("Trig — interior mean MS-STFT (↓)", A["trig_msstft_interior_mean"], B["trig_msstft_interior_mean"], higher_better=False),
        row("Trig — whole-path mean MS-STFT (↓)", A["trig_msstft_whole_mean"], B["trig_msstft_whole_mean"], higher_better=False),
        row("Lerp — MS-STFT ratio mix/recon (↓)", A["lerp_msstft_ratio_of_means"], B["lerp_msstft_ratio_of_means"], higher_better=False),
        row("Trig — MS-STFT ratio mix/recon (↓)", A["trig_msstft_ratio_of_means"], B["trig_msstft_ratio_of_means"], higher_better=False),
        "",
        "### 2a. Per-α MS-STFT curves (↓)",
        "",
        "| α | Lerp A | Lerp B | Trig A | Trig B |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(meta["alphas"]):
        L.append(f"| {a:.1f} | {_fmt(A['lerp_msstft_mean'][i])} | "
                 f"{_fmt(B['lerp_msstft_mean'][i])} | "
                 f"{_fmt(A['trig_msstft_mean'][i])} | "
                 f"{_fmt(B['trig_msstft_mean'][i])} |")

    L += [
        "",
        "## 3. Energy similarity — decoded vs real audio mix energy",
        "",
        "Energy gap = |normalised-RMS(decoded) − normalised-RMS(ideal mix)| "
        "(lower = model preserves the real mix energy more closely).",
        "",
        "| Metric | Model A (best_full) | Model B (implicit_v2) | Better |",
        "|---|---|---|---|",
        row("Lerp — interior mean energy gap (↓)", A["lerp_energy_interior_gap"], B["lerp_energy_interior_gap"], higher_better=False),
        row("Lerp — whole-path mean energy gap (↓)", A["lerp_energy_whole_gap"], B["lerp_energy_whole_gap"], higher_better=False),
        row("Trig — interior mean energy gap (↓)", A["trig_energy_interior_gap"], B["trig_energy_interior_gap"], higher_better=False),
        row("Trig — whole-path mean energy gap (↓)", A["trig_energy_whole_gap"], B["trig_energy_whole_gap"], higher_better=False),
        "",
        "### 3a. Per-α energy curves (decoded vs ideal)",
        "",
        "| α | Lerp A | Lerp ideal | Lerp B | Trig A | Trig ideal | Trig B |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, a in enumerate(meta["alphas"]):
        L.append(
            f"| {a:.1f} | {_fmt(A['lerp_energy_mean'][i])} | "
            f"{_fmt(A['ground_truth_lerp_energy_mean'][i])} | "
            f"{_fmt(B['lerp_energy_mean'][i])} | "
            f"{_fmt(A['trig_energy_mean'][i])} | "
            f"{_fmt(A['ground_truth_trig_energy_mean'][i])} | "
            f"{_fmt(B['trig_energy_mean'][i])} |"
        )

    L += [
        "",
        "### 3b. Per-α energy gaps",
        "",
        "| α | Lerp gap A | Lerp gap B | Trig gap A | Trig gap B |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(meta["alphas"]):
        L.append(
            f"| {a:.1f} | {_fmt(A['lerp_energy_gap'][i])} | "
            f"{_fmt(B['lerp_energy_gap'][i])} | "
            f"{_fmt(A['trig_energy_gap'][i])} | "
            f"{_fmt(B['trig_energy_gap'][i])} |"
        )

    L += ["", "## 4. Verdict", ""]
    name = {"A": "best_full", "B": "implicit_v2"}
    # path-following winners from interior mean SI-SDR
    pf_sisdr_lerp = "A" if A["lerp_sisdr_interior_mean"] >= B["lerp_sisdr_interior_mean"] else "B"
    pf_sisdr_trig = "A" if A["trig_sisdr_interior_mean"] >= B["trig_sisdr_interior_mean"] else "B"
    pf_msstft_lerp = "A" if A["lerp_msstft_interior_mean"] <= B["lerp_msstft_interior_mean"] else "B"
    pf_msstft_trig = "A" if A["trig_msstft_interior_mean"] <= B["trig_msstft_interior_mean"] else "B"
    en_lerp = "A" if A["lerp_energy_interior_gap"] <= B["lerp_energy_interior_gap"] else "B"
    en_trig = "A" if A["trig_energy_interior_gap"] <= B["trig_energy_interior_gap"] else "B"
    L.append(f"- **Path-following — Lerp, SI-SDR (↑):** Model **{pf_sisdr_lerp}** ({name[pf_sisdr_lerp]}) "
             f"(interior mean {_fmt(A['lerp_sisdr_interior_mean'] if pf_sisdr_lerp=='A' else B['lerp_sisdr_interior_mean'])} "
             f"vs {_fmt(B['lerp_sisdr_interior_mean'] if pf_sisdr_lerp=='A' else A['lerp_sisdr_interior_mean'])})")
    L.append(f"- **Path-following — Trig, SI-SDR (↑):** Model **{pf_sisdr_trig}** ({name[pf_sisdr_trig]}) "
             f"(interior mean {_fmt(A['trig_sisdr_interior_mean'] if pf_sisdr_trig=='A' else B['trig_sisdr_interior_mean'])} "
             f"vs {_fmt(B['trig_sisdr_interior_mean'] if pf_sisdr_trig=='A' else A['trig_sisdr_interior_mean'])})")
    L.append(f"- **Path-following — Lerp, MS-STFT (↓):** Model **{pf_msstft_lerp}** ({name[pf_msstft_lerp]}) "
             f"(interior mean {_fmt(A['lerp_msstft_interior_mean'] if pf_msstft_lerp=='A' else B['lerp_msstft_interior_mean'])} "
             f"vs {_fmt(B['lerp_msstft_interior_mean'] if pf_msstft_lerp=='A' else A['lerp_msstft_interior_mean'])})")
    L.append(f"- **Path-following — Trig, MS-STFT (↓):** Model **{pf_msstft_trig}** ({name[pf_msstft_trig]}) "
             f"(interior mean {_fmt(A['trig_msstft_interior_mean'] if pf_msstft_trig=='A' else B['trig_msstft_interior_mean'])} "
             f"vs {_fmt(B['trig_msstft_interior_mean'] if pf_msstft_trig=='A' else A['trig_msstft_interior_mean'])})")
    L.append(f"- **Energy similarity — Lerp (↓):** Model **{en_lerp}** ({name[en_lerp]}) "
             f"(interior gap {_fmt(A['lerp_energy_interior_gap'] if en_lerp=='A' else B['lerp_energy_interior_gap'])} "
             f"vs {_fmt(B['lerp_energy_interior_gap'] if en_lerp=='A' else A['lerp_energy_interior_gap'])})")
    L.append(f"- **Energy similarity — Trig (↓):** Model **{en_trig}** ({name[en_trig]}) "
             f"(interior gap {_fmt(A['trig_energy_interior_gap'] if en_trig=='A' else B['trig_energy_interior_gap'])} "
             f"vs {_fmt(B['trig_energy_interior_gap'] if en_trig=='A' else A['trig_energy_interior_gap'])})")
    L.append("")
    L.append("> Higher SI-SDR is better; lower MS-STFT and lower energy gap are "
             "better. “Interior” = α ∈ {0.1…0.9} (the interpolation path between "
             "the endpoints); “whole-path” also counts the α=0/1 endpoint "
             "reconstructions.")

    out_md.write_text("\n".join(L))


# =============================================================================
#  Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_a", default=CKPT_A)
    p.add_argument("--ckpt_b", default=CKPT_B)
    p.add_argument("--num_pairs", type=int, default=NUM_PAIRS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Collecting validation stems …")
    stems = collect_validation_stems()
    print(f"  {len(stems)} validation stems")
    print(f"[2/4] Generating {args.num_pairs} pairs (seed {args.seed}) …")
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    print(f"  {len(pairs)} pairs")
    loss_fn = SAMELoss().to(device)

    results = {}
    for label, ckpt in [("A", args.ckpt_a), ("B", args.ckpt_b)]:
        print(f"[3/4] Evaluating model {label}: {ckpt}")
        t0 = time.time()
        results[label] = evaluate_checkpoint(ckpt, stems, pairs, device, loss_fn)
        print(f"  done in {time.time() - t0:.1f}s — valid_pairs={results[label]['valid_pairs']}")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "num_pairs": args.num_pairs, "seed": args.seed,
        "interp_steps": INTERP_STEPS, "alphas": results["A"]["alphas"],
    }

    json_path = out_dir / "best_full_vs_implicit_v2_1000_sisdr_msstft.json"
    with open(json_path, "w") as fp:
        json.dump({"meta": meta, "model_A": results["A"], "model_B": results["B"]}, fp, indent=2)
    print(f"[4/4] JSON → {json_path}")

    md_path = out_dir / "REPORT_sisdr_msstft.md"
    write_report(results["A"], results["B"], meta, md_path)
    print(f"      REPORT → {md_path}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A, B = results["A"], results["B"]
    alphas = np.asarray(meta["alphas"])
    cA, cB = "#e74c3c", "#2980b9"
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    # Row 0: SI-SDR
    for ax, (la, lb, title) in zip([axes[0, 0], axes[0, 1]],
            [(A["lerp_sisdr_mean"], B["lerp_sisdr_mean"], "Lerp (linear) — SI-SDR vs ideal mix"),
             (A["trig_sisdr_mean"], B["trig_sisdr_mean"], "Trig (constant-power) — SI-SDR vs ideal mix")]):
        ax.plot(alphas, la, "o-", color=cA, lw=2, ms=5, label="A: best_full")
        ax.plot(alphas, lb, "s--", color=cB, lw=2, ms=5, label="B: implicit_v2")
        ax.set_xlabel("Interpolation α"); ax.set_ylabel("SI-SDR (dB)"); ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(-0.02, 1.02)
    # Row 1: MS-STFT
    for ax, (la, lb, title) in zip([axes[1, 0], axes[1, 1]],
            [(A["lerp_msstft_mean"], B["lerp_msstft_mean"], "Lerp (linear) — MS-STFT vs ideal mix"),
             (A["trig_msstft_mean"], B["trig_msstft_mean"], "Trig (constant-power) — MS-STFT vs ideal mix")]):
        ax.plot(alphas, la, "o-", color=cA, lw=2, ms=5, label="A: best_full")
        ax.plot(alphas, lb, "s--", color=cB, lw=2, ms=5, label="B: implicit_v2")
        ax.set_xlabel("Interpolation α"); ax.set_ylabel("MS-STFT (magnitude loss)"); ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(-0.02, 1.02)
    # Row 2: Energy
    for ax, (da, db, ideal, title) in zip([axes[2, 0], axes[2, 1]],
            [(A["lerp_energy_mean"], B["lerp_energy_mean"], A["ground_truth_lerp_energy_mean"], "Lerp (linear) — energy vs ideal mix"),
             (A["trig_energy_mean"], B["trig_energy_mean"], A["ground_truth_trig_energy_mean"], "Trig (constant-power) — energy vs ideal mix")]):
        ax.plot(alphas, ideal, "k:", lw=2.5, label="ideal mix", alpha=0.8)
        ax.plot(alphas, da, "o-", color=cA, lw=2, ms=5, label="A: best_full")
        ax.plot(alphas, db, "s--", color=cB, lw=2, ms=5, label="B: implicit_v2")
        ax.set_xlabel("Interpolation α"); ax.set_ylabel("Normalised RMS energy"); ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(-0.02, 1.02); ax.axhline(1.0, color="gray", ls=":", alpha=0.4)
    fig.suptitle("Interpolation-path comparison (1000 pairs) — best_full (A) vs implicit_v2 (B)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    png = out_dir / "path_compare_1000_sisdr_msstft.png"
    fig.savefig(str(png), dpi=140, bbox_inches="tight")
    print(f"      Plot → {png}")
    print("Done.")


if __name__ == "__main__":
    main()
