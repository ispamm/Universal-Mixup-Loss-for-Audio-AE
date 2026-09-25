#!/usr/bin/env python3
"""
Toy AE — Latent Distance Analysis (Encoder Topological Homomorphism)
=====================================================================

Measures how well the toy autoencoder's **encoder** preserves the geometric
structure of the input space.

Core idea
---------
For a pair of audio clips (A, B), we compare two routes to a "mixed" latent:

  1. **Latent-space interpolation**:
       Z_latent_interp = interp(encode(A), encode(B), α)   [Lerp / Slerp / Trig]

  2. **Encode the audio-domain mixture**:
       Z_audio_mix = encode(mix(A, B))

If the encoder is topologically linear (a homomorphism), these two latent
vectors should be identical.  Deviations reveal non-linear warping.

Scheme/target consistency (fixed)
---------------------------------
Each latent interpolation scheme is compared against the audio mix that obeys
the *same* weighting law, matching the training-time additivity objective:

- **Lerp**  ↔ encode( (1−α)·A + α·B )           (linear mix)
- **Trig**  ↔ encode( cos(απ/2)·A + sin(απ/2)·B ) (constant-power mix —
  the exact mix the additivity loss trains on)
- **Slerp** ↔ encode( (1−α)·A + α·B )           (slerp lerps magnitudes)

Metrics (per α, per model)
--------------------------
  - MSE                — element-wise Euclidean distance between
    Z_interp and Z_encoded_mix (lower = more homomorphic).  This is the
    primary encoder-homomorphism probe (MSE captures both angle AND scale,
    unlike cosine similarity which is scale-blind).
  - Norm preservation  — ‖trigMix(Z_A, Z_B, α)‖ − ‖Z_A‖ (signed, per pair).
    Probes the constant-power property of the latent space induced by the
    trigMix loss: if trigMix is energy-preserving in latent space the error
    is ≈ 0 (headline value at α = 0.5).
  - Path length        — chord-sum ∫‖dZ/dα‖dα of the Lerp / Trig
    interpolation paths along a fine α-grid.  Shorter paths = more efficient
    (more direct) interpolation geometry; the Trig-vs-Lerp ratio shows how
    much longer the constant-power arc is than the straight line.

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``toy_evaluation.data``):
validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.

Outputs
-------
  toy_evaluation/latent_distance_summary.json
  toy_evaluation/latent_distance_plot.png

Usage
-----
    python toy_evaluation/latent_distance_toy.py \\
        --ckpt_baseline toy_training/checkpoints/baseline.pt \\
        --ckpt_phase toy_training/checkpoints/with_mixup.pt \\
        --num_pairs 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent            # toy_evaluation/
_REPO_ROOT = _EVAL_DIR.parents[0]        # repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import (
    CLIP_DURATION,
    DEFAULT_DATA_DIRS,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from toy_evaluation.model_utils import ToyModelWrapper

# =============================================================================
#  Constants
# =============================================================================

INTERP_STEPS = 11  # α ∈ {0.0, 0.1, …, 1.0}
PATH_INTEGRATION_STEPS = 101  # fine α-grid for path-length integration


# =============================================================================
#  Interpolation
# =============================================================================

def lerp(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * z_a + alpha * z_b


def slerp(z_a: torch.Tensor, z_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation with (linearly interpolated) magnitude."""
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
    """Trigonometric (constant-power) interpolation."""
    theta = alpha * (np.pi / 2.0)
    return np.cos(theta) * z_a + np.sin(theta) * z_b


# =============================================================================
#  Latent-space metrics
# =============================================================================

def latent_mse(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Element-wise MSE between two latent tensors."""
    return float(F.mse_loss(z_a, z_b).item())


# =============================================================================
#  Single-checkpoint core (used by evaluate_all.py and the CLI)
# =============================================================================

def evaluate_latent_distance_checkpoint(
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
) -> Dict:
    """Run the encoder-homomorphism test for ONE checkpoint.

    Parameters
    ----------
    stems : list[(pt_path, stem_name, track_id)] — validation stems.
    pairs : optional pre-generated pair list — pass the SAME pair list when
        comparing multiple checkpoints so every model sees identical pairs.

    Returns the aggregated summary dict (per-α mean/std MSE for
    Lerp / Slerp / Trig, each against the matching audio-domain mix, plus
    trigMix norm-preservation and interpolation path lengths).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_samples = int(clip_duration * sample_rate)
    alphas = np.linspace(0.0, 1.0, interp_steps).tolist()
    path_alphas = np.linspace(0.0, 1.0, PATH_INTEGRATION_STEPS)

    if model_type is None:
        from toy_evaluation.model_utils import detect_model_type
        model_type = detect_model_type(ckpt_path, device)
        print(f"  [Auto-detect] Model type: {model_type}")

    model = ToyModelWrapper(ckpt_path, device, model_type)

    if pairs is None:
        pairs = generate_pairs(stems, num_pairs, seed=seed, mode=pair_mode)
    print(f"  {model.name} ({model_type}): {len(pairs)} pairs")

    # ---- Accumulators ----
    def _make_accum():
        return {
            "lerp_mse": np.full((len(pairs), interp_steps), np.nan),
            "slerp_mse": np.full((len(pairs), interp_steps), np.nan),
            "trig_mse": np.full((len(pairs), interp_steps), np.nan),
            # Signed norm-preservation error of the trigMix path vs ‖Z_A‖:
            #   ‖trigMix(Z_A, Z_B, α)‖ − ‖Z_A‖   (ideal 0 → constant-power)
            "norm_pres": np.full((len(pairs), interp_steps), np.nan),
        }

    acc = _make_accum()
    # Scalar per-pair path lengths (chord sums over the fine α-grid).
    path_length_lerp = np.full(len(pairs), np.nan)
    path_length_trig = np.full(len(pairs), np.nan)

    for pair_idx, (i, j) in enumerate(tqdm(pairs, desc="Pairs", unit="pair")):
        path_a, stem_a, track_a = stems[i]
        path_b, stem_b, track_b = stems[j]

        try:
            audio_a = stem_to_numpy(path_a, clip_samples, centre=True)
            audio_b = stem_to_numpy(path_b, clip_samples, centre=True)
        except Exception as exc:
            print(f"  ⚠ Skip pair {pair_idx}: load error – {exc}")
            continue

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

        for a_idx, alpha in enumerate(alphas):
            # ---- Audio-domain mixes (matched to each scheme's weighting law) ----
            audio_mix_lerp = (1.0 - alpha) * audio_a + alpha * audio_b
            audio_mix_trig = trig_mix(
                torch.from_numpy(audio_a), torch.from_numpy(audio_b), alpha
            ).numpy()

            try:
                z_mix_lerp = model.encode(audio_mix_lerp.astype(np.float32))
                z_mix_trig = model.encode(audio_mix_trig.astype(np.float32))
            except Exception:
                continue

            mlen = min(z_mix_lerp.shape[-1], z_a.shape[-1])
            z_a_u = z_a[..., :mlen]
            z_b_u = z_b[..., :mlen]
            z_ml_u = z_mix_lerp[..., :mlen]
            z_mt_u = z_mix_trig[..., :mlen]

            if a_idx == 0:
                # ---- Path length (fine-grid chord integral, once per pair) ----
                z_a_p = z_a_u.reshape(-1)
                z_b_p = z_b_u.reshape(-1)
                pl_lerp = 0.0
                pl_trig = 0.0
                z_prev_lerp = lerp(z_a_p, z_b_p, 0.0)
                z_prev_trig = trig_mix(z_a_p, z_b_p, 0.0)
                for alpha in path_alphas[1:]:
                    z_cur_lerp = lerp(z_a_p, z_b_p, alpha)
                    z_cur_trig = trig_mix(z_a_p, z_b_p, alpha)
                    pl_lerp += float(torch.norm(z_cur_lerp - z_prev_lerp).item())
                    pl_trig += float(torch.norm(z_cur_trig - z_prev_trig).item())
                    z_prev_lerp, z_prev_trig = z_cur_lerp, z_cur_trig
                path_length_lerp[pair_idx] = pl_lerp
                path_length_trig[pair_idx] = pl_trig

            # ---- Lerp ↔ lerp audio mix ----
            z_lerp = lerp(z_a_u, z_b_u, alpha)
            acc["lerp_mse"][pair_idx, a_idx] = latent_mse(z_lerp, z_ml_u)

            # ---- Slerp ↔ lerp audio mix (slerp lerps magnitudes) ----
            z_slerp = slerp(z_a_u, z_b_u, alpha)
            acc["slerp_mse"][pair_idx, a_idx] = latent_mse(z_slerp, z_ml_u)

            # ---- Trig ↔ trig (constant-power) audio mix ----
            z_trig = trig_mix(z_a_u, z_b_u, alpha)
            acc["trig_mse"][pair_idx, a_idx] = latent_mse(z_trig, z_mt_u)
            # Norm preservation of the trigMix path relative to ‖Z_A‖.
            acc["norm_pres"][pair_idx, a_idx] = float(
                (torch.norm(z_trig) - torch.norm(z_a_u)).item()
            )

    # ---- Aggregate ----
    def _drop_nan(*arrays):
        mask = ~np.isnan(arrays[0]).any(axis=1)
        for a in arrays[1:]:
            mask &= ~np.isnan(a).any(axis=1)
        return mask, [a[mask] for a in arrays]

    acc_keys = list(acc.keys())
    row_mask, valid = _drop_nan(*[acc[k] for k in acc_keys])
    valid_map = {k: v for k, v in zip(acc_keys, valid)}
    n_valid = valid[0].shape[0]

    # Scalar per-pair arrays (path lengths), filtered to the same valid pairs.
    pl_mask = row_mask & ~np.isnan(path_length_lerp) & ~np.isnan(path_length_trig)
    path_lerp_valid = path_length_lerp[pl_mask]
    path_trig_valid = path_length_trig[pl_mask]

    def _mean_std(key: str) -> Tuple[List[float], List[float]]:
        arr = valid_map[key]
        return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

    _nan = float("nan")
    return {
        "label": model.name,
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "valid_pairs": n_valid,
        "total_pairs": len(pairs),
        "alphas": alphas,
        "mean_lerp_mse": _mean_std("lerp_mse")[0],
        "std_lerp_mse": _mean_std("lerp_mse")[1],
        "mean_slerp_mse": _mean_std("slerp_mse")[0],
        "std_slerp_mse": _mean_std("slerp_mse")[1],
        "mean_trig_mse": _mean_std("trig_mse")[0],
        "std_trig_mse": _mean_std("trig_mse")[1],
        # ---- Norm preservation of the trigMix path (ideal 0 at every α) ----
        "mean_norm_preservation": _mean_std("norm_pres")[0],     # signed
        "std_norm_preservation": _mean_std("norm_pres")[1],
        "mean_abs_norm_preservation": np.abs(valid_map["norm_pres"]).mean(axis=0).tolist(),
        # ---- Interpolation path lengths (chord sums over fine α-grid) ----
        "mean_path_length_lerp": (float(np.mean(path_lerp_valid))
                                  if path_lerp_valid.size else _nan),
        "std_path_length_lerp": (float(np.std(path_lerp_valid))
                                 if path_lerp_valid.size else _nan),
        "mean_path_length_trig": (float(np.mean(path_trig_valid))
                                  if path_trig_valid.size else _nan),
        "std_path_length_trig": (float(np.std(path_trig_valid))
                                 if path_trig_valid.size else _nan),
        "mean_path_length_ratio_trig_over_lerp": (
            float(np.mean(path_trig_valid / (path_lerp_valid + 1e-12)))
            if path_lerp_valid.size else _nan),
    }


# =============================================================================
#  Legacy two-model pipeline (kept for CLI backward compatibility)
# =============================================================================

def run_analysis(
    ckpt_baseline: str,
    ckpt_phase: str,
    data_dirs=None,
    cache_dir: Optional[str] = None,
    device: torch.device | None = None,
    num_pairs: int = 100,
    seed: int = 42,
    pair_mode: str = "random",
    model_type: str | None = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> None:
    """Two-checkpoint comparison wrapper around
    ``evaluate_latent_distance_checkpoint``."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/5] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(data_dirs, cache_dir, sample_rate=sample_rate)
    print(f"  Found {len(stems)} validation stems")
    pairs = generate_pairs(stems, num_pairs, seed=seed, mode=pair_mode)
    print(f"  Generated {len(pairs)} pairs (mode={pair_mode})")

    print("[2/5] Running latent-distance analysis (baseline) …")
    agg_base = evaluate_latent_distance_checkpoint(
        ckpt_baseline, stems, device,
        num_pairs=num_pairs, seed=seed, pair_mode=pair_mode,
        model_type=model_type, sample_rate=sample_rate,
        clip_duration=CLIP_DURATION, interp_steps=INTERP_STEPS,
        pairs=pairs,
    )
    print("[3/5] Running latent-distance analysis (phase/compare) …")
    agg_phase = evaluate_latent_distance_checkpoint(
        ckpt_phase, stems, device,
        num_pairs=num_pairs, seed=seed, pair_mode=pair_mode,
        model_type=model_type, sample_rate=sample_rate,
        clip_duration=CLIP_DURATION, interp_steps=INTERP_STEPS,
        pairs=pairs,
    )

    print("[4/5] Saving summary …")
    summary = {
        "description": "Toy AE latent distance — encoder topological homomorphism",
        "timestamp": datetime.now().isoformat(),
        "clip_duration_s": CLIP_DURATION,
        "interp_steps": INTERP_STEPS,
        "pair_mode": pair_mode,
        "data_dirs": list(data_dirs) if data_dirs is not None else DEFAULT_DATA_DIRS,
        "baseline": agg_base,
        "phase": agg_phase,
    }
    json_path = _EVAL_DIR / "latent_distance_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  📄 JSON → {json_path}")

    print("[5/5] Plotting MSE comparison …")
    _plot_mse_comparison(agg_base, agg_phase, _EVAL_DIR / "latent_distance_plot.png")
    print("✓ Latent distance analysis complete.")


# =============================================================================
#  Plotting
# =============================================================================

def _plot_mse_comparison(base: Dict, phase: Dict, out_path: Path) -> None:
    alphas = np.array(base["alphas"])
    fig, ax = plt.subplots(figsize=(10, 6))
    interp_styles = {
        "lerp":  ("Lerp",  "o-"),
        "slerp": ("Slerp", "s--"),
        "trig":  ("Trig",  "^-."),
    }
    for agg, label, color in [
        (base, "Baseline", "#e74c3c"),
        (phase, "Phase", "#2980b9"),
    ]:
        for key, (disp_name, style) in interp_styles.items():
            mean = np.array(agg[f"mean_{key}_mse"])
            std = np.array(agg[f"std_{key}_mse"])
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.10, color=color)
            ax.plot(alphas, mean, style, color=color, linewidth=2, markersize=5,
                    label=f"{label} {disp_name}")

    ax.set_xlabel("Interpolation α", fontsize=13)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=13)
    ax.set_title("Toy AE — Encoder Topological Homomorphism\n"
                 "MSE: Z_interp vs Z_encoded_mix", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=alphas[len(alphas) // 2], color="gray", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Plot → {out_path}")


# =============================================================================
#  CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy AE Latent Distance — Encoder Topological Homomorphism",
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
                   help="Preprocessed cache base dir.")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["conv", "transformer"],
                   help="Model architecture. Auto-detected from checkpoint if omitted.")
    p.add_argument("--num_pairs", type=int, default=100,
                   help="Number of stem pairs (default: 100).")
    p.add_argument("--pair_mode", default="random",
                   choices=["random", "intra", "inter"],
                   help="Pair mode (default: random).")
    p.add_argument("--device", default="cuda",
                   help="Device (default: cuda).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=MODEL_SAMPLE_RATE,
                   help=f"Sample rate for audio loading (default: {MODEL_SAMPLE_RATE}).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    run_analysis(
        ckpt_baseline=args.ckpt_baseline,
        ckpt_phase=args.ckpt_phase,
        data_dirs=args.data_dirs,
        cache_dir=args.cache_dir,
        device=device,
        num_pairs=args.num_pairs,
        seed=args.seed,
        pair_mode=args.pair_mode,
        model_type=args.model_type,
        sample_rate=args.sample_rate,
    )
