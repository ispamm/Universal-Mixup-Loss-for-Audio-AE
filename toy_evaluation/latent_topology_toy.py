#!/usr/bin/env python3
"""
Toy AE — Latent Topology Analysis (Global Manifold Structure)
==============================================================

Measures the **global geometric structure** of the latent manifold over the
entire held-out validation set, independent of any specific pair or
interpolation schedule.

Metrics
-------
1. **Latent Density** : Average pairwise L2 distance between all distinct
   real latent vectors in the pool.  Higher density → latent codes are more
   spread out, less collapsed.

2. **Latent Convexity** : For random pairs of real latents, interpolate at
   α = 0.5 (Lerp, Slerp, Trig) and measure the L2 distance from each
   interpolated midpoint to its **nearest neighbour** among the real latent
   pool (excluding the pair's own endpoints).  Lower convexity → the latent
   manifold is more "filled in" — interpolations stay close to real data.

3. **Latent Anisotropy** : Condition number (σ_max / σ_min) of the latent
   covariance matrix (centred, flattened latents).  Ideal 1.0 = perfectly
   isotropic latent space; large values reveal elongated / collapsed
   directions, i.e. how "directionally uniform" the manifold is.  Because the
   pool is rank-deficient (N clips ≪ flattened dims), the raw ratio is
   dominated by the noise floor — read it as a *relative* comparison across
   checkpoints, alongside the interpretable companions
   ``latent_effective_rank`` (participation ratio: effective dimensionality)
   and ``latent_top_var_fraction`` (variance share of the single largest
   direction; isotropic ≈ 0).

Both metrics are computed with ``torch.cdist`` on GPU for efficiency.

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``toy_evaluation.data``):
validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.

Outputs
-------
  toy_evaluation/latent_topology_summary.json

Usage
-----
    python toy_evaluation/latent_topology_toy.py \\
        --ckpt toy_training/checkpoints/my_model.pt \\
        --num_clips 256 \\
        --num_mixups 512
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
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent             # toy_evaluation/
_REPO_ROOT = _EVAL_DIR.parent             # repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    stem_to_numpy,
)
from toy_evaluation.model_utils import ToyModelWrapper

# =============================================================================
#  Constants
# =============================================================================

DEFAULT_NUM_CLIPS = 256
DEFAULT_NUM_MIXUPS = 512


# =============================================================================
#  Interpolation helpers
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
#  Single-checkpoint core (used by evaluate_all.py and the CLI)
# =============================================================================

def evaluate_topology_checkpoint(
    ckpt_path: str,
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_clips: int = DEFAULT_NUM_CLIPS,
    num_mixups: int = DEFAULT_NUM_MIXUPS,
    seed: int = 42,
    model_type: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    clip_duration: float = CLIP_DURATION,
    clip_indices: Optional[List[int]] = None,
) -> Dict:
    """Compute latent density and convexity for ONE checkpoint.

    Parameters
    ----------
    stems : list[(pt_path, stem_name, track_id)] — validation stems.
    clip_indices : optional pre-selected stem indices — pass the SAME index
        list when comparing multiple checkpoints so every model sees the same
        latent pool.

    Returns dict with ``latent_density``, ``latent_convexity``
    (lerp/slerp/trig), ``latent_anisotropy`` (σ_max/σ_min of the latent
    covariance) and pool metadata.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_samples = int(clip_duration * sample_rate)

    if model_type is None:
        from toy_evaluation.model_utils import detect_model_type
        model_type = detect_model_type(ckpt_path, device)
        print(f"  [Auto-detect] Model type: {model_type}")

    model = ToyModelWrapper(ckpt_path, device, model_type)
    print(f"  Model: {model.name}  ({model_type})")

    n_available = len(stems)
    if n_available < num_clips:
        print(f"  ⚠ Only {n_available} stems available (requested {num_clips}) — using all.")
        num_clips = n_available

    if clip_indices is None:
        rng = random.Random(seed)
        clip_indices = rng.sample(range(n_available), num_clips)

    # ---- Encode all clips → Z_real pool ----
    print(f"[1/4] Encoding {len(clip_indices)} clips …")
    Z_pool: List[torch.Tensor] = []
    for idx in tqdm(clip_indices, desc="Encoding", unit="clip"):
        path, _stem, _track = stems[idx]
        try:
            audio = stem_to_numpy(path, clip_samples, centre=True)
        except Exception:
            continue
        try:
            z = model.encode(audio)            # (1, D, T_i)
            Z_pool.append(z)
        except Exception:
            continue

    n_valid = len(Z_pool)
    print(f"  Successfully encoded {n_valid}/{len(clip_indices)} clips")

    if n_valid < 2:
        print("  ⚠ Not enough valid clips — aborting.")
        return {
            "checkpoint": ckpt_path,
            "model_type": model_type,
            "error": "not enough valid clips",
            "latent_density": float("nan"),
            "latent_convexity": {"lerp": float("nan"), "slerp": float("nan"),
                                 "trig": float("nan")},
            "latent_anisotropy": float("nan"),
            "latent_sigma_max": float("nan"),
            "latent_sigma_min": float("nan"),
            "latent_effective_rank": float("nan"),
            "latent_top_var_fraction": float("nan"),
        }

    # ---- Common temporal length + flatten ----
    T_min = min(z.shape[-1] for z in Z_pool)
    D_latent = Z_pool[0].shape[1]
    Z_stack = torch.stack([z[:, :, :T_min] for z in Z_pool], dim=0).squeeze(1).to(device)
    N = Z_stack.shape[0]
    D_total = D_latent * T_min
    Z_flat = Z_stack.reshape(N, D_total)
    print(f"  Latent pool: {tuple(Z_stack.shape)} → flattened {tuple(Z_flat.shape)}")

    # =========================================================================
    #  Metric 1: Latent Density
    # =========================================================================
    print("  Computing Latent Density …")
    pairwise_dists = torch.cdist(Z_flat, Z_flat, p=2)   # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    latent_density = pairwise_dists[mask].mean().item()
    latent_density_std = pairwise_dists[mask].std().item()
    print(f"    Density = {latent_density:.6f} ± {latent_density_std:.6f}")

    # =========================================================================
    #  Metric 2: Latent Anisotropy (isotropy of the latent manifold)
    # =========================================================================
    print("  Computing Latent Anisotropy (covariance singular-value ratio) …")
    Z_cent = Z_flat - Z_flat.mean(dim=0, keepdim=True)
    try:
        # σ_max / σ_min of the (centred) latent covariance — float64 SVD for
        # numerical stability; near-zero directions are excluded.  NOTE: with
        # N << D_total the spectrum is rank-deficient, so σ_min is dominated
        # by the noise floor and the ratio is most meaningful as a *relative*
        # comparison across checkpoints.  The companion metrics
        # (``latent_effective_rank``, ``latent_top_var_fraction``) give an
        # interpretable absolute reading of isotropy.
        sigma = torch.linalg.svdvals(Z_cent.double())
        sigma = sigma[sigma > 1e-12]
        if sigma.numel() >= 2:
            latent_sigma_max = float(sigma[0].item())
            latent_sigma_min = float(sigma[-1].item())
            latent_anisotropy = latent_sigma_max / latent_sigma_min
            # Participation ratio (effective dimensionality): (Σσ²)² / Σσ⁴.
            var = sigma ** 2
            latent_effective_rank = float((var.sum() ** 2 / (var ** 2).sum()).item())
            latent_top_var_fraction = float((var[0] / var.sum()).item())
        else:
            latent_sigma_max = latent_sigma_min = float("nan")
            latent_anisotropy = float("nan")
            latent_effective_rank = float("nan")
            latent_top_var_fraction = float("nan")
    except Exception:  # noqa: BLE001 — SVD failure must not kill the battery
        latent_sigma_max = latent_sigma_min = float("nan")
        latent_anisotropy = float("nan")
        latent_effective_rank = float("nan")
        latent_top_var_fraction = float("nan")
    print(f"    Anisotropy (σ_max/σ_min) = {latent_anisotropy:.6f} "
          f"[σ_max={latent_sigma_max:.3e}, σ_min={latent_sigma_min:.3e}, "
          f"eff_rank={latent_effective_rank:.2f}, "
          f"top1_var={latent_top_var_fraction:.4f}]")

    # =========================================================================
    #  Metric 2: Latent Convexity
    # =========================================================================
    print(f"  Computing Latent Convexity ({num_mixups} mixups) …")
    n_pairs = min(num_mixups, N * (N - 1) // 2)
    rng = random.Random(seed)
    pair_indices: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = n_pairs * 200
    while len(pair_indices) < n_pairs and attempts < max_attempts:
        i, j = rng.sample(range(N), 2)
        if (i, j) not in pair_indices and (j, i) not in pair_indices:
            pair_indices.append((i, j))
        attempts += 1

    pair_i = torch.tensor([p[0] for p in pair_indices], device=device)  # (M,)
    pair_j = torch.tensor([p[1] for p in pair_indices], device=device)  # (M,)
    Z_i = Z_flat[pair_i]
    Z_j = Z_flat[pair_j]

    def _convexity(Z_mid: torch.Tensor) -> float:
        return _compute_convexity_batched(Z_mid, Z_flat, pair_i, pair_j, device)

    lerp_convexity = _convexity(0.5 * Z_i + 0.5 * Z_j)
    slerp_convexity = _convexity(torch.stack([
        slerp(Z_stack[i:i+1], Z_stack[j:j+1], 0.5).reshape(-1)
        for i, j in zip(pair_i.tolist(), pair_j.tolist())
    ], dim=0).to(device))
    trig_convexity = _convexity(trig_mix(Z_i, Z_j, 0.5))

    print(f"    Lerp  Convexity = {lerp_convexity:.6f}")
    print(f"    Slerp Convexity = {slerp_convexity:.6f}")
    print(f"    Trig  Convexity = {trig_convexity:.6f}")

    return {
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "num_clips_requested": len(clip_indices),
        "num_clips_valid": n_valid,
        "num_mixups": len(pair_indices),
        "latent_dim": D_latent,
        "temporal_frames": T_min,
        "flattened_dim": D_total,
        "latent_density": latent_density,
        "latent_density_std": latent_density_std,
        "latent_anisotropy": latent_anisotropy,
        "latent_sigma_max": latent_sigma_max,
        "latent_sigma_min": latent_sigma_min,
        "latent_effective_rank": latent_effective_rank,
        "latent_top_var_fraction": latent_top_var_fraction,
        "latent_convexity": {
            "lerp": lerp_convexity,
            "slerp": slerp_convexity,
            "trig": trig_convexity,
        },
    }


def _compute_convexity_batched(
    Z_mid: torch.Tensor,       # (M, D_total)
    Z_pool: torch.Tensor,      # (N, D_total)
    pair_i: torch.Tensor,      # (M,)
    pair_j: torch.Tensor,      # (M,)
    device: torch.device,
    chunk_size: int = 64,
) -> float:
    """Mean nearest-neighbour distance of midpoints to real latents,
    excluding each pair's own endpoints.  Chunked ``torch.cdist``."""
    M = Z_mid.shape[0]
    N = Z_pool.shape[0]
    if M == 0:
        return float("nan")

    all_min_dists: List[float] = []
    for chunk_start in range(0, M, chunk_size):
        chunk_end = min(chunk_start + chunk_size, M)
        Z_chunk = Z_mid[chunk_start:chunk_end]
        dists_chunk = torch.cdist(Z_chunk, Z_pool, p=2)  # (C, N)
        for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
            dists_chunk[local_idx, pair_i[global_idx]] = float("inf")
            dists_chunk[local_idx, pair_j[global_idx]] = float("inf")
        min_dists = dists_chunk.min(dim=1).values
        all_min_dists.extend(min_dists.cpu().tolist())

    return float(np.mean(all_min_dists))


# =============================================================================
#  CLI wrapper
# =============================================================================

def run_topology_analysis(
    ckpt_path: str,
    data_dirs=None,
    cache_dir: Optional[str] = None,
    device: torch.device | None = None,
    num_clips: int = DEFAULT_NUM_CLIPS,
    num_mixups: int = DEFAULT_NUM_MIXUPS,
    seed: int = 42,
    model_type: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    output_json: Optional[Path] = None,
) -> Dict:
    """Collect validation stems and run ``evaluate_topology_checkpoint``,
    writing the standard summary JSON."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[0/3] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(data_dirs, cache_dir, sample_rate=sample_rate)
    print(f"  Found {len(stems)} validation stems")

    summary = evaluate_topology_checkpoint(
        ckpt_path, stems, device,
        num_clips=num_clips, num_mixups=num_mixups, seed=seed,
        model_type=model_type, sample_rate=sample_rate,
        clip_duration=CLIP_DURATION,
    )
    summary.update({
        "description": "Toy AE latent topology — global manifold structure "
                       "(Latent Density + Latent Convexity)",
        "timestamp": datetime.now().isoformat(),
    })

    json_path = output_json or (_EVAL_DIR / "latent_topology_summary.json")
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  📄 JSON → {json_path}")
    print("✓ Latent topology analysis complete.")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy AE Latent Topology — Global Manifold Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True,
                   help="Path to a toy autoencoder checkpoint (.pt).")
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                   nargs="+", default=None,
                   help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                        "training set.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Preprocessed cache base dir.")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["conv", "transformer"],
                   help="Model architecture. Auto-detected if omitted.")
    p.add_argument("--num_clips", type=int, default=DEFAULT_NUM_CLIPS,
                   help=f"Number of random clips to encode (default: {DEFAULT_NUM_CLIPS}).")
    p.add_argument("--num_mixups", type=int, default=DEFAULT_NUM_MIXUPS,
                   help=f"Number of random interpolations for convexity "
                        f"(default: {DEFAULT_NUM_MIXUPS}).")
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
    run_topology_analysis(
        ckpt_path=args.ckpt,
        data_dirs=args.data_dirs,
        cache_dir=args.cache_dir,
        device=device,
        num_clips=args.num_clips,
        num_mixups=args.num_mixups,
        seed=args.seed,
        model_type=args.model_type,
        sample_rate=args.sample_rate,
    )
