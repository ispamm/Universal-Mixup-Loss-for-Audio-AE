#!/usr/bin/env python3
"""
Quick per-trial report for the encoder-cone anti-collapse search
=================================================================

For ONE checkpoint, prints a compact battery (200 samples, seed 42) used to
rank training strategies:

  * recon MS-STFT (200 clips)                        — reconstruction quality
  * encoder homomorphism MSE trig/lerp @0.5 + norm    — does it keep the
    error @0.5 (200 pairs)                              linearity win?
  * latent diffusion-readiness (200 clips):
      mean pairwise |cosine|   (0 = decorrelated, healthy; ~0.8 = collapse)
      participation ratio / top-1 variance
      E||z||^2/P and per-channel variance range (target ~1)

Usage:
    python toy_evaluation/quick_trial_report.py --ckpt <path> [--device cuda:0]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

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
from toy_evaluation.eval_reconstruction import evaluate_reconstruction_checkpoint
from toy_evaluation.latent_distance_toy import evaluate_latent_distance_checkpoint
from toy_evaluation.model_utils import ToyModelWrapper

N_CLIPS = 200
N_PAIRS = 200
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stems = collect_validation_stems()
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print("== checkpoint:", Path(args.ckpt).stem)
    print("   epoch:", ck.get("epoch"),
          "| best val:", ck.get("scheduler_G_state_dict", {}).get("best"),
          "| config:", ck.get("model_config"))

    # ---- 1. reconstruction (200 clips) ----
    rng = random.Random(SEED)
    recon_idx = rng.sample(range(len(stems)), N_CLIPS)
    rec = evaluate_reconstruction_checkpoint(
        args.ckpt, stems, device, num_clips=N_CLIPS, seed=SEED,
        clip_indices=recon_idx, clip_duration=CLIP_DURATION,
        sample_rate=MODEL_SAMPLE_RATE,
    )
    print(f"   recon MS-STFT = {rec['mean_msstft']:.4f}  (valid {rec['num_clips_valid']})")

    # ---- 2. encoder homomorphism + norm error (200 pairs) ----
    ld = evaluate_latent_distance_checkpoint(
        args.ckpt, stems, device, num_pairs=N_PAIRS, seed=SEED,
        clip_duration=CLIP_DURATION, sample_rate=MODEL_SAMPLE_RATE,
    )
    alphas = np.asarray(ld["alphas"])

    def vat(curve, a=0.5):
        arr = np.asarray(curve)
        n = len(arr)
        return float(arr[max(0, min(n - 1, int(round(a * (n - 1)))))])

    print(f"   enc trig MSE@0.5 = {vat(ld['mean_trig_mse']):.4f} | "
          f"lerp MSE@0.5 = {vat(ld['mean_lerp_mse']):.4f} | "
          f"norm err@0.5 = {vat(ld['mean_abs_norm_preservation']):.2f} "
          f"(valid {ld['valid_pairs']} pairs)")

    # ---- 3. latent diffusion-readiness (200 clips) ----
    model = ToyModelWrapper(args.ckpt, device)
    Z = np.stack([
        model.encode(stem_to_numpy(stems[i][0], clip_samples, centre=True)).squeeze(0).cpu().numpy()
        for i in recon_idx
    ])
    P = Z.shape[1] * Z.shape[2]
    flat = Z.reshape(N_CLIPS, -1)
    energy = (flat ** 2).sum(1) / P
    chvar = Z.var(axis=(0, 2))
    prng = np.random.RandomState(SEED)
    cos_vals = []
    for _ in range(100):
        i, j = prng.randint(0, N_CLIPS, 2)
        while i == j:
            j = prng.randint(0, N_CLIPS)
        a, b = flat[i], flat[j]
        cos_vals.append(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    fc = flat - flat.mean(0)
    G = fc @ fc.T / N_CLIPS
    ev = np.linalg.eigvalsh(G)[::-1]
    ev = ev[ev > 1e-12]
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    top1 = ev[0] / ev.sum()
    print(f"   latent: mean|cosine| = {np.mean(cos_vals):.4f} | participation ratio = {pr:.1f} | "
          f"top1-var = {top1:.4f}")
    print(f"   E||z||^2/P = {energy.mean():.4f} (std {energy.std():.4f}) | "
          f"per-channel var: mean {chvar.mean():.4f}, min {chvar.min():.4f}, max {chvar.max():.4f}")


if __name__ == "__main__":
    main()
