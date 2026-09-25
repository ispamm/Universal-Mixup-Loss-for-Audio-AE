#!/usr/bin/env python3
"""
Supplemental per-pair norm-preservation evaluation (Norm Err column std).

Replicates the norm-preservation part of the latent_distance protocol
(toy_evaluation/latent_distance_toy.py::evaluate_latent_distance_checkpoint):

    norm_pres(pair, α) = ‖trigMix(Z_A, Z_B, α)‖ − ‖Z_A‖,  Z = E(clip)

and stores the per-pair absolute values at α = 0.5 (plus all-α aggregates to
cross-validate the mean against the saved battery summaries).

Same data convention: 1000 random validation pairs, seed 42, 2 s centre-crop
@ 16 kHz from the preprocessed cache.

Usage:
    .venv/bin/python toy_evaluation/unified_ablation/norm_pres_supplement.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_EVAL = _THIS.parents[1]
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from toy_evaluation.data import (  # noqa: E402
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from toy_evaluation.lims_test_toy import trig_mix  # noqa: E402
from toy_evaluation.model_utils import ToyModelWrapper  # noqa: E402

CKPTS = [
    "toy_training/checkpoints/transformer_minimal_full_best.pt",
    "toy_training/checkpoints/transformer_softMax_full_best.pt",
    "toy_training/checkpoints/transformer_norm_full_best.pt",
    "toy_training/checkpoints/transformer_best_full_best.pt",
    "toy_training/checkpoints/transformer_implicit_full_v2_best.pt",
    "toy_training/checkpoints/transformer_mixup_enc_best.pt",
]

NUM_PAIRS = 1000
SEED = 42
ALPHAS = np.linspace(0.0, 1.0, 11)


def main() -> None:
    torch.set_num_threads(max(1, min(24, torch.get_num_threads())))
    device = torch.device("cpu")
    print(f"threads={torch.get_num_threads()} pairs={NUM_PAIRS} seed={SEED} device={device}")

    stems = collect_validation_stems()
    pairs = generate_pairs(stems, NUM_PAIRS, seed=SEED, mode="random")
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)
    print(f"stems={len(stems)} pairs={len(pairs)}")

    out: dict = {
        "description": "Per-pair |norm-preservation error| at α (trigMix), seed 42",
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "seed": SEED,
        "alphas": ALPHAS.tolist(),
        "models": {},
    }

    for ckpt in CKPTS:
        stem = Path(ckpt).stem
        print(f"\n=== {stem} ===")
        model = ToyModelWrapper(ckpt, device)
        norm_pres = np.full((len(pairs), len(ALPHAS)), np.nan)
        for p_idx, (i, j) in enumerate(pairs):
            a = stem_to_numpy(stems[i][0], clip_samples, centre=True)
            b = stem_to_numpy(stems[j][0], clip_samples, centre=True)
            z_a = model.encode(a)
            z_b = model.encode(b)
            if z_a.shape[-1] != z_b.shape[-1]:
                mlen = min(z_a.shape[-1], z_b.shape[-1])
                z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]
            za_norm = float(torch.norm(z_a).item())
            for a_idx, alpha in enumerate(ALPHAS):
                z_trig = trig_mix(z_a, z_b, float(alpha))
                norm_pres[p_idx, a_idx] = float(torch.norm(z_trig).item()) - za_norm
            if (p_idx + 1) % 200 == 0:
                print(f"    pair {p_idx + 1}/{len(pairs)}")

        valid = norm_pres[~np.isnan(norm_pres).any(axis=1)]
        abs_arr = np.abs(valid)
        mid = int(np.argmin(np.abs(ALPHAS - 0.5)))
        out["models"][stem] = {
            "valid_pairs": int(valid.shape[0]),
            "mean_abs_per_alpha": abs_arr.mean(axis=0).tolist(),
            "std_abs_per_alpha": abs_arr.std(axis=0).tolist(),
            "mean_signed_per_alpha": valid.mean(axis=0).tolist(),
            "std_signed_per_alpha": valid.std(axis=0).tolist(),
            "norm_err_05": {
                "mean": float(abs_arr[:, mid].mean()),
                "std": float(abs_arr[:, mid].std()),
                "n": int(abs_arr.shape[0]),
            },
            # per-pair absolute error at α=0.5 (kept for bootstrap work)
            "per_pair_abs_05": abs_arr[:, mid].tolist(),
        }
        r = out["models"][stem]["norm_err_05"]
        print(f"    α=0.5 |Δnorm| mean={r['mean']:.4f} std={r['std']:.4f} (n={r['n']})")

    json_path = _THIS.parent / "norm_pres_supplement.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {json_path}")


if __name__ == "__main__":
    main()
