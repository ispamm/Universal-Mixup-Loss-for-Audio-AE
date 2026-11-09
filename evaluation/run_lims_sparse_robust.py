#!/usr/bin/env python3
"""Re-run the LIMS decoder-additivity protocol with sparse-audio-robust LSD.

Runs ``evaluate_lims_checkpoint`` for the four checkpoints on the same
100-pair / seed-42 pair list used by the eval_all battery (CPU), and dumps the
per-checkpoint aggregates — including the new ``mean_trig_lsd_active`` /
``mean_trig_ref_sparsity`` keys — to a JSON for the analysis docs.

Usage:
    python evaluation/run_lims_sparse_robust.py [--num_pairs 100] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import collect_validation_stems, generate_pairs
from evaluation.lims_test import evaluate_lims_checkpoint
from training.losses import SAMELoss

CKPTS = [
    "training/checkpoints/transformer_best_full_best.pt",
    "training/checkpoints/transformer_norm_full_best.pt",
    "training/checkpoints/transformer_softMax_full_best.pt",
    "training/checkpoints/transformer_minimal_full_best.pt",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--json", type=Path, default=_EVAL_DIR / "lims_sparse_robust_summary.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stems = collect_validation_stems()
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    print(f"stems={len(stems)}  pairs={len(pairs)}  device={device}")

    results = {}
    msstft_loss_fn = SAMELoss().to(device)  # training-matched MS-STFT cross-check
    for ckpt in CKPTS:
        print(f"\n=== {Path(ckpt).stem} ===")
        res = evaluate_lims_checkpoint(
            ckpt, stems, device,
            num_pairs=len(pairs), seed=args.seed, pair_mode="random",
            pairs=pairs,
            msstft_loss_fn=msstft_loss_fn,
        )
        results[Path(ckpt).stem] = res
        print(f"  dec_trig_lsd@0.5        = {res['mean_trig_lsd'][5]:.3f}")
        print(f"  dec_trig_lsd_active@0.5 = {res['mean_trig_lsd_active'][5]:.3f}")
        print(f"  trig_ref_sparsity@0.5   = {res['mean_trig_ref_sparsity'][5]:.3f}")
        print(f"  trig_msstft@0.5         = {res['mean_trig_msstft'][5]:.4f}")

    out = {
        "description": "LIMS decoder additivity — sparse-audio-robust LSD",
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "seed": args.seed,
        "results": results,
    }
    args.json.write_text(json.dumps(out, indent=2))
    print(f"\n📄 JSON -> {args.json}")


if __name__ == "__main__":
    main()
