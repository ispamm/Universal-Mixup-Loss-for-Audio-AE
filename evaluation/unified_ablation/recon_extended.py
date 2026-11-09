#!/usr/bin/env python3
"""
Extended reconstruction evaluation (MS-STFT, per-clip) for ONE checkpoint.

Reuses evaluation/eval_reconstruction.py::evaluate_reconstruction_checkpoint
(identical metric code path as the battery) over a larger, deterministic clip
set (default 2000 validation clips, seed 42). The first 200 sampled clips are a
strict subset of the original battery's 200 recon clips (same Random(42)
stream), so the new mean is a superset extension, not a re-roll.

Writes a JSON with the summary + per-clip msstft/LSD for paired analyses.

Usage:
    .venv/bin/python evaluation/unified_ablation/recon_extended.py \
        --ckpt training/checkpoints/transformer_norm_full_best.pt \
        --out evaluation/unified_ablation/recon_2000/transformer_norm_full_best.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.data import (  # noqa: E402
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
)
from evaluation.eval_reconstruction import (  # noqa: E402
    evaluate_reconstruction_checkpoint,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num_clips", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--include_lsd", action="store_true", default=True,
                    help="keep True: the core only stores per_clip records when LSD is on")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    stems = collect_validation_stems()
    # identical sampling stream as evaluate_all.run_battery (recon drawn first)
    rng = random.Random(args.seed)
    n_avail = len(stems)
    clip_indices = rng.sample(range(n_avail), min(args.num_clips, n_avail))

    print(f"[{Path(args.ckpt).stem}] threads={args.threads} clips={len(clip_indices)} "
          f"seed={args.seed} device={device}", flush=True)

    res = evaluate_reconstruction_checkpoint(
        args.ckpt, stems, device,
        num_clips=len(clip_indices), seed=args.seed,
        clip_indices=clip_indices, clip_duration=CLIP_DURATION,
        sample_rate=MODEL_SAMPLE_RATE, include_lsd=args.include_lsd,
    )
    print(f"[{Path(args.ckpt).stem}] valid={res['num_clips_valid']} "
          f"mean_msstft={res['mean_msstft']:.4f} std={res['std_msstft']:.4f}", flush=True)

    out = {
        "description": "Extended reconstruction eval, MS-STFT per clip",
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "num_clips_requested": len(clip_indices),
        "model": Path(args.ckpt).stem,
        "mean_msstft": res["mean_msstft"],
        "std_msstft": res["std_msstft"],
        "num_clips_valid": res["num_clips_valid"],
        # per-clip msstft only (drop the bulky LSD fields we did not need)
        "per_clip_msstft": [c["msstft"] for c in res.get("per_clip", [])],
        "per_clip_stem": [c["stem"] for c in res.get("per_clip", [])],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
