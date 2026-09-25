#!/usr/bin/env python3
"""
Merge the extended (2000-clip) reconstruction results into the unified data JSON.

Overwrites each model's ``recon_msstft`` entry with the 2000-clip mean/std/sem,
and records the sample size. Run AFTER toy_evaluation/unified_ablation/recon_2000/run_all.sh.

Usage:
    .venv/bin/python toy_evaluation/unified_ablation/merge_recon_2000.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DATA_PATH = _HERE / "unified_data_from_artifacts.json"
RECON_DIR = _HERE / "recon_2000"

MODELS = [
    "transformer_minimal_full_best",
    "transformer_softMax_full_best",
    "transformer_norm_full_best",
    "transformer_best_full_best",
    "transformer_implicit_full_v2_best",
    "transformer_mixup_enc_best",
]


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    combined = {"description": "Extended recon eval, 2000 clips, seed 42", "models": {}}
    for stem in MODELS:
        path = RECON_DIR / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path} — run recon_2000/run_all.sh first")
        r = json.loads(path.read_text())
        n = int(r["num_clips_valid"])
        mean, std = float(r["mean_msstft"]), float(r["std_msstft"])
        sem = std / np.sqrt(n) if n > 0 else float("nan")
        cell = data["models"][stem]["recon_msstft"]
        old = dict(cell)
        cell.update({"mean": mean, "std": std, "sem": sem, "n_clips": n,
                     "old_200clip": {"mean": old.get("mean"), "std": old.get("std"),
                                     "n_clips": old.get("n_clips")}})
        combined["models"][stem] = {
            "mean_msstft": mean, "std_msstft": std, "sem": sem, "n": n,
            "per_clip_msstft": r.get("per_clip_msstft", []),
        }
        print(f"{stem:36s} n={n:4d}  mean {mean:.4f} (old {old.get('mean'):.4f})  "
              f"std {std:.4f}  sem {sem:.4f}")

    DATA_PATH.write_text(json.dumps(data, indent=2))
    (RECON_DIR / "recon_2000_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"\nupdated {DATA_PATH}")
    print(f"wrote {RECON_DIR / 'recon_2000_summary.json'}")


if __name__ == "__main__":
    main()
