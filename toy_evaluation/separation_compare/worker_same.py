#!/usr/bin/env python3
"""
SAME toy-model worker for the algebraic separation study.

Runs in the project venv (torch 2.7.1+cu126) on cuda:0 by default.  Loads the
16 kHz chunks produced by ``prepare_chunks.py``, and for every (track, stem):

  * reconstruction baseline : recon = decode(encode(stem))           -> recon_*
  * algebraic separation    : z_sep = z_mix - sum(z_other_stems),
                              sep   = decode(z_sep)                  -> sep_*

with SI-SDR and MS-STFT MSE of the estimate against the true stem waveform.
Writes one JSON with the same schema as ``worker_cae.py``.

Usage (venv):
    python toy_evaluation/separation_compare/worker_same.py \
        --chunks toy_evaluation/separation_compare/chunks \
        --out    toy_evaluation/separation_compare/results_same.json \
        --ckpt   toy_training/checkpoints/transformer_best_full_best.pt \
        --sr     16 \
        --device cuda:0

    # 44.1 kHz toy checkpoint (e.g. transformer_best_full_441_best.pt):
    python toy_evaluation/separation_compare/worker_same.py \
        --chunks toy_evaluation/separation_compare/chunks \
        --out    toy_evaluation/separation_compare/results_same_441.json \
        --ckpt   toy_training/checkpoints/transformer_best_full_441_best.pt \
        --sr     44 \
        --device cuda:0

``--sr`` selects which chunk resolution the worker uses (16 = ``*16.npy``
16 kHz chunks, 44 = ``*44.npy`` 44.1 kHz chunks — pick the model's native
sample rate).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.model_utils import ToyModelWrapper  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path, default=_THIS_FILE.parent / "results_same.json")
    ap.add_argument("--ckpt", type=str,
                    default="toy_training/checkpoints/transformer_best_full_best.pt")
    ap.add_argument("--sr", type=int, default=16, choices=[16, 44],
                    help="Chunk sample rate: 16 (16 kHz) or 44 (44.1 kHz). "
                         "Must match the checkpoint's native sample rate.")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    sr = manifest[f"sr{args.sr}"]
    suffix = f"{args.sr}"
    print(f"SAME worker | sr={sr} device={device} "
          f"tracks={len(manifest['tracks'])}")

    model = ToyModelWrapper(args.ckpt, device)
    print(f"model: {model.name} ({model.model_type})")

    trials = []
    for idx, track in enumerate(manifest["tracks"]):
        tdir = args.chunks / f"{idx:03d}"
        mix = np.load(tdir / f"mix{suffix}.npy")
        n = len(mix)
        stems = {s: np.load(tdir / f"{s}{suffix}.npy") for s in STEMS}

        z_mix = model.encode(mix)
        z_stems = {s: model.encode(stems[s]) for s in STEMS}

        for stem in STEMS:
            tgt = stems[stem]
            # --- reconstruction baseline ---
            recon = model.decode(z_stems[stem], n)
            # --- algebraic separation: z_target = z_mix - sum(z_others) ---
            z_others = sum(z_stems[o] for o in STEMS if o != stem)
            z_sep = z_mix - z_others
            sep = model.decode(z_sep, n)

            trials.append({
                "track": track["id"],
                "stem": stem,
                "recon_sisdr": si_sdr(tgt, recon),
                "sep_sisdr": si_sdr(tgt, sep),
                "recon_ms": ms_stft_mse(recon, tgt, device),
                "sep_ms": ms_stft_mse(sep, tgt, device),
            })
        if (idx + 1) % 20 == 0 or idx == len(manifest["tracks"]) - 1:
            print(f"  [{idx+1}/{len(manifest['tracks'])}] done")

    out = {
        "model": f"same-{sr}",
        "ckpt": args.ckpt,
        "sr": sr,
        "timestamp": datetime.now().isoformat(),
        "num_trials": len(trials),
        "trials": trials,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{len(trials)} trials -> {args.out}")


if __name__ == "__main__":
    main()
