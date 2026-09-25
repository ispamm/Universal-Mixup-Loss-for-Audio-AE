#!/usr/bin/env python3
"""
Lin-CAE worker for the algebraic separation study.

Runs in the cae_env conda environment (torch 2.7.1+cu126 after the driver-
compatible reinstall) on cuda:0 by default.  Loads the 44.1 kHz chunks produced
by ``prepare_chunks.py`` and computes, for every (track, stem):

  * reconstruction baseline : recon = decode(encode(stem))           -> recon_*
  * algebraic separation    : z_sep = z_mix - sum(z_other_stems),
                              sep   = decode(z_sep)                  -> sep_*

with SI-SDR and MS-STFT MSE against the true stem waveform.  The decode is a
one-step consistency-model reverse pass that draws random initial noise, so a
deterministic seed is set before every decode for reproducibility.

Usage (cae_env):
    python toy_evaluation/separation_compare/worker_cae.py \
        --chunks toy_evaluation/separation_compare/chunks \
        --out    toy_evaluation/separation_compare/results_cae.json \
        --model_id lin-cae --device cuda:0
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

from linear_cae import Autoencoder  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]
DECODE_SEED_BASE = 1234


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path, default=_THIS_FILE.parent / "results_cae.json")
    ap.add_argument("--model_id", type=str, default="lin-cae")
    ap.add_argument("--ckpt_type", type=str, default="last")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    print(f"Lin-CAE worker | sr={manifest['sr44']} device={device} "
          f"tracks={len(manifest['tracks'])}")

    model = Autoencoder.from_pretrained(args.model_id, ckpt_type=args.ckpt_type)
    model = model.to(device).eval()
    print(f"model: {args.model_id} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

    trials = []
    for idx, track in enumerate(manifest["tracks"]):
        tdir = args.chunks / f"{idx:03d}"
        mix = np.load(tdir / "mix44.npy")
        n = len(mix)
        stems = {s: np.load(tdir / f"{s}44.npy") for s in STEMS}

        with torch.no_grad():
            z_mix = model.encode(torch.from_numpy(mix).unsqueeze(0).to(device))
            z_stems = {
                s: model.encode(torch.from_numpy(stems[s]).unsqueeze(0).to(device))
                for s in STEMS
            }

            for si, stem in enumerate(STEMS):
                tgt = stems[stem]
                seed = DECODE_SEED_BASE + idx * 10 + si

                # --- reconstruction baseline ---
                torch.manual_seed(seed)
                recon = model.decode(z_stems[stem], full_length=n)
                recon = recon.squeeze(0).cpu().numpy().astype(np.float32)

                # --- algebraic separation: z_target = z_mix - sum(z_others) ---
                z_others = sum(z_stems[o] for o in STEMS if o != stem)
                z_sep = z_mix - z_others
                torch.manual_seed(seed + 1)
                sep = model.decode(z_sep, full_length=n)
                sep = sep.squeeze(0).cpu().numpy().astype(np.float32)

                trials.append({
                    "track": track["id"],
                    "stem": stem,
                    "recon_sisdr": si_sdr(tgt, recon),
                    "sep_sisdr": si_sdr(tgt, sep),
                    "recon_ms": ms_stft_mse(recon, tgt, device),
                    "sep_ms": ms_stft_mse(sep, tgt, device),
                })
        if (idx + 1) % 10 == 0 or idx == len(manifest["tracks"]) - 1:
            print(f"  [{idx+1}/{len(manifest['tracks'])}] done")

    out = {
        "model": "lin-cae",
        "model_id": args.model_id,
        "ckpt_type": args.ckpt_type,
        "sr": manifest["sr44"],
        "timestamp": datetime.now().isoformat(),
        "num_trials": len(trials),
        "trials": trials,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{len(trials)} trials -> {args.out}")


if __name__ == "__main__":
    main()
