#!/usr/bin/env python3
"""
Original SAME (same-l) worker for the algebraic separation study.

Runs in the project venv (torch 2.7.1+cu126) on cuda:0 by default.  Loads the
44.1 kHz chunks produced by ``prepare_chunks.py`` and computes, for every
(track, stem):

  * reconstruction baseline : recon = decode(encode(stem))           -> recon_*
  * algebraic separation    : z_sep = z_mix - sum(z_other_stems),
                              sep   = decode(z_sep)                  -> sep_*

with SI-SDR and MS-STFT MSE against the true stem waveform (mono; the SAME
autoencoder is fed a stereo copy of the mono chunk, matching the usage in
``same_studies/lims_test_same.py``).  Output schema is identical to the other
workers so the same ``aggregate.py`` consumes it.

Usage (venv):
    python toy_evaluation/separation_compare/worker_same_orig.py \
        --chunks toy_evaluation/separation_compare/chunks \
        --out    toy_evaluation/separation_compare/results_same_l.json \
        --model_name same-l --device cuda:0
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

from stable_audio_3 import AutoencoderModel  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]


def common_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim/pad both arrays to the common usable length."""
    L = min(len(a), len(b))
    return a[:L], b[:L]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path, default=_THIS_FILE.parent / "results_same_l.json")
    ap.add_argument("--model_name", type=str, default="same-l")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    sr = manifest["sr44"]
    print(f"SAME-{args.model_name} worker | sr={sr} device={device} "
          f"tracks={len(manifest['tracks'])}")

    model = AutoencoderModel.from_pretrained(args.model_name, device=str(device))
    print(f"model sample rate: {model.sample_rate}")

    trials = []
    for idx, track in enumerate(manifest["tracks"]):
        tdir = args.chunks / f"{idx:03d}"
        mix = np.load(tdir / "mix44.npy")
        n = len(mix)
        stems = {s: np.load(tdir / f"{s}44.npy") for s in STEMS}

        def _enc(x: np.ndarray) -> torch.Tensor:
            # mono (n,) -> stereo (2, n) tensor, as lims_test_same.py does
            audio_t = torch.from_numpy(np.stack([x, x]).astype(np.float32)).to(device)
            return model.encode(audio_t, sr=sr)  # (1, D, T_latent)

        def _dec(z: torch.Tensor) -> np.ndarray:
            out = model.decode(z)  # (1, C, samples)
            mono = out.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
            return mono

        with torch.inference_mode():
            z_mix = _enc(mix)
            z_stems = {s: _enc(stems[s]) for s in STEMS}

            for stem in STEMS:
                tgt = stems[stem]
                recon = _dec(z_stems[stem])
                z_others = sum(z_stems[o] for o in STEMS if o != stem)
                sep = _dec(z_mix - z_others)

                r_t, r_e = common_len(tgt, recon)
                s_t, s_e = common_len(tgt, sep)
                trials.append({
                    "track": track["id"],
                    "stem": stem,
                    "recon_sisdr": si_sdr(r_t, r_e),
                    "sep_sisdr": si_sdr(s_t, s_e),
                    "recon_ms": ms_stft_mse(r_e, r_t, device),
                    "sep_ms": ms_stft_mse(s_e, s_t, device),
                })
        if (idx + 1) % 10 == 0 or idx == len(manifest["tracks"]) - 1:
            print(f"  [{idx+1}/{len(manifest['tracks'])}] done")

    out = {
        "model": args.model_name,
        "sr": sr,
        "timestamp": datetime.now().isoformat(),
        "num_trials": len(trials),
        "trials": trials,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{len(trials)} trials -> {args.out}")


if __name__ == "__main__":
    main()
