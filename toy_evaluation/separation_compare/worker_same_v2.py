#!/usr/bin/env python3
"""
Unified SAME toy-model worker — separation + trigonometric/linear latent mix.

Runs in the project venv (torch 2.7.1+cu126) on cuda:0 by default.  Loads the
44.1 kHz chunks produced by ``prepare_chunks.py`` and, for every track:

  * reconstruction baseline : recon = decode(encode(stem))           (per stem)
  * algebraic separation    : z_sep = z_mix - sum(z_other_stems),
                              sep   = decode(z_sep)                  (per stem)
  * trigonometric latent mix: z_trig = cos(θ)·z_A + sin(θ)·z_B (θ=π/4),
                              dec, vs ideal audio trig mix           (per stem pair)
  * linear latent mix       : z_lerp = 0.5·z_A + 0.5·z_B,
                              dec, vs ideal audio lerp mix            (per stem pair)

with SI-SDR and MS-STFT MSE against the true target waveform.  Writes one JSON
consumed by ``aggregate_v2.py``.

Usage (venv, GPU):
    CUDA_VISIBLE_DEVICES=0 python toy_evaluation/separation_compare/worker_same_v2.py \\
        --chunks toy_evaluation/separation_compare/chunks \\
        --out    toy_evaluation/separation_compare/results_v2_same_441.json \\
        --ckpt   toy_training/checkpoints/transformer_best_full_441_best.pt \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from itertools import combinations
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
# Full interpolation path: 9 steps from 0.0 to 1.0 (inclusive endpoints).
ALPHAS = np.linspace(0.0, 1.0, 9).tolist()
N_ALPHA = len(ALPHAS)


def ideal_trig(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    theta = alpha * (math.pi / 2.0)
    return math.cos(theta) * a + math.sin(theta) * b


def ideal_lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path,
                    default=_THIS_FILE.parent / "results_v2_same_441.json")
    ap.add_argument("--ckpt", type=str,
                    default="toy_training/checkpoints/transformer_best_full_441_best.pt")
    ap.add_argument("--sr", type=int, default=44, choices=[16, 44])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of tracks (0 = all).")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    sr = manifest[f"sr{args.sr}"]
    suffix = f"{args.sr}"
    tracks = manifest["tracks"]
    if args.limit > 0:
        tracks = tracks[:args.limit]
    n_tracks = len(tracks)
    print(f"SAME-v2 worker | ckpt={Path(args.ckpt).name} sr={sr} device={device} "
          f"tracks={n_tracks}")

    model = ToyModelWrapper(args.ckpt, device)
    print(f"model: {model.name} ({model.model_type})")

    recon_trials = []
    sep_trials = []
    mix_trials = []  # trig + lerp per stem pair

    for idx, track in enumerate(tracks):
        tdir = args.chunks / f"{idx:03d}"
        mix = np.load(tdir / f"mix{suffix}.npy")
        n = len(mix)
        stems = {s: np.load(tdir / f"{s}{suffix}.npy") for s in STEMS}

        z_mix = model.encode(mix)
        z_stems = {s: model.encode(stems[s]) for s in STEMS}

        # --- reconstruction + separation (per stem) ---
        for stem in STEMS:
            tgt = stems[stem]
            recon = model.decode(z_stems[stem], n)
            recon_trials.append({
                "track": track["id"], "stem": stem, "kind": "recon",
                "recon_sisdr": si_sdr(tgt, recon),
                "recon_ms": ms_stft_mse(recon, tgt, device),
            })
            z_others = sum(z_stems[o] for o in STEMS if o != stem)
            z_sep = z_mix - z_others
            sep = model.decode(z_sep, n)
            sep_trials.append({
                "track": track["id"], "stem": stem, "kind": "sep",
                "sep_sisdr": si_sdr(tgt, sep),
                "sep_ms": ms_stft_mse(sep, tgt, device),
                # keep recon alongside for ΔSI-SDR convenience
                "recon_sisdr": recon_trials[-1]["recon_sisdr"],
                "recon_ms": recon_trials[-1]["recon_ms"],
            })

        # --- trig + lerp latent mix (per stem pair, full α path) ---
        for a, b in combinations(STEMS, 2):
            za, zb = z_stems[a], z_stems[b]
            ta, tb = stems[a], stems[b]
            trig_sisdr_path, trig_ms_path = [], []
            lerp_sisdr_path, lerp_ms_path = [], []
            for alpha in ALPHAS:
                theta = alpha * (math.pi / 2.0)
                # trigonometric
                z_trig = math.cos(theta) * za + math.sin(theta) * zb
                audio_trig = model.decode(z_trig, n)
                ideal_t = ideal_trig(ta, tb, alpha)
                trig_sisdr_path.append(si_sdr(ideal_t, audio_trig))
                trig_ms_path.append(ms_stft_mse(audio_trig, ideal_t, device))
                # lerp
                z_lerp = (1.0 - alpha) * za + alpha * zb
                audio_lerp = model.decode(z_lerp, n)
                ideal_l = ideal_lerp(ta, tb, alpha)
                lerp_sisdr_path.append(si_sdr(ideal_l, audio_lerp))
                lerp_ms_path.append(ms_stft_mse(audio_lerp, ideal_l, device))
            mix_trials.append({
                "track": track["id"], "stem_a": a, "stem_b": b,
                "alphas": ALPHAS,
                "trig_sisdr_path": trig_sisdr_path,
                "trig_ms_path": trig_ms_path,
                "lerp_sisdr_path": lerp_sisdr_path,
                "lerp_ms_path": lerp_ms_path,
                # midpoint (α=0.5) kept for backward comparability
                "trig_sisdr": trig_sisdr_path[N_ALPHA // 2],
                "trig_ms": trig_ms_path[N_ALPHA // 2],
                "lerp_sisdr": lerp_sisdr_path[N_ALPHA // 2],
                "lerp_ms": lerp_ms_path[N_ALPHA // 2],
            })

        if (idx + 1) % 10 == 0 or idx == n_tracks - 1:
            print(f"  [{idx+1}/{n_tracks}] done "
                  f"(recon={len(recon_trials)} sep={len(sep_trials)} "
                  f"mix={len(mix_trials)})")

    out = {
        "model": f"same-{sr}",
        "ckpt": args.ckpt,
        "sr": sr,
        "timestamp": datetime.now().isoformat(),
        "alphas": ALPHAS,
        "num_recon_trials": len(recon_trials),
        "num_sep_trials": len(sep_trials),
        "num_mix_trials": len(mix_trials),
        "recon_trials": recon_trials,
        "sep_trials": sep_trials,
        "mix_trials": mix_trials,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nrecon={len(recon_trials)} sep={len(sep_trials)} mix={len(mix_trials)} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
