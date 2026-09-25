#!/usr/bin/env python3
"""
Original SAME (same-l) worker v2 — separation + trigonometric/linear latent mix.

Same protocol as ``worker_same_v2.py`` but loads the original SAME (same-l)
autoencoder via ``stable_audio_3.AutoencoderModel``.  Mono chunks are fed as a
stereo copy (matching ``same_studies/lims_test_same.py``).

Usage (venv, GPU):
    CUDA_VISIBLE_DEVICES=0 python toy_evaluation/separation_compare/worker_same_orig_v2.py \\
        --chunks toy_evaluation/separation_compare/chunks \\
        --out    toy_evaluation/separation_compare/results_v2_same_l.json \\
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

from stable_audio_3 import AutoencoderModel  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]
ALPHAS = np.linspace(0.0, 1.0, 9).tolist()
N_ALPHA = len(ALPHAS)


def ideal_trig(a, b, alpha):
    theta = alpha * (math.pi / 2.0)
    return math.cos(theta) * a + math.sin(theta) * b


def ideal_lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b


def common_len(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path,
                    default=_THIS_FILE.parent / "results_v2_same_l.json")
    ap.add_argument("--model_name", type=str, default="same-l")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    sr = manifest["sr44"]
    tracks = manifest["tracks"]
    if args.limit > 0:
        tracks = tracks[:args.limit]
    n_tracks = len(tracks)
    print(f"SAME-{args.model_name} v2 worker | sr={sr} device={device} "
          f"tracks={n_tracks}")

    model = AutoencoderModel.from_pretrained(args.model_name, device=str(device))
    print(f"model sample rate: {model.sample_rate}")

    def _enc(x):
        audio_t = torch.from_numpy(np.stack([x, x]).astype(np.float32)).to(device)
        return model.encode(audio_t, sr=sr)

    def _dec(z):
        out = model.decode(z)
        return out.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)

    recon_trials, sep_trials, mix_trials = [], [], []

    for idx, track in enumerate(tracks):
        tdir = args.chunks / f"{idx:03d}"
        mix = np.load(tdir / "mix44.npy")
        n = len(mix)
        stems = {s: np.load(tdir / f"{s}44.npy") for s in STEMS}

        with torch.inference_mode():
            z_mix = _enc(mix)
            z_stems = {s: _enc(stems[s]) for s in STEMS}

            for stem in STEMS:
                tgt = stems[stem]
                recon = _dec(z_stems[stem])
                r_t, r_e = common_len(tgt, recon)
                ri = si_sdr(r_t, r_e)
                rm = ms_stft_mse(r_e, r_t, device)
                recon_trials.append({
                    "track": track["id"], "stem": stem, "kind": "recon",
                    "recon_sisdr": ri, "recon_ms": rm,
                })
                z_others = sum(z_stems[o] for o in STEMS if o != stem)
                sep = _dec(z_mix - z_others)
                s_t, s_e = common_len(tgt, sep)
                sep_trials.append({
                    "track": track["id"], "stem": stem, "kind": "sep",
                    "sep_sisdr": si_sdr(s_t, s_e),
                    "sep_ms": ms_stft_mse(s_e, s_t, device),
                    "recon_sisdr": ri, "recon_ms": rm,
                })

            for a, b in combinations(STEMS, 2):
                za, zb = z_stems[a], z_stems[b]
                ta, tb = stems[a], stems[b]
                trig_sisdr_path, trig_ms_path = [], []
                lerp_sisdr_path, lerp_ms_path = [], []
                for alpha in ALPHAS:
                    theta = alpha * (math.pi / 2.0)
                    audio_trig = _dec(math.cos(theta) * za + math.sin(theta) * zb)
                    ideal_t = ideal_trig(ta, tb, alpha)
                    it_t, it_e = common_len(ideal_t, audio_trig)
                    trig_sisdr_path.append(si_sdr(it_t, it_e))
                    trig_ms_path.append(ms_stft_mse(it_e, it_t, device))
                    audio_lerp = _dec((1.0 - alpha) * za + alpha * zb)
                    ideal_l = ideal_lerp(ta, tb, alpha)
                    il_t, il_e = common_len(ideal_l, audio_lerp)
                    lerp_sisdr_path.append(si_sdr(il_t, il_e))
                    lerp_ms_path.append(ms_stft_mse(il_e, il_t, device))
                mix_trials.append({
                    "track": track["id"], "stem_a": a, "stem_b": b,
                    "alphas": ALPHAS,
                    "trig_sisdr_path": trig_sisdr_path,
                    "trig_ms_path": trig_ms_path,
                    "lerp_sisdr_path": lerp_sisdr_path,
                    "lerp_ms_path": lerp_ms_path,
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
        "model": args.model_name,
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
