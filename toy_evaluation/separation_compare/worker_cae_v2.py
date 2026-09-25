#!/usr/bin/env python3
"""
Lin-CAE worker v2 — separation + trigonometric/linear latent mix.

Same protocol as ``worker_same_v2.py`` but loads Lin-CAE via
``linear_cae.Autoencoder``.  Runs in the cae_env conda environment.  The decode
is a one-step consistency-model reverse pass with random initial noise, so a
fixed seed is set before every decode for reproducibility.

Usage (cae_env, GPU):
    CUDA_VISIBLE_DEVICES=0 python toy_evaluation/separation_compare/worker_cae_v2.py \\
        --chunks toy_evaluation/separation_compare/chunks \\
        --out    toy_evaluation/separation_compare/results_v2_cae.json \\
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

from linear_cae import Autoencoder  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]
ALPHAS = np.linspace(0.0, 1.0, 9).tolist()
N_ALPHA = len(ALPHAS)
DECODE_SEED_BASE = 1234


def ideal_trig(a, b, alpha):
    theta = alpha * (math.pi / 2.0)
    return math.cos(theta) * a + math.sin(theta) * b


def ideal_lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path,
                    default=_THIS_FILE.parent / "results_v2_cae.json")
    ap.add_argument("--model_id", type=str, default="lin-cae")
    ap.add_argument("--ckpt_type", type=str, default="last")
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
    print(f"Lin-CAE v2 worker | sr={sr} device={device} tracks={n_tracks}")

    model = Autoencoder.from_pretrained(args.model_id, ckpt_type=args.ckpt_type)
    model = model.to(device).eval()
    print(f"model: {args.model_id} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

    recon_trials, sep_trials, mix_trials = [], [], []

    for idx, track in enumerate(tracks):
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
                torch.manual_seed(seed)
                recon = model.decode(z_stems[stem], full_length=n)
                recon = recon.squeeze(0).cpu().numpy().astype(np.float32)
                ri = si_sdr(tgt, recon)
                rm = ms_stft_mse(recon, tgt, device)
                recon_trials.append({
                    "track": track["id"], "stem": stem, "kind": "recon",
                    "recon_sisdr": ri, "recon_ms": rm,
                })
                z_others = sum(z_stems[o] for o in STEMS if o != stem)
                z_sep = z_mix - z_others
                torch.manual_seed(seed + 1)
                sep = model.decode(z_sep, full_length=n)
                sep = sep.squeeze(0).cpu().numpy().astype(np.float32)
                sep_trials.append({
                    "track": track["id"], "stem": stem, "kind": "sep",
                    "sep_sisdr": si_sdr(tgt, sep),
                    "sep_ms": ms_stft_mse(sep, tgt, device),
                    "recon_sisdr": ri, "recon_ms": rm,
                })

            for pj, (a, b) in enumerate(combinations(STEMS, 2)):
                za, zb = z_stems[a], z_stems[b]
                ta, tb = stems[a], stems[b]
                trig_sisdr_path, trig_ms_path = [], []
                lerp_sisdr_path, lerp_ms_path = [], []
                for ak, alpha in enumerate(ALPHAS):
                    theta = alpha * (math.pi / 2.0)
                    seed = DECODE_SEED_BASE + idx * 10 + si + 1 + pj * 9 + ak
                    torch.manual_seed(seed)
                    audio_trig = model.decode(
                        math.cos(theta) * za + math.sin(theta) * zb,
                        full_length=n)
                    audio_trig = audio_trig.squeeze(0).cpu().numpy().astype(np.float32)
                    it = ideal_trig(ta, tb, alpha)
                    trig_sisdr_path.append(si_sdr(it, audio_trig))
                    trig_ms_path.append(ms_stft_mse(audio_trig, it, device))
                    torch.manual_seed(seed + 1)
                    audio_lerp = model.decode(
                        (1.0 - alpha) * za + alpha * zb, full_length=n)
                    audio_lerp = audio_lerp.squeeze(0).cpu().numpy().astype(np.float32)
                    il = ideal_lerp(ta, tb, alpha)
                    lerp_sisdr_path.append(si_sdr(il, audio_lerp))
                    lerp_ms_path.append(ms_stft_mse(audio_lerp, il, device))
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
        "model": "lin-cae",
        "model_id": args.model_id,
        "ckpt_type": args.ckpt_type,
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
