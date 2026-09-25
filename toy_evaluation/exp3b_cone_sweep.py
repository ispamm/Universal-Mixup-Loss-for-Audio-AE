#!/usr/bin/env python3
"""
Exp 3b — Cone-fidelity sweep: WHERE in the positive-weight cone does each
universal-loss sampler leave its fingerprint?
================================================================================

WHY THIS TEST.  Exp 3 (`exp3_interp_laws.py`) scores the 7 interpolation laws,
but every one of them lives in the thin shell A = ||(w1, w2)|| ∈ [0.707, 1]
with θ sweeping [0, π/2] — the region where ALL universal-mix arms overlap, so
the sampler differences show up only as small recon-normalized gaps.  The three
sampler arms of the ablation differ most OUTSIDE that shell:

  * polar / cart_iso : A ~ U(gain_min, gain_max) (annealed (0,3) → (1,1)),
    w.p. gain_prob else A = 1;  θ ~ U(0, π/2)  (uniform in angle)
  * cart_square      : w_i ~ U(0, S)², S = annealed gain_max.  Radius
    A = √(w1²+w2²) has density ∝ A over [0, S√2] (no A=1 spike, mass at low
    and high gain); angle density f(θ) = ½sec²θ on [0, π/4] — 1.45× the mean
    at 45°, 0.79× at the axes (MC-verified; NOTE: the losses.py docstring's
    parenthetical states the opposite and should be corrected).

This script maps each checkpoint's decoder fidelity ACROSS the cone on the
same 600 MUSDB18 pairs (2 s @ 16 kHz, seed 42 chunks):

  * gain   : θ = 0, A ∈ [0.05 … 3] — pure gain-scaled reconstruction
             D(A·Z) vs clamp(A·X) per stem (the homogeneity objective; the
             recon branch's exact training signal, whose A-law differs per
             arm).
  * cone45 : θ = 45°, A ∈ [0.1 … 2.6] — equal-power mixing across the whole
             radial range (mixing + gain jointly).
  * angular: A = 1, θ ∈ [0° … 90°] — the constant-power arc (in-support for
             every arm; isolates the angular anisotropy).
  * multi4 / multi3 : additivity composition beyond pairs —
             D(¼·Σ Z_stem) vs ¼·Σ X_stem  (Σ w = 1 simplex point),
             D(Σ Z_stem)    vs the true mixture (high-gain sum, A = 2),
             D(⅓·Σ_{k≠lo} Z) vs ⅓·Σ_{k≠lo} X  (leave-one-out, k = 3).
             None of the arms trained on >2 sources, so this stresses how
             pairwise cone coverage composes.

Targets are clamped to [-1, 1] exactly like the training objective.  Metrics:
SI-SDR (↑), MS-STFT (↓), and RMS ratio D/ideal (gain linearity, ≈1 ideal).

Usage:
    python toy_evaluation/exp3b_cone_sweep.py \\
        --label cart_square --ckpt toy_training/checkpoints/ablation_sampler_cart_square_best.pt \\
        --device cuda:0 --out toy_evaluation/exp3b/results_conesweep_cart_square.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.exp3_interp_laws import (  # noqa: E402
    PAIRS,
    STEMS,
    load_adapter,
    rms,
)
from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

# ---- grids ------------------------------------------------------------------
GAIN_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.25, 1.5, 1.8, 2.2, 2.6, 3.0]
A_GRID_45 = [0.1, 0.2, 0.35, 0.5, 0.71, 1.0, 1.25, 1.5, 1.8, 2.2, 2.6]
THETA_GRID = [0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0]


def clamp01(x: np.ndarray) -> np.ndarray:
    """Training targets are clamped to [-1, 1] — match that here."""
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def score(dec: np.ndarray, ideal: np.ndarray, device: torch.device) -> dict:
    L = min(len(dec), len(ideal))
    d, r = dec[:L], ideal[:L]
    rr = rms(r)
    return {
        "sisdr": float(si_sdr(r, d)),
        "ms": float(ms_stft_mse(d, r, device)),
        "rms_ratio": float(rms(d) / rr) if rr > 1e-12 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="univ_441",
                    help="Adapter name (toy checkpoints: univ_441).")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sr", type=int, default=16000, choices=[44100, 16000])
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--chunks", type=Path,
                    default=_REPO_ROOT / "toy_evaluation/separation_compare/chunks")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_tracks", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = "44" if args.sr == 44100 else "16"
    sr_key = "sr44" if args.sr == 44100 else "sr16"
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    tracks = manifest["tracks"]
    if args.max_tracks is not None:
        tracks = tracks[: args.max_tracks]
    print(f"worker {args.label} | sr={manifest[sr_key]} device={device} "
          f"tracks={len(tracks)}")

    adapter = load_adapter(args.model, device, args.ckpt, suffix=suffix, sr=args.sr)
    c45 = math.cos(math.pi / 4.0)

    trials: list[dict] = []
    for idx, track in enumerate(tracks):
        tdir = args.chunks / f"{idx:03d}"
        stems = {s: np.load(tdir / f"{s}{suffix}.npy") for s in STEMS}
        n = min(min(len(x) for x in stems.values()),
                len(np.load(tdir / f"mix{suffix}.npy")))
        z = {s: adapter.encode(x) for s, x in stems.items()}

        # ---- 1. radial sweep at θ=0 (gain-scaled reconstruction per stem) ----
        for s in STEMS:
            x, zs = stems[s], z[s]
            for A in GAIN_GRID:
                dec = adapter.decode(A * zs, n)
                trials.append({"type": "gain", "track": track["id"], "stem": s,
                               "A": A, **score(dec, clamp01(A * x), device)})

        # ---- 2. radial sweep at θ=45° and angular sweep at A=1, per pair ----
        for (sa, sb) in PAIRS:
            xa, xb, za, zb = stems[sa], stems[sb], z[sa], z[sb]
            for A in A_GRID_45:
                w = A * c45
                dec = adapter.decode(w * za + w * zb, n)
                trials.append({"type": "cone45", "track": track["id"],
                              "pair": f"{sa}/{sb}", "A": A,
                              **score(dec, clamp01(w * xa + w * xb), device)})
            for th in THETA_GRID:
                t = math.radians(th)
                w1, w2 = math.cos(t), math.sin(t)
                dec = adapter.decode(w1 * za + w2 * zb, n)
                trials.append({"type": "angular", "track": track["id"],
                              "pair": f"{sa}/{sb}", "theta": th,
                              **score(dec, clamp01(w1 * xa + w2 * xb), device)})

        # ---- 3. multi-stem additivity composition (once per track) ----
        z_sum = sum(z[s] for s in STEMS)
        x_sum = sum(stems[s] for s in STEMS)  # == the mixture (MUSDB18 exact)
        for w, mode in ((0.25, "quarter"), (1.0, "sum")):
            dec = adapter.decode(w * z_sum, n)
            trials.append({"type": "multi4", "track": track["id"], "mode": mode,
                           **score(dec, clamp01(w * x_sum), device)})
        for lo in STEMS:
            z3 = sum(z[s] for s in STEMS if s != lo)
            x3 = sum(stems[s] for s in STEMS if s != lo)
            dec = adapter.decode(z3 / 3.0, n)
            trials.append({"type": "multi3", "track": track["id"], "left_out": lo,
                           **score(dec, clamp01(x3 / 3.0), device)})

        if (idx + 1) % 10 == 0 or idx == len(tracks) - 1:
            print(f"  [{idx + 1}/{len(tracks)}] tracks done, {len(trials)} points")

    out = {
        "model": args.label,
        "ckpt": args.ckpt,
        "sr": manifest[sr_key],
        "chunk_suffix": suffix,
        "grids": {"gain": GAIN_GRID, "cone45": A_GRID_45, "theta": THETA_GRID},
        "num_tracks": len(tracks),
        "num_points": len(trials),
        "timestamp": datetime.now().isoformat(),
        "trials": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{len(trials)} points -> {args.out}")


if __name__ == "__main__":
    main()
