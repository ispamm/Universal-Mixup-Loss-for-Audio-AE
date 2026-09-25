#!/usr/bin/env python3
"""
Decoder-path quality along generalized (non-convex, non-normalised)
interpolation laws — LSD + energy curves
=====================================================================

Companion to ``generalized_interpolation_eval.py`` (which measured encoder
homomorphism and, secondarily, ΔSI-SDR / MS-STFT of the decoded mixes).
This script focuses on the DECODER along the same three non-linear weight
laws (logarithmic / parabolic / exponential) over the interior grid
t ∈ [0.1, 0.9]:

  decoded(t) = D(w1(t)·Z_A + w2(t)·Z_B)      (latent weighted mix — the path
                                              the model must render)
  ideal(t)   = w1(t)·A + w2(t)·B             (the ideal acoustic weighted mix)

Per pair (A, B) and per t it reports:

  * LSD / LSD-active / LSD-energy-weighted of decoded vs ideal
    (``compute_lsd_decomposed`` — identical convention to the LIMS battery;
    lower = spectrally closer; active frames only avoid the silence floor
    of sparse targets).
  * energy(t) = rms(decoded) / rms(ideal)   (≈1 → the decoder renders the
    path's energy correctly; ≠1 → energy leak/drop).

Endpoint single-clip reconstructions (same pairs) are computed as the
reference ceiling, mirroring ``mix_recon_ratio.py`` conventions.

Usage:
    python toy_evaluation/generalized_decoder_path_eval.py \
        --checkpoints toy_training/checkpoints/transformer_univ_full_best.pt \
                       toy_training/checkpoints/transformer_implicit_full_best.pt \
                       toy_training/checkpoints/transformer_best_full_best.pt \
        --num_pairs 300 --seed 42 --device cuda:0 \
        --json toy_evaluation/generalized_decoder_path_summary.json
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
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from toy_evaluation.lims_test_toy import compute_lsd_decomposed, rms_energy
from toy_evaluation.model_utils import ToyModelWrapper

T_MIN, T_MAX = 0.1, 0.9


def law_logarithmic(t: float) -> tuple[float, float]:
    w2 = float(np.log10(1.0 + 9.0 * t))
    w1 = float(np.log10(1.0 + 9.0 * (1.0 - t)))
    return w1, w2


def law_parabolic(t: float) -> tuple[float, float]:
    return float((1.0 - t) ** 2), float(t ** 2)


def law_exponential(t: float) -> tuple[float, float]:
    w2 = float((10.0 ** (2.0 * t) - 1.0) / 99.0)
    w1 = float((10.0 ** (2.0 * (1.0 - t)) - 1.0) / 99.0)
    return w1, w2


LAWS = {
    "logarithmic": law_logarithmic,
    "parabolic": law_parabolic,
    "exponential": law_exponential,
}
LAW_DESCRIPTIONS = {
    "logarithmic": "w2=log10(1+9t), w1=log10(1+9(1-t))  [audio fader]",
    "parabolic": "w2=t^2, w1=(1-t)^2",
    "exponential": "w2=(10^(2t)-1)/99, w1=(10^(2(1-t))-1)/99",
}


def evaluate_checkpoint(
    ckpt_path: str,
    device: torch.device,
    stems,
    pairs,
    clip_samples: int,
    t_grid: np.ndarray,
) -> dict:
    model = ToyModelWrapper(ckpt_path, device)
    name = Path(ckpt_path).stem
    n_t = len(t_grid)
    print(f"=== {name} ===  pairs={len(pairs)}  t ∈ [{T_MIN}, {T_MAX}] ({n_t} pts)")

    acc = {law: {
        "lsd": np.full((len(pairs), n_t), np.nan),
        "lsd_active": np.full((len(pairs), n_t), np.nan),
        "lsd_energy_weighted": np.full((len(pairs), n_t), np.nan),
        "energy_ratio": np.full((len(pairs), n_t), np.nan),
    } for law in LAWS}
    recon_lsd_active = np.full((len(pairs),), np.nan)
    recon_lsd = np.full((len(pairs),), np.nan)

    for p_idx, (i, j) in enumerate(pairs):
        a = stem_to_numpy(stems[i][0], clip_samples, centre=True)
        b = stem_to_numpy(stems[j][0], clip_samples, centre=True)
        n = len(a)
        z_a = model.encode(a)
        z_b = model.encode(b)
        if z_a.shape[-1] != z_b.shape[-1]:
            mlen = min(z_a.shape[-1], z_b.shape[-1])
            z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]

        # endpoint reconstruction ceiling (same clips as the mixes)
        ra = model.decode(z_a, n)
        rb = model.decode(z_b, n)
        d = compute_lsd_decomposed(a, ra)
        e = compute_lsd_decomposed(b, rb)
        recon_lsd_active[p_idx] = float(np.nanmean([d["lsd_active"], e["lsd_active"]]))
        recon_lsd[p_idx] = float(np.nanmean([d["lsd"], e["lsd"]]))

        for law, fn in LAWS.items():
            for t_idx, t in enumerate(t_grid):
                w1, w2 = fn(float(t))
                ideal = w1 * a + w2 * b
                z_mix = w1 * z_a + w2 * z_b
                decoded = model.decode(z_mix, n)
                m = compute_lsd_decomposed(ideal, decoded)
                acc[law]["lsd"][p_idx, t_idx] = m["lsd"]
                acc[law]["lsd_active"][p_idx, t_idx] = m["lsd_active"]
                acc[law]["lsd_energy_weighted"][p_idx, t_idx] = m["lsd_energy_weighted"]
                rms_ideal = rms_energy(ideal)
                acc[law]["energy_ratio"][p_idx, t_idx] = (
                    rms_energy(decoded) / (rms_ideal + 1e-12)
                )

        if (p_idx + 1) % 50 == 0 or p_idx == len(pairs) - 1:
            print(f"    pair {p_idx + 1}/{len(pairs)}  recon LSD-active "
                  f"{np.nanmean(recon_lsd_active[:p_idx + 1]):.2f} dB")

    results = {}
    for law in LAWS:
        per_t = {}
        for key in ("lsd", "lsd_active", "lsd_energy_weighted", "energy_ratio"):
            curve = np.nanmean(acc[law][key], axis=0)
            per_t[key] = {
                "mean": float(np.nanmean(curve)),
                "std_over_pairs": float(np.nanmean(np.nanstd(acc[law][key], axis=0))),
                "flatness_std_over_t": float(np.nanstd(curve)),
                "per_t_mean": curve.tolist(),
                "at_t_mid": float(curve[int(np.argmin(np.abs(t_grid - 0.5)))]),
            }
        results[law] = {
            "law_definition": LAW_DESCRIPTIONS[law],
            "t_grid": t_grid.tolist(),
            **per_t,
        }
        print(f"  [{law:12s}] LSD-act mid={results[law]['lsd_active']['at_t_mid']:.2f} "
              f"mean={results[law]['lsd_active']['mean']:.2f} dB | "
              f"LSD mid={results[law]['lsd']['at_t_mid']:.2f} dB | "
              f"energy mid={results[law]['energy_ratio']['at_t_mid']:.3f} "
              f"(flat {results[law]['energy_ratio']['flatness_std_over_t']:.3f})")

    return {
        "name": name,
        "checkpoint": ckpt_path,
        "recon_lsd_active_mean": float(np.nanmean(recon_lsd_active)),
        "recon_lsd_mean": float(np.nanmean(recon_lsd)),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decoder-path LSD / energy evaluation along generalized laws."
    )
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--num_pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--json", type=Path,
                    default=_EVAL_DIR / "generalized_decoder_path_summary.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t_grid = np.linspace(T_MIN, T_MAX, args.steps)
    stems = collect_validation_stems()
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)
    print(f"pairs={len(pairs)} seed={args.seed} device={device}")

    models = [evaluate_checkpoint(c, device, stems, pairs, clip_samples, t_grid)
              for c in args.checkpoints]
    out = {
        "description": "Decoder-path quality (LSD + RMS energy) along generalized "
                       "interpolation laws — same pairs/protocol for all models",
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "seed": args.seed,
        "device": str(device),
        "t_grid": t_grid.tolist(),
        "laws": LAW_DESCRIPTIONS,
        "models": models,
    }
    args.json.write_text(json.dumps(out, indent=2))
    print(f"\n📄 JSON -> {args.json}")


if __name__ == "__main__":
    main()
