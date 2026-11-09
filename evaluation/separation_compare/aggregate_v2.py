#!/usr/bin/env python3
"""
Aggregate v2 results — separation + trigonometric/linear latent mix.

Computes mean ± SEM (standard error = sample std / sqrt(n), ddof=1) for:
  * reconstruction quality (SI-SDR, MS-STFT MSE)
  * algebraic separation (SI-SDR, MS-STFT MSE) and ΔSI-SDR = sep - recon
  * trigonometric latent mix (SI-SDR, MS-STFT MSE vs ideal trig audio mix)
  * linear latent mix (SI-SDR, MS-STFT MSE vs ideal lerp audio mix)

For the mix tasks, in addition to the midpoint (α=0.5), path-following
trajectory metrics are computed across the full α ∈ [0,1] path (9 steps):
  * path_mean  — mean of the metric across α (how well the whole path is
                followed on average)
  * path_std   — std of the metric across α within each trial (path
                flatness / consistency; lower = more uniform)
  * path_range — max − min across α within each trial (worst-case
                deviation along the path)
  Each of these per-trial trajectory summaries is then aggregated across
  trials as mean ± SEM.

for each model, overall and per-stem (separation/recon) / per-pair (mix).
NaNs (silent-target SI-SDR) are dropped before computing statistics.

Usage:
    python evaluation/separation_compare/aggregate_v2.py \\
        --results evaluation/separation_compare/results_v2_same_441.json \\
                  evaluation/separation_compare/results_v2_same_l.json \\
                  evaluation/separation_compare/results_v2_cae.json \\
                  evaluation/separation_compare/results_v2_dac.json \\
        --out evaluation/separation_compare/separation_summary_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

STEMS = ["drums", "bass", "other", "vocals"]
PAIRS = [f"{a}+{b}" for a, b in combinations(STEMS, 2)]


def _mean_std(arr: np.ndarray) -> dict:
    """Mean and sample std (ddof=1).  Returns n, mean, std, sem."""
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan")}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    return {"n": n, "mean": mean, "std": std, "sem": sem}


def _path_stats(paths: list[list[float]]) -> dict:
    """Aggregate per-α path arrays into trajectory summary stats.

    ``paths`` is a list of per-trial α-trajectories (each length n_alpha).
    For each trial we compute the mean/std/range across α, then aggregate
    those per-trial summaries across trials with mean ± SEM.
    """
    M = np.array(paths, dtype=np.float64)  # (n_trials, n_alpha)
    # per-trial trajectory summaries (across α, axis=1)
    trial_means = np.nanmean(M, axis=1)
    trial_stds = np.nanstd(M, axis=1, ddof=1)
    trial_ranges = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
    return {
        "path_mean": _mean_std(trial_means),   # mean across-α, then mean±SEM across trials
        "path_std": _mean_std(trial_stds),     # std across-α (flatness), then mean±SEM
        "path_range": _mean_std(trial_ranges), # range across-α, then mean±SEM
    }


def model_stats(data: dict) -> dict:
    """Compute all statistics for one model's v2 result file."""
    recon = data["recon_trials"]
    sep = data["sep_trials"]
    mix = data["mix_trials"]

    # ---- reconstruction ----
    recon_sisdr = np.array([t["recon_sisdr"] for t in recon], dtype=np.float64)
    recon_ms = np.array([t["recon_ms"] for t in recon], dtype=np.float64)
    recon_overall = {
        "sisdr": _mean_std(recon_sisdr),
        "ms": _mean_std(recon_ms),
    }
    recon_per_stem = {}
    for stem in STEMS:
        sel = [t for t in recon if t["stem"] == stem]
        recon_per_stem[stem] = {
            "sisdr": _mean_std(np.array([t["recon_sisdr"] for t in sel])),
            "ms": _mean_std(np.array([t["recon_ms"] for t in sel])),
        }

    # ---- separation ----
    sep_sisdr = np.array([t["sep_sisdr"] for t in sep], dtype=np.float64)
    sep_ms = np.array([t["sep_ms"] for t in sep], dtype=np.float64)
    delta_sisdr = np.array([t["sep_sisdr"] - t["recon_sisdr"] for t in sep], dtype=np.float64)
    sep_overall = {
        "sisdr": _mean_std(sep_sisdr),
        "ms": _mean_std(sep_ms),
        "delta_sisdr": _mean_std(delta_sisdr),
    }
    sep_per_stem = {}
    for stem in STEMS:
        sel = [t for t in sep if t["stem"] == stem]
        sep_per_stem[stem] = {
            "sisdr": _mean_std(np.array([t["sep_sisdr"] for t in sel])),
            "ms": _mean_std(np.array([t["sep_ms"] for t in sel])),
            "delta_sisdr": _mean_std(np.array([t["sep_sisdr"] - t["recon_sisdr"] for t in sel])),
        }

    # ---- trig mix: midpoint + path-following ----
    trig_sisdr = np.array([t["trig_sisdr"] for t in mix], dtype=np.float64)
    trig_ms = np.array([t["trig_ms"] for t in mix], dtype=np.float64)
    trig_paths_sisdr = [t["trig_sisdr_path"] for t in mix]
    trig_paths_ms = [t["trig_ms_path"] for t in mix]
    trig_overall = {
        "sisdr": _mean_std(trig_sisdr),
        "ms": _mean_std(trig_ms),
        "path_sisdr": _path_stats(trig_paths_sisdr),
        "path_ms": _path_stats(trig_paths_ms),
    }
    trig_per_pair = {}
    for pair in PAIRS:
        a, b = pair.split("+")
        sel = [t for t in mix if t["stem_a"] == a and t["stem_b"] == b]
        trig_per_pair[pair] = {
            "sisdr": _mean_std(np.array([t["trig_sisdr"] for t in sel])),
            "ms": _mean_std(np.array([t["trig_ms"] for t in sel])),
            "path_sisdr": _path_stats([t["trig_sisdr_path"] for t in sel]),
            "path_ms": _path_stats([t["trig_ms_path"] for t in sel]),
        }

    # ---- lerp mix: midpoint + path-following ----
    lerp_sisdr = np.array([t["lerp_sisdr"] for t in mix], dtype=np.float64)
    lerp_ms = np.array([t["lerp_ms"] for t in mix], dtype=np.float64)
    lerp_paths_sisdr = [t["lerp_sisdr_path"] for t in mix]
    lerp_paths_ms = [t["lerp_ms_path"] for t in mix]
    lerp_overall = {
        "sisdr": _mean_std(lerp_sisdr),
        "ms": _mean_std(lerp_ms),
        "path_sisdr": _path_stats(lerp_paths_sisdr),
        "path_ms": _path_stats(lerp_paths_ms),
    }
    lerp_per_pair = {}
    for pair in PAIRS:
        a, b = pair.split("+")
        sel = [t for t in mix if t["stem_a"] == a and t["stem_b"] == b]
        lerp_per_pair[pair] = {
            "sisdr": _mean_std(np.array([t["lerp_sisdr"] for t in sel])),
            "ms": _mean_std(np.array([t["lerp_ms"] for t in sel])),
            "path_sisdr": _path_stats([t["lerp_sisdr_path"] for t in sel]),
            "path_ms": _path_stats([t["lerp_ms_path"] for t in sel]),
        }

    return {
        "model": data.get("model", "?"),
        "ckpt": data.get("ckpt", "?"),
        "sr": data.get("sr", "?"),
        "alphas": data.get("alphas", []),
        "n_recon": len(recon),
        "n_sep": len(sep),
        "n_mix": len(mix),
        "recon_overall": recon_overall,
        "recon_per_stem": recon_per_stem,
        "sep_overall": sep_overall,
        "sep_per_stem": sep_per_stem,
        "trig_overall": trig_overall,
        "trig_per_pair": trig_per_pair,
        "lerp_overall": lerp_overall,
        "lerp_per_pair": lerp_per_pair,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    models = {}
    order = []
    for path in args.results:
        data = json.loads(path.read_text())
        name = data.get("model", path.stem)
        models[name] = model_stats(data)
        order.append(name)

    summary = {
        "description": "Algebraic source separation + trigonometric/linear latent "
                       "mixing on MUSDB18-HQ (100 tracks, seed 42; 2 s mono chunks). "
                       "Metrics: SI-SDR (dB, higher=better) and MS-STFT magnitude MSE "
                       "(lower=better). Mix path metrics sweep α∈[0,1] (9 steps). "
                       "Stats: mean ± SEM (std/sqrt(n)).",
        "timestamp": datetime.now().isoformat(),
        "model_order": order,
        "models": models,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {args.out}")

    # ---- quick console preview ----
    alphas = models[order[0]].get("alphas", [])
    print(f"\nalphas: {alphas}")
    print(f"\n{'model':<12} {'reconSI-SDR':>16} {'sepSI-SDR':>16} {'ΔSI-SDR':>16} "
          f"{'trig mid':>12} {'trig path':>12} {'lerp mid':>12} {'lerp path':>12}")
    for name in order:
        m = models[name]
        def fms(d): return f"{d['mean']:.2f}±{d['sem']:.2f}"
        pm_t = m["trig_overall"]["path_sisdr"]["path_mean"]
        pm_l = m["lerp_overall"]["path_sisdr"]["path_mean"]
        print(f"{name:<12} {fms(m['recon_overall']['sisdr']):>16} "
              f"{fms(m['sep_overall']['sisdr']):>16} {fms(m['sep_overall']['delta_sisdr']):>16} "
              f"{m['trig_overall']['sisdr']['mean']:>10.2f} {fms(pm_t):>12} "
              f"{m['lerp_overall']['sisdr']['mean']:>10.2f} {fms(pm_l):>12}")
    print(f"\n{'model':<12} {'trig flat':>16} {'trig range':>16} {'lerp flat':>16} {'lerp range':>16}")
    for name in order:
        m = models[name]
        ps_t = m["trig_overall"]["path_sisdr"]["path_std"]
        pr_t = m["trig_overall"]["path_sisdr"]["path_range"]
        ps_l = m["lerp_overall"]["path_sisdr"]["path_std"]
        pr_l = m["lerp_overall"]["path_sisdr"]["path_range"]
        print(f"{name:<12} {fms(ps_t):>16} {fms(pr_t):>16} {fms(ps_l):>16} {fms(pr_l):>16}")


if __name__ == "__main__":
    main()
