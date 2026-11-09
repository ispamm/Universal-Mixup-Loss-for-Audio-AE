#!/usr/bin/env python3
"""
ICLR Experiment 3 — aggregator for held-out interpolation laws.
================================================================

Reads one results JSON per model (from exp3_interp_laws.py) and produces:

Per (model, law):
  * path_mean_ms    — per pair: mean over the 9 α points of mix_ms   (lower=better)
  * path_mean_sisdr — per pair: mean over the 9 α points of mix_sisdr (higher=better)
  then over pairs:
    mean ± std, and SEM = std / sqrt(n)   (the user's "std of the mean value"
    to judge whether the sample set is statistically significant).

Across-law summary per model:
  * law_invariance_score = std over laws of (mean path MS-STFT)  (lower = more universal)
  * overall mean MS-STFT / SI-SDR (mean over all laws × pairs)

The 4 models: univ_441 (ours), same-l, lin-cae, dac  (trig_441 REMOVED).

Usage:
    python evaluation/aggregate_exp3.py \\
        --results evaluation/exp3/results_univ_441.json \\
                  evaluation/exp3/results_same-l.json \\
                  evaluation/exp3/results_lin-cae.json \\
                  evaluation/exp3/results_dac.json \\
        --out evaluation/exp3/exp3_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _arr(trial: dict, key: str) -> np.ndarray:
    return np.asarray(trial[key], dtype=np.float64)


def per_pair_path_means(trial: dict) -> dict:
    """For one (pair, law) trial, average the 9 α points into a single scalar."""
    ms = _arr(trial, "mix_ms")
    sisdr = _arr(trial, "mix_sisdr")
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "path_mean_ms": float(np.nanmean(ms)),
            "path_mean_sisdr": float(np.nanmean(sisdr)),
            # also keep interior (α∈[0.1,0.9]) for reference
            "int_mean_ms": float(np.nanmean(ms[1:-1])) if len(ms) > 2 else float(np.nanmean(ms)),
            "int_mean_sisdr": float(np.nanmean(sisdr[1:-1])) if len(sisdr) > 2 else float(np.nanmean(sisdr)),
        }


def law_stats(trials: list[dict], law: str) -> dict:
    """Aggregate all pairs for one (model, law)."""
    sel = [t for t in trials if t["law"] == law]
    if not sel:
        return None
    n = len(sel)
    pm_ms = np.array([per_pair_path_means(t)["path_mean_ms"] for t in sel])
    pm_sisdr = np.array([per_pair_path_means(t)["path_mean_sisdr"] for t in sel])
    # reconstruction anchors (shared across laws for the same pair, but we
    # average them per-law subset for completeness)
    recon_ms = np.array([t["recon_ms"] for t in sel])
    recon_sisdr = np.array([t["recon_sisdr"] for t in sel])

    valid_ms = pm_ms[~np.isnan(pm_ms)]
    valid_sisdr = pm_sisdr[~np.isnan(pm_sisdr)]
    n_ms = len(valid_ms)
    n_sisdr = len(valid_sisdr)

    sem_ms = float(np.std(valid_ms, ddof=1) / np.sqrt(n_ms)) if n_ms > 1 else float("nan")
    sem_sisdr = float(np.std(valid_sisdr, ddof=1) / np.sqrt(n_sisdr)) if n_sisdr > 1 else float("nan")

    # 95% CI half-width (normal approx) for the mean — a quick significance gauge
    from math import sqrt
    z95 = 1.959963984540054
    ci95_ms = z95 * sem_ms if n_ms > 1 else float("nan")
    ci95_sisdr = z95 * sem_sisdr if n_sisdr > 1 else float("nan")

    return {
        "n_pairs": n,
        "n_valid_ms": n_ms,
        "n_valid_sisdr": n_sisdr,
        # path-mean MS-STFT (lower = better)
        "path_ms_mean": float(np.nanmean(pm_ms)),
        "path_ms_std": float(np.nanstd(pm_ms, ddof=1)) if n_ms > 1 else 0.0,
        "path_ms_sem": sem_ms,
        "path_ms_ci95": ci95_ms,
        "path_ms_median": float(np.nanmedian(pm_ms)),
        # path-mean SI-SDR (higher = better)
        "path_sisdr_mean": float(np.nanmean(pm_sisdr)),
        "path_sisdr_std": float(np.nanstd(pm_sisdr, ddof=1)) if n_sisdr > 1 else 0.0,
        "path_sisdr_sem": sem_sisdr,
        "path_sisdr_ci95": ci95_sisdr,
        "path_sisdr_median": float(np.nanmedian(pm_sisdr)),
        # reconstruction anchor (same for every law; reported once per model)
        "recon_ms_mean": float(np.nanmean(recon_ms)),
        "recon_sisdr_mean": float(np.nanmean(recon_sisdr)),
    }


def model_stats(data: dict) -> dict:
    trials = data["trials"]
    laws = data["laws"]
    out = {"model": data["model"], "n_trials": len(trials), "laws": {}}
    law_means_ms = []
    law_means_sisdr = []
    for law in laws:
        s = law_stats(trials, law)
        if s is None:
            continue
        out["laws"][law] = s
        law_means_ms.append(s["path_ms_mean"])
        law_means_sisdr.append(s["path_sisdr_mean"])
    # cross-law invariance
    out["law_invariance_ms"] = float(np.std(law_means_ms, ddof=1)) if len(law_means_ms) > 1 else 0.0
    out["law_invariance_sisdr"] = float(np.std(law_means_sisdr, ddof=1)) if len(law_means_sisdr) > 1 else 0.0
    out["overall_ms_mean"] = float(np.mean(law_means_ms)) if law_means_ms else float("nan")
    out["overall_sisdr_mean"] = float(np.mean(law_means_sisdr)) if law_means_sisdr else float("nan")
    # reconstruction anchor (report once, from the first law)
    first_law = laws[0]
    if first_law in out["laws"]:
        out["recon_ms_mean"] = out["laws"][first_law]["recon_ms_mean"]
        out["recon_sisdr_mean"] = out["laws"][first_law]["recon_sisdr_mean"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for path in args.results:
        data = json.loads(path.read_text())
        results[data["model"]] = data
    order = list(results)
    # canonical display order: ours first
    pref = ["univ_441", "same-l", "lin-cae", "dac"]
    order = [m for m in pref if m in results] + [m for m in order if m not in pref]
    print(f"loaded {len(results)} models: {order}")

    laws = results[order[0]]["laws"]
    summary = {
        "description": "ICLR Exp 3 — held-out interpolation laws (44.1 kHz, 4 models, "
                       "trig_441 removed). Path-mean MS-STFT (lower=better) and SI-SDR "
                       "(higher=better) averaged over 9 α points per pair, mean ± std "
                       "over 600 pairs, with SEM = std/√n.",
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "interp": "z_mix = w1(α)·z_A + w2(α)·z_B; ideal = w1(α)·A + w2(α)·B",
            "laws": laws,
            "alphas": results[order[0]]["alphas"],
            "steps": results[order[0]]["steps"],
            "pairs": "all 6 same-track stem pairs × 100 MUSDB18 tracks (seed 42) = 600 pairs",
            "metric": "path_mean = mean over 9 α points per pair; then mean ± std over pairs; "
                      "SEM = std/√n (significance gauge)",
        },
        "models": {},
    }

    # ---- per-model per-law table ----
    print("\n=== Path-mean MS-STFT (lower = better) — mean ± std (SEM) over 600 pairs ===")
    hdr = f"{'law':<10}" + "".join(f"{m:>26}" for m in order)
    print(hdr)
    for law in laws:
        row = f"{law:<10}"
        for m in order:
            s = summary["models"].setdefault(m, model_stats(results[m]))["laws"].get(law)
            if s is None:
                row += f"{'—':>26}"
                continue
            row += f"  {s['path_ms_mean']:>8.4f}±{s['path_ms_std']:<7.4f}({s['path_ms_sem']:.4f})"
        print(row)

    print("\n=== Path-mean SI-SDR dB (higher = better) — mean ± std (SEM) over 600 pairs ===")
    print(hdr)
    for law in laws:
        row = f"{law:<10}"
        for m in order:
            s = summary["models"][m]["laws"].get(law)
            if s is None:
                row += f"{'—':>26}"
                continue
            row += f"  {s['path_sisdr_mean']:>7.2f}±{s['path_sisdr_std']:<7.2f}({s['path_sisdr_sem']:.2f})"
        print(row)

    # ---- cross-law invariance (the universality summary) ----
    print("\n=== Law-invariance score (std over laws of mean path metric; lower = more universal) ===")
    print(f"{'model':<10} {'MS-STFT std':>14} {'SI-SDR std':>12} {'overall MS':>12} {'overall SISDR':>14}")
    for m in order:
        ms = summary["models"][m]
        print(f"{m:<10} {ms['law_invariance_ms']:>14.4f} {ms['law_invariance_sisdr']:>12.3f} "
              f"{ms['overall_ms_mean']:>12.4f} {ms['overall_sisdr_mean']:>14.3f}")

    # ---- reconstruction anchors ----
    print("\n=== Reconstruction anchor (each model's own recon of the same stems) ===")
    print(f"{'model':<10} {'recon SI-SDR dB':>16} {'recon MS-STFT':>14}")
    for m in order:
        ms = summary["models"][m]
        print(f"{m:<10} {ms.get('recon_sisdr_mean', float('nan')):>16.3f} "
              f"{ms.get('recon_ms_mean', float('nan')):>14.4f}")

    # ---- significance: is SEM small relative to cross-model differences? ----
    print("\n=== Significance check: smallest cross-model MS-STFT gap vs max SEM ===")
    for law in laws:
        means = {m: summary["models"][m]["laws"][law]["path_ms_mean"] for m in order
                 if law in summary["models"][m]["laws"]}
        sems = {m: summary["models"][m]["laws"][law]["path_ms_sem"] for m in order
                if law in summary["models"][m]["laws"]}
        if len(means) < 2:
            continue
        vals = sorted(means.values())
        min_gap = vals[1] - vals[0]
        max_sem = max(sems.values())
        ratio = min_gap / max_sem if max_sem > 0 else float("inf")
        print(f"  {law:<10} min model gap = {min_gap:.4f}, max SEM = {max_sem:.4f}, "
              f"gap/SEM = {ratio:.1f}  ({'significant' if ratio > 3 else 'marginal' if ratio > 1.5 else 'low'})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
