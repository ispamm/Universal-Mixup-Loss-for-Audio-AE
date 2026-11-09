#!/usr/bin/env python3
"""
Aggregate the SAME vs Lin-CAE algebraic-separation results.

The two models operate at different sample rates (SAME: 16 kHz, Lin-CAE:
44.1 kHz) and have very different absolute reconstruction qualities, so raw
SI-SDR / MS-STFT values are NOT comparable across models.  Everything is
therefore reported NORMALIZED to each model's own reconstruction error on the
same clips (per-trial, same track + stem):

  * MS-STFT ratio   = sep_ms / recon_ms   (lower = better; <1 means the
    algebraic separation is as good as or better than the model's own
    plain reconstruction of the same stem).
  * SI-SDR delta    = sep_sisdr - recon_sisdr  (dB, higher = better; ~0 means
    separation is as good as the model's own reconstruction).

Both "ratio of means" and "mean of per-trial ratios" are reported (same
convention as evaluation/mix_recon_ratio.py).  A per-stem breakdown and a
per-trial paired comparison between the two models are included.

Usage:
    python evaluation/separation_compare/aggregate.py \
        --results evaluation/separation_compare/results_same.json \
                  evaluation/separation_compare/results_cae.json \
        --out     evaluation/separation_compare/separation_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

STEMS = ["drums", "bass", "other", "vocals"]


def stats(trials: list[dict], stem: str | None = None) -> dict:
    sel = [t for t in trials if stem is None or t["stem"] == stem]
    n = len(sel)

    # MS-STFT (all trials valid)
    recon_ms = np.array([t["recon_ms"] for t in sel], dtype=np.float64)
    sep_ms = np.array([t["sep_ms"] for t in sel], dtype=np.float64)
    ratios = sep_ms / recon_ms
    ratio_of_means = float(np.mean(sep_ms) / np.mean(recon_ms)) if np.mean(recon_ms) > 0 else float("nan")
    mean_of_ratios = float(np.mean(ratios))
    median_of_ratios = float(np.median(ratios))
    geomean_of_ratios = float(np.exp(np.mean(np.log(ratios))))

    # --- Percentage increment of separation MSE over reconstruction MSE ---
    # pct_i = (MSE_sep_i - MSE_recon_i) / MSE_recon_i * 100   ("how much larger
    # is the separation error than the model's own reconstruction error")
    pct = (sep_ms - recon_ms) / recon_ms * 100.0
    pct_agg = (np.mean(sep_ms) - np.mean(recon_ms)) / np.mean(recon_ms) * 100.0 if np.mean(recon_ms) > 0 else float("nan")
    ms_inc_pct_mean = float(np.mean(pct))
    ms_inc_pct_median = float(np.median(pct))

    # SI-SDR (drop NaN trials, e.g. silent targets)
    sisdr_ok = [t for t in sel if np.isfinite(t["sep_sisdr"]) and np.isfinite(t["recon_sisdr"])]
    n_ok = len(sisdr_ok)
    if n_ok:
        recon_sisdr = np.array([t["recon_sisdr"] for t in sisdr_ok])
        sep_sisdr = np.array([t["sep_sisdr"] for t in sisdr_ok])
        delta_sisdr = float(np.mean(sep_sisdr - recon_sisdr))
        delta_sisdr_median = float(np.median(sep_sisdr - recon_sisdr))
        recon_sisdr_mean = float(np.mean(recon_sisdr))
        sep_sisdr_mean = float(np.mean(sep_sisdr))
    else:
        delta_sisdr = delta_sisdr_median = recon_sisdr_mean = sep_sisdr_mean = float("nan")

    return {
        "n": n,
        "n_sisdr": n_ok,
        "recon_ms_mean": float(np.mean(recon_ms)),
        "sep_ms_mean": float(np.mean(sep_ms)),
        "ms_ratio_of_means": ratio_of_means,
        "ms_mean_of_ratios": mean_of_ratios,
        "ms_median_of_ratios": median_of_ratios,
        "ms_geomean_of_ratios": geomean_of_ratios,
        "ms_inc_pct_agg": pct_agg,
        "ms_inc_pct_mean": ms_inc_pct_mean,
        "ms_inc_pct_median": ms_inc_pct_median,
        "recon_sisdr_mean": recon_sisdr_mean,
        "sep_sisdr_mean": sep_sisdr_mean,
        "delta_sisdr_db": delta_sisdr,
        "delta_sisdr_db_median": delta_sisdr_median,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path,
                    default=[Path(__file__).resolve().parent / "results_same.json",
                             Path(__file__).resolve().parent / "results_cae.json"])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "separation_summary.json")
    args = ap.parse_args()

    results = {}
    for path in args.results:
        data = json.loads(path.read_text())
        results[data["model"]] = data

    summary = {
        "description": "SAME (16kHz) vs Lin-CAE (44.1kHz): algebraic source separation "
                       "(real full mix, remove-3-stems), normalized to each model's own "
                       "reconstruction error on the same clips.",
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "mix": "real MUSDB18-HQ mixture.wav (sum of 4 stems)",
            "separation": "z_target = z_mix - sum(z_other_stems); decode",
            "normalization": "per-trial, relative to the same model's plain reconstruction of the same stem",
            "metric_definitions": {
                "ms_inc_pct": "(sep_ms - recon_ms) / recon_ms * 100  (%% increment of separation MSE over the model's own reconstruction MSE; 0%% = separation as good as own recon, lower = better)",
                "delta_sisdr_db": "sep_sisdr - recon_sisdr (dB; same 0-dB anchor as mix_recon_ratio.py's delta_sisdr: 0 = separation as good as own recon, higher = better)",
            },
        },
        "models": {},
        "per_stem": {},
        "paired": {},
    }

    print(f"{'model':<9} {'n':>4} {'recon_ms':>10} {'sep_ms':>10} "
          f"{'inc%(agg)':>10} {'inc%(mean)':>10} {'inc%(med)':>10} "
          f"{'reconSISDR':>10} {'dSISDR(dB)':>11} {'dSISDR med':>10}")
    for name in results:
        s = stats(results[name]["trials"])
        summary["models"][name] = s
        print(f"{name:<9} {s['n']:>4} {s['recon_ms_mean']:>10.5f} {s['sep_ms_mean']:>10.5f} "
              f"{s['ms_inc_pct_agg']:>10.1f} {s['ms_inc_pct_mean']:>10.1f} {s['ms_inc_pct_median']:>10.1f} "
              f"{s['recon_sisdr_mean']:>10.2f} {s['delta_sisdr_db']:>11.2f} "
              f"{s['delta_sisdr_db_median']:>10.2f}")

    # ---- per-stem breakdown ----
    print("\nPer-stem (anchored metrics: 0 % / 0 dB = as good as own reconstruction):")
    print(f"{'stem':<9} {'model':<9} {'n':>4} {'inc%(agg)':>10} {'inc%(med)':>10} {'dSISDR(dB)':>11} {'dSISDR med':>10}")
    for stem in STEMS:
        summary["per_stem"][stem] = {}
        for name in results:
            s = stats(results[name]["trials"], stem=stem)
            summary["per_stem"][stem][name] = s
            print(f"{stem:<9} {name:<9} {s['n']:>4} {s['ms_inc_pct_agg']:>10.1f} "
                  f"{s['ms_inc_pct_median']:>10.1f} {s['delta_sisdr_db']:>11.2f} {s['delta_sisdr_db_median']:>10.2f}")

    # ---- pairwise comparison (aligned trials, all model pairs) ----
    names = list(results)
    paired_all = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ta = {t["track"] + "/" + t["stem"]: t for t in results[a]["trials"]}
            tb = {t["track"] + "/" + t["stem"]: t for t in results[b]["trials"]}
            keys = [k for k in ta if k in tb]
            pct_a = np.array([(ta[k]["sep_ms"] - ta[k]["recon_ms"]) / ta[k]["recon_ms"] * 100.0 for k in keys])
            pct_b = np.array([(tb[k]["sep_ms"] - tb[k]["recon_ms"]) / tb[k]["recon_ms"] * 100.0 for k in keys])
            keys_ok = [k for k in keys
                       if np.isfinite(ta[k]["sep_sisdr"]) and np.isfinite(ta[k]["recon_sisdr"])
                       and np.isfinite(tb[k]["sep_sisdr"]) and np.isfinite(tb[k]["recon_sisdr"])]
            da = np.array([ta[k]["sep_sisdr"] - ta[k]["recon_sisdr"] for k in keys_ok])
            db = np.array([tb[k]["sep_sisdr"] - tb[k]["recon_sisdr"] for k in keys_ok])

            paired = {
                "n_ms": len(keys),
                "mean_ms_inc_pct_a": float(np.mean(pct_a)),
                "mean_ms_inc_pct_b": float(np.mean(pct_b)),
                "median_ms_inc_pct_a": float(np.median(pct_a)),
                "median_ms_inc_pct_b": float(np.median(pct_b)),
                "mean_paired_ms_inc_pct_diff_a_minus_b": float(np.mean(pct_a - pct_b)),
                "n_sisdr": len(keys_ok),
                "mean_delta_sisdr_a": float(np.mean(da)),
                "mean_delta_sisdr_b": float(np.mean(db)),
                "median_delta_sisdr_a": float(np.median(da)),
                "median_delta_sisdr_b": float(np.median(db)),
                "mean_paired_delta_sisdr_diff_a_minus_b": float(np.mean(da - db)),
                "frac_trials_a_lower_ms_inc_pct": float(np.mean(pct_a < pct_b)),
                "frac_trials_a_better_delta_sisdr": float(np.mean(da > db)),
            }
            paired_all[f"{a}_vs_{b}"] = {"model_a": a, "model_b": b, **paired}
            print(f"\nPaired {a} vs {b} (same track+stem; 0 % / 0 dB = as good as own reconstruction):")
            print(f"  median MS inc%   {a}: {paired['median_ms_inc_pct_a']:+.1f}   {b}: {paired['median_ms_inc_pct_b']:+.1f}"
                  f"   (mean diff a-b: {paired['mean_paired_ms_inc_pct_diff_a_minus_b']:+.1f})")
            print(f"  median dSISDR(dB) {a}: {paired['median_delta_sisdr_a']:+.2f}   {b}: {paired['median_delta_sisdr_b']:+.2f}"
                  f"   (mean diff a-b: {paired['mean_paired_delta_sisdr_diff_a_minus_b']:+.2f})")
            print(f"  fraction of trials where {a} has lower MS inc%: {paired['frac_trials_a_lower_ms_inc_pct']:.3f}"
                  f" ; higher dSISDR: {paired['frac_trials_a_better_delta_sisdr']:.3f}")
    summary["paired"] = paired_all

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
