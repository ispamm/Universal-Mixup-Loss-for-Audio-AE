#!/usr/bin/env python3
"""
Exp 3 (sampler ablation) — PAIRED significance analysis over identical pairs.
=============================================================================

``aggregate_exp3.py`` treats each model independently (mean ± std, SEM over
pairs).  But the exp3 protocol is *paired*: every model decodes the SAME
(track, stem-pair, law, α) grid, so cross-model comparisons should use
per-pair differences, not between-model SEM overlap.  This script adds:

  1. Pair-level stats per (model, law): mean ± std, SEM = std/√n_pairs.
  2. Track-level stats: the 6 stem-pairs of a track share the track, so
     pair-level SEM understates uncertainty if track effects dominate.
     We therefore also average the 6 pairs per track and report
     SEM = std/√n_tracks (the honest, clustering-robust SEM).
  3. Paired comparisons for every model pair: per-pair Δ with std, SEM,
     95 % CI, paired t-test + Wilcoxon, effect size dz — at pair level AND
     track level (complete-case: pairs where either model is NaN, e.g.
     silent stems, are dropped from that comparison).
  4. Recon-normalized *generalization* metrics: the exp3 protocol records each
     model's own reconstruction of the same stems exactly so cross-model
     comparison can be normalized to the model's own recon floor.  A model
     generalizes to held-out laws iff its interpolated-path fidelity stays
     close to its own reconstruction:
       sisdr_drop  = path SI-SDR − own recon SI-SDR   (higher = better)
       ms_excess   = path MS-STFT − own recon MS-STFT  (lower  = better)
     Interior-α variants (α ∈ {0.125 … 0.875}) remove the α=0/1 endpoints,
     which degenerate to pure reconstruction.
  5. Sample-size adequacy: given the observed gap and Δ-spread, the number of
     pairs that 95 %-power-80 % detection would need, vs. the n we have.

Usage:
    python evaluation/exp3_sampler_significance.py \\
        --results evaluation/exp3/results_sampler_cart_square.json \\
                  evaluation/exp3/results_sampler_cart_iso.json \\
                  evaluation/exp3/results_sampler_univ_warm.json \\
        --out evaluation/exp3/exp3_sampler_significance.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

HELDOUT_LAWS = ["logfade", "sqrt", "parabolic", "exp", "random"]
INDIST_LAWS = ["lerp", "trig"]

# metric -> (label, higher_is_better)
METRICS = {
    "ms": ("path MS-STFT (lower better)", False),
    "sisdr": ("path SI-SDR dB (higher better)", True),
    "norm_ms": ("MS excess over own recon (lower better)", False),
    "norm_sisdr": ("SI-SDR drop vs own recon dB (higher better)", True),
    "int_ms": ("interior-α path MS-STFT (lower better)", False),
    "int_sisdr": ("interior-α path SI-SDR dB (higher better)", True),
    "norm_int_ms": ("interior-α MS excess over own recon (lower better)", False),
    "norm_int_sisdr": ("interior-α SI-SDR drop vs own recon (higher better)", True),
}


def path_means(trial: dict) -> dict:
    ms = np.asarray(trial["mix_ms"], dtype=np.float64)
    si = np.asarray(trial["mix_sisdr"], dtype=np.float64)
    return {
        "ms": float(np.nanmean(ms)),
        "sisdr": float(np.nanmean(si)),
        "int_ms": float(np.nanmean(ms[1:-1])) if len(ms) > 2 else float(np.nanmean(ms)),
        "int_sisdr": float(np.nanmean(si[1:-1])) if len(si) > 2 else float(np.nanmean(si)),
        "recon_ms": float(trial["recon_ms"]),
        "recon_sisdr": float(trial["recon_sisdr"]),
    }


def load_model(path: Path) -> dict:
    data = json.loads(path.read_text())
    per_trial = {}  # (track, pair, law) -> path/recon metrics
    for t in data["trials"]:
        key = (t["track"], t["pair"], t["law"])
        per_trial[key] = path_means(t)
    return {
        "label": data["model"],
        "ckpt": data["ckpt"],
        "laws": data["laws"],
        "sr": data["sr"],
        "per_trial": per_trial,
        "tracks": sorted({k[0] for k in per_trial}),
        "n_pairs": len({(k[0], k[1]) for k in per_trial}),
    }


def sem(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    valid = x[~np.isnan(x)]
    n = len(valid)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if n > 1 else 0.0,
        "sem": sem(x),
    }
    out["ci95"] = 1.959963984540054 * out["sem"] if n > 1 else float("nan")
    return out


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """a − b paired over identical units (complete-case). Returns stats + effect."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    d = a - b
    n = len(d)
    if n < 2 or np.allclose(d, 0.0):
        return {"n": n, "n_dropped": int((~mask).sum()), "delta": describe(d),
                "note": "no variation"}
    t_stat, t_p = stats.ttest_rel(a, b)
    dz = float(np.mean(d) / np.std(d, ddof=1))
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    dd = describe(d)
    sd = np.std(d, ddof=1)
    gap = abs(np.mean(d))
    return {
        "n": n,
        "n_dropped": int((~mask).sum()),
        "delta": dd,
        "delta_mean": dd["mean"],
        "delta_sem": dd["sem"],
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_pvalue": float(w_p),
        "effect_dz": dz,
        # pairs needed for 95%/80% detection of the observed |gap|
        "n_pairs_needed_95_80": float(((1.959963984540054 + 0.8416212336) * sd / gap) ** 2) if gap > 0 else float("inf"),
    }


def subset_means(per_trial: dict, keys: list, laws: list, metric: str) -> np.ndarray:
    """Per-pair metric averaged over the given laws (one value per pair).

    ``norm_*`` metrics subtract the pair's own reconstruction anchor, so they
    measure the interpolation-generalization gap relative to the model's own
    recon floor."""
    base = metric[5:] if metric.startswith("norm_") else metric
    is_ms = base.endswith("ms")
    by_pair: dict = {}
    recon: dict = {}
    for k in keys:
        if k[2] not in laws:
            continue
        pk = (k[0], k[1])
        by_pair.setdefault(pk, []).append(per_trial[k][base])
        recon[pk] = per_trial[k]["recon_ms" if is_ms else "recon_sisdr"]
    out = []
    for pk, vals in by_pair.items():
        v = float(np.mean(vals))
        if metric.startswith("norm_"):
            r = recon[pk]
            v = v - r if np.isfinite(r) else float("nan")
        out.append(v)
    return np.array(out, dtype=np.float64)


def track_reduce(pair_keys: list, values: np.ndarray) -> np.ndarray:
    """Average the 6 stem-pairs of each track -> per-track means."""
    idx_by_track: dict = {}
    for i, (tr, _pr) in enumerate(pair_keys):
        idx_by_track.setdefault(tr, []).append(i)
    tracks = sorted(idx_by_track)
    return np.array([np.mean(values[idx_by_track[t]]) for t in tracks])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    models = [load_model(p) for p in args.results]
    labels = [m["label"] for m in models]
    print(f"loaded {len(models)} models: {labels}")
    print(f"sr={models[0]['sr']}  pairs/model={models[0]['n_pairs']}  "
          f"tracks={len(models[0]['tracks'])}")

    laws = models[0]["laws"]
    for m in models:
        assert m["laws"] == laws, "law sets differ across models"
    keys = sorted(models[0]["per_trial"].keys())
    for m in models[1:]:
        assert set(m["per_trial"].keys()) == set(keys), \
            f"trial sets differ for {m['label']} — comparison is NOT paired"
    pair_keys = sorted({(k[0], k[1]) for k in keys})

    summary = {
        "description": "Exp3 sampler-ablation paired significance analysis "
                       "(identical track/pair/law/alpha grid for every model)",
        "timestamp": datetime.now().isoformat(),
        "models": labels,
        "checkpoints": {m["label"]: m["ckpt"] for m in models},
        "n_pairs": models[0]["n_pairs"],
        "n_tracks": len(models[0]["tracks"]),
        "sr": models[0]["sr"],
        "laws": laws,
        "metric_meaning": {k: v[0] for k, v in METRICS.items()},
        "subsets": {"overall": laws, "heldout": HELDOUT_LAWS, "indist": INDIST_LAWS},
        "per_model": {},
        "paired": {},
    }

    # ---------------- per-model descriptive stats ----------------
    print("\n=== Per-model path metrics — mean ± std (SEM pairs | SEM tracks) ===")
    print(f"{'':<10}" + "".join(f"  [{i}] {m['label']}" for i, m in enumerate(models)))
    for law in laws + ["OVERALL", "HELDOUT", "INDIST"]:
        subset = laws if law == "OVERALL" else (
            HELDOUT_LAWS if law == "HELDOUT" else (INDIST_LAWS if law == "INDIST" else [law]))
        row = f"{law:<10}"
        for m in models:
            for metric in ("ms", "sisdr", "norm_ms", "norm_sisdr"):
                v = subset_means(m["per_trial"], keys, subset, metric)
                d, dt = describe(v), describe(track_reduce(pair_keys, v))
                row += (f"  {metric}={d['mean']:+.4f}±{d['std']:.4f}"
                        f"({d['sem']:.4f}|{dt['sem']:.4f})")
                s = summary["per_model"].setdefault(m["label"], {}).setdefault(law, {})
                s[metric] = {"pair": d, "track": dt}
        print(row)

    # ---------------- paired comparisons ----------------
    for i, j in combinations(range(len(models)), 2):
        ma, mb = models[i], models[j]
        name = f"{ma['label']}_vs_{mb['label']}"
        print(f"\n=== Paired Δ = {ma['label']} − {mb['label']} "
              f"(for 'better' metrics Δ<0 ⇒ {ma['label']} worse; p-values: pair | track) ===")
        comp = summary["paired"].setdefault(name, {"a": ma["label"], "b": mb["label"]})
        for law in laws + ["OVERALL", "HELDOUT", "INDIST"]:
            subset = laws if law == "OVERALL" else (
                HELDOUT_LAWS if law == "HELDOUT" else (INDIST_LAWS if law == "INDIST" else [law]))
            row = f"{law:<10}"
            cl = comp.setdefault(law, {})
            for metric in ("ms", "sisdr", "norm_ms", "norm_sisdr",
                           "int_ms", "int_sisdr", "norm_int_ms", "norm_int_sisdr"):
                va = subset_means(ma["per_trial"], keys, subset, metric)
                vb = subset_means(mb["per_trial"], keys, subset, metric)
                tp = paired_test(va, vb)
                tt = paired_test(track_reduce(pair_keys, va),
                                 track_reduce(pair_keys, vb))
                cl[metric] = {"pairs": tp, "tracks": tt}
                if metric in ("ms", "sisdr", "norm_ms", "norm_sisdr"):
                    row += (f"  {metric}: Δ={tp['delta_mean']:+.4f}±{tp['delta']['std']:.4f}"
                            f" p={tp['t_pvalue']:.1e}|{tt['t_pvalue']:.1e}")
            print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsignificance -> {args.out}")


if __name__ == "__main__":
    main()
