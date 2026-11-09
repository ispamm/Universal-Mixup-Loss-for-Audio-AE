#!/usr/bin/env python3
"""
Exp 3b — Cone-fidelity sweep: paired analysis + curves figure.
================================================================

Loads the per-model sweep results (exp3b_cone_sweep.py) and produces:

  * per-model curves: mean SI-SDR (± SEM at pair/stem level and track level)
    vs A (gain, cone45) and vs θ (angular);
  * PAIRED Δ curves for every model pair with 95 % CIs and paired t-tests
    (pair/stem level and clustering-robust track level) — every model saw the
    identical (track, stem/pair, grid point);
  * multi-stem additivity table (multi4 quarter / multi4 sum / multi3);
  * a 6-panel PNG figure (curves + paired differences + multi-stem bars).

Usage:
    python evaluation/exp3b_analyze.py \\
        --results evaluation/exp3b/results_conesweep_cart_square.json \\
                  evaluation/exp3b/results_conesweep_cart_iso.json \\
                  evaluation/exp3b/results_conesweep_univ_warm.json \\
        --out evaluation/exp3b/conesweep_summary.json \\
        --fig evaluation/exp3b/conesweep_figure.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from evaluation.exp3_sampler_significance import (  # noqa: E402
    describe,
    paired_test,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_model(path: Path) -> dict:
    data = json.loads(path.read_text())
    by_key: dict = {}
    for t in data["trials"]:
        if t["type"] == "gain":
            key = ("gain", t["track"], t["stem"], t["A"])
        elif t["type"] == "cone45":
            key = ("cone45", t["track"], t["pair"], t["A"])
        elif t["type"] == "angular":
            key = ("angular", t["track"], t["pair"], t["theta"])
        elif t["type"] == "multi4":
            key = ("multi4", t["track"], t["mode"])
        elif t["type"] == "multi3":
            key = ("multi3", t["track"], t["left_out"])
        else:
            raise ValueError(f"unknown trial type {t['type']}")
        by_key[key] = {"sisdr": t["sisdr"], "ms": t["ms"], "rms_ratio": t["rms_ratio"]}
    return {"label": data["model"], "ckpt": data["ckpt"], "by_key": by_key}


def unit_series(model: dict, prefix: str, grid_val, metric: str):
    """(unit_key, value) for one sweep point, averaged over the fine dimension."""
    sel = {k[1:-1]: v for k, v in model["by_key"].items()
           if k[0] == prefix and k[-1] == grid_val}
    return list(sel.keys()), np.array([sel[u][metric] for u in sel.keys()],
                                      dtype=np.float64)


def track_means(units: list, values: np.ndarray) -> np.ndarray:
    idx: dict = {}
    for i, u in enumerate(units):
        idx.setdefault(u[0], []).append(i)
    return np.array([np.mean(values[ix]) for ix in (idx[t] for t in sorted(idx))])


def sweep_stats(models: list, prefix: str, grid: list, metric: str,
                summary: dict, comparisons: list) -> dict:
    """Per-model curves + paired Δ at every grid point."""
    out = {"grid": grid, "models": {}, "paired": {}}
    for m in models:
        curve = []
        for g in grid:
            units, vals = unit_series(m, prefix, g, metric)
            d, dt = describe(vals), describe(track_means(units, vals))
            curve.append({"mean": d["mean"], "std": d["std"],
                         "sem_unit": d["sem"], "sem_track": dt["sem"], "n": d["n"]})
        out["models"][m["label"]] = curve
    for (ma, mb) in comparisons:
        rows = []
        for g in grid:
            ua, va = unit_series(ma, prefix, g, metric)
            ub, vb = unit_series(mb, prefix, g, metric)
            assert ua == ub, "unit sets differ — not paired"
            tp = paired_test(va, vb)
            tt = paired_test(track_means(ua, va), track_means(ub, vb))
            rows.append({"delta": tp["delta_mean"], "ci": tp["delta"]["ci95"],
                         "p_unit": tp["t_pvalue"], "p_track": tt["t_pvalue"]})
        out["paired"][f"{ma['label']}-vs-{mb['label']}"] = rows
    summary[prefix] = out
    return out


def multi_stats(models: list, types: list, modes: list, metric: str,
                summary: dict, comparisons: list) -> None:
    for typ in types:
        for mode in modes:
            rows = {}
            for m in models:
                units, vals = unit_series(m, typ, mode, metric)
                d, dt = describe(vals), describe(track_means(units, vals))
                rows[m["label"]] = {"mean": d["mean"], "std": d["std"],
                                    "sem_unit": d["sem"], "sem_track": dt["sem"],
                                    "n": d["n"]}
            pair = {}
            for (ma, mb) in comparisons:
                ua, va = unit_series(ma, typ, mode, metric)
                ub, vb = unit_series(mb, typ, mode, metric)
                assert ua == ub
                tp = paired_test(va, vb)
                tt = paired_test(track_means(ua, va), track_means(ub, vb))
                pair[f"{ma['label']}-vs-{mb['label']}"] = {
                    "delta": tp["delta_mean"], "ci": tp["delta"]["ci95"],
                    "p_unit": tp["t_pvalue"], "p_track": tt["t_pvalue"]}
            summary.setdefault(typ, {})[str(mode)] = {"models": rows, "paired": pair}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    args = ap.parse_args()

    models = [load_model(p) for p in args.results]
    labels = [m["label"] for m in models]
    print(f"loaded {len(models)} models: {labels}")
    comparisons = [(a, b) for a, b in combinations(models, 2)]
    cmp_names = [f"{a['label']}-vs-{b['label']}" for a, b in comparisons]

    grids = json.loads(args.results[0].read_text())["grids"]
    summary = {"description": "Exp3b cone-fidelity sweep — paired analysis",
               "timestamp": datetime.now().isoformat(),
               "models": labels, "metric": "sisdr", "grids": grids}

    gain = sweep_stats(models, "gain", grids["gain"], "sisdr", summary, comparisons)
    cone45 = sweep_stats(models, "cone45", grids["cone45"], "sisdr", summary, comparisons)
    ang = sweep_stats(models, "angular", grids["theta"], "sisdr", summary, comparisons)
    multi_stats(models, ["multi4"], ["quarter", "sum"], "sisdr", summary, comparisons)
    multi_stats(models, ["multi3"], sorted(
        {k[-1] for k in models[0]["by_key"] if k[0] == "multi3"}), "sisdr",
        summary, comparisons)

    # ---- console tables ----
    for name, sw in (("GAIN sweep (θ=0, per stem)", gain),
                     ("CONE45 sweep (θ=45°, per pair)", cone45),
                     ("ANGULAR sweep (A=1, per pair)", ang)):
        print(f"\n=== {name} — SI-SDR mean (per model) and paired Δ ===")
        hdr = f"{'point':>6}" + "".join(f"{l:>12}" for l in labels) + \
              "".join(f"{c:>34}" for c in cmp_names)
        print(hdr)
        for i, g in enumerate(sw["grid"]):
            row = f"{g:>6}"
            for l in labels:
                row += f"{sw['models'][l][i]['mean']:>12.2f}"
            for cn in cmp_names:
                r = sw["paired"][cn][i]
                row += f"  Δ={r['delta']:>+7.2f} CI±{r['ci']:.2f} p={r['p_unit']:.1e}|{r['p_track']:.1e}"
            print(row)

    print("\n=== MULTI-STEM additivity — SI-SDR mean ± SEM(track) and paired Δ ===")
    for typ in ("multi4", "multi3"):
        for mode, blk in summary[typ].items():
            line = f"{typ}/{mode:<8}"
            for l in labels:
                mm = blk["models"][l]
                line += f"  {l}={mm['mean']:>6.2f}±{mm['sem_track']:.2f}"
            for cn, pr in blk["paired"].items():
                line += f"  {cn}: Δ={pr['delta']:+.2f} p={pr['p_unit']:.1e}|{pr['p_track']:.1e}"
            print(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {args.out}")

    # ---- figure ----
    colors = {"cart_square": "#d62728", "cart_iso": "#1f77b4", "univ_warm": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("Exp 3b — Cone-fidelity sweep (600 MUSDB18 pairs, 2 s @ 16 kHz, "
                 "paired; SI-SDR dB)", fontsize=14)

    def curve_panel(ax, sw, xlabel, title, logx=True):
        for l in labels:
            ys = [p["mean"] for p in sw["models"][l]]
            es = [p["sem_track"] for p in sw["models"][l]]
            ax.plot(sw["grid"], ys, marker="o", ms=4, label=l, color=colors.get(l))
            ax.fill_between(sw["grid"], np.array(ys) - es, np.array(ys) + es,
                            alpha=0.15, color=colors.get(l))
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SI-SDR dB (↑)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    def delta_panel(ax, sw, xlabel, title, logx=True):
        for cn in cmp_names:
            ys = [r["delta"] for r in sw["paired"][cn]]
            es = [r["ci"] for r in sw["paired"][cn]]
            ax.plot(sw["grid"], ys, marker="o", ms=4, label=f"Δ {cn}")
            ax.fill_between(sw["grid"], np.array(ys) - es, np.array(ys) + es, alpha=0.15)
        ax.axhline(0, color="k", lw=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Δ SI-SDR dB (A − B)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    curve_panel(axes[0, 0], gain, "gain A (θ=0, pure recon)", "Gain sweep — curves")
    delta_panel(axes[0, 1], gain, "gain A", "Gain sweep — paired Δ")
    curve_panel(axes[0, 2], ang, "θ (deg, A=1)", "Angular sweep — curves", logx=False)
    curve_panel(axes[1, 0], cone45, "radius A (θ=45°)", "Cone45 sweep — curves")
    delta_panel(axes[1, 1], cone45, "radius A", "Cone45 sweep — paired Δ")

    # multi-stem bars
    ax = axes[1, 2]
    groups, pos, width = [], [], 0.27
    blocks = [("multi4/quarter", "4-stem ¼ (Σw=1)"),
              ("multi4/sum", "4-stem sum (A=2)"),
              ("multi3/mean", "3-stem ⅓ (mean)")]
    multi3_modes = sorted(summary["multi3"].keys())
    for gi, (blk_name, title) in enumerate(blocks):
        means, sems = [], []
        for l in labels:
            if blk_name == "multi3/mean":
                mm = np.mean([summary["multi3"][md]["models"][l]["mean"] for md in multi3_modes])
                ss = np.mean([summary["multi3"][md]["models"][l]["sem_track"] for md in multi3_modes])
            else:
                b = summary["multi4"][blk_name.split("/")[1]]["models"][l]
                mm, ss = b["mean"], b["sem_track"]
            means.append(mm)
            sems.append(ss)
        xs = np.arange(len(labels)) + (gi - 1) * width
        ax.bar(xs, means, width, yerr=sems, capsize=3, label=title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("SI-SDR dB (↑)")
    ax.set_title("Multi-stem additivity (composed, untrained)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
