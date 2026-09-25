#!/usr/bin/env python3
"""
Aggregate the E6b seed-replication results (Exp 4, 2 s geometry).

Reads the per-seed outputs of
    eval_fm_null_separation.py --tag _2s_{rf,mse}_s{42,43,44}
and reports mean +/- std across the three DiT seeds for every headline number,
alongside the AE-determined baselines (which must be identical across seeds — a
useful sanity check that the evaluation is deterministic given the autoencoder).

Also reports each model's margin over the no-op floor, because absolute SI-SDR is
NOT comparable between the 2 s and 10 s geometries (the floor is +2.7 dB at 2 s and
-0.1 dB at 10 s).

USAGE
    python toy_evaluation/exp4_aggregate_seeds.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

OUT = Path("toy_evaluation/exp4_fm_separation")
SEEDS = [42, 43, 44]
OBJS = ["rf", "mse"]


def load(obj, seed):
    p = OUT / f"fm_null_separation_2s_{obj}_s{seed}.json"
    if not p.exists():
        print(f"[warn] missing {p}")
        return None
    return json.loads(p.read_text())


def main():
    agg = {}
    for obj in OBJS:
        js = [(s, load(obj, s)) for s in SEEDS]
        js = [(s, j) for s, j in js if j]
        if not js:
            continue
        # labels carry the seed, so read them per JSON (order is [univ, softnorm])
        A0, B0 = list(js[0][1]["results"].keys())
        rows = {}
        for tag in ("ceiling", "oracle", "noop", "seed0", "wavavg", "latavg", "bestofK"):
            vals = {"A": [], "B": [], "delta": []}
            for _, j in js:
                a_lab, b_lab = list(j["results"].keys())
                va, vb = j["results"][a_lab].get(tag), j["results"][b_lab].get(tag)
                if not va or not vb:
                    continue
                vals["A"].append(va["sisdr"]); vals["B"].append(vb["sisdr"])
                vals["delta"].append(vb["sisdr"] - va["sisdr"])
            if vals["A"]:
                rows[tag] = {
                    "univ_mean": float(np.mean(vals["A"])), "univ_std": float(np.std(vals["A"])),
                    "softnorm_mean": float(np.mean(vals["B"])),
                    "softnorm_std": float(np.std(vals["B"])),
                    "delta_mean": float(np.mean(vals["delta"])),
                    "delta_std": float(np.std(vals["delta"])),
                    "per_seed_univ": [round(v, 3) for v in vals["A"]],
                    "per_seed_softnorm": [round(v, 3) for v in vals["B"]],
                }
        # MS-STFT and paired p-values per seed
        ms, ps = {}, {}
        for tag in ("wavavg", "latavg"):
            ma, mb, pv, wins = [], [], [], []
            for _, j in js:
                a_lab, b_lab = list(j["results"].keys())
                ma.append(j["results"][a_lab][tag]["ms"])
                mb.append(j["results"][b_lab][tag]["ms"])
                t = j["paired_tests"][tag]
                pv.append(t["ms_p"]); wins.append((t["ms_win_univ"], t["ms_win_softnorm"]))
            ms[tag] = {"univ_mean": float(np.mean(ma)), "softnorm_mean": float(np.mean(mb))}
            ps[tag] = {"ms_p_per_seed": [f"{p:.1e}" for p in pv],
                       "ms_wins_per_seed": wins}
        agg[obj] = {"rows": rows, "msstft": ms, "paired": ps, "n_seeds": len(js),
                    "labels": [A0, B0]}

    # ---------------- report ----------------
    print("=" * 100)
    print("E6b — seed replication, 2 s geometry, 3 DiT seeds per configuration "
          "(n=128 val mixtures, 6 draws)")
    print("=" * 100)
    for obj, d in agg.items():
        R = d["rows"]
        name = "RF (generative model)" if obj == "rf" else "regressor (conditional mean)"
        print(f"\n### {obj.upper()} — {name}")
        print(f"{'estimator':<12}{'univ':>18}{'softnorm':>18}{'delta (soft-univ)':>20}")
        print("-" * 100)
        for tag in ("ceiling", "oracle", "noop", "seed0", "bestofK", "latavg", "wavavg"):
            if tag not in R:
                continue
            r = R[tag]
            star = " *" if tag in ("wavavg", "latavg") else ""
            print(f"{tag + star:<12}"
                  f"{r['univ_mean']:>11.2f} ± {r['univ_std']:<4.2f}"
                  f"{r['softnorm_mean']:>11.2f} ± {r['softnorm_std']:<4.2f}"
                  f"{r['delta_mean']:>+13.2f} ± {r['delta_std']:<4.2f}")
        print("-" * 100)
        for tag in ("wavavg", "latavg"):
            if tag in R:
                f_ = R["noop"]["univ_mean"] if tag != "latavg" else R["noop"]["univ_mean"]
                print(f"  margin over floor ({tag}): univ "
                      f"{R[tag]['univ_mean'] - R['noop']['univ_mean']:+.2f} dB | softnorm "
                      f"{R[tag]['softnorm_mean'] - R['noop']['softnorm_mean']:+.2f} dB")
            if tag in d["msstft"]:
                m = d["msstft"][tag]
                print(f"  MS-STFT {tag}: univ {m['univ_mean']:.4f} vs softnorm "
                      f"{m['softnorm_mean']:.4f}  (p per seed: "
                      f"{', '.join(d['paired'][tag]['ms_p_per_seed'])}; "
                      f"wins {d['paired'][tag]['ms_wins_per_seed']})")
        print(f"  baseline identity across seeds: ceiling "
              f"{[round(v,2) for v in R['ceiling']['per_seed_univ']]} / "
              f"{[round(v,2) for v in R['ceiling']['per_seed_softnorm']]}  "
              f"oracle {[round(v,2) for v in R['oracle']['per_seed_univ']]} / "
              f"{[round(v,2) for v in R['oracle']['per_seed_softnorm']]}")

    (OUT / "seed_aggregation_2s.json").write_text(json.dumps(agg, indent=2))
    print(f"\nSaved: {OUT / 'seed_aggregation_2s.json'}")


if __name__ == "__main__":
    main()
