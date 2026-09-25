#!/usr/bin/env python3
"""
Aggregate the trig-interpolation energy-preservation battery (4 models).

Models run at different sample rates / have very different absolute quality, so
everything is anchored to each model's own reconstruction of the SAME two stems
(identical clips), mirroring ``aggregate.py``:

  * energy:  energy_ratio(α) = RMS(D(trigMix z))/RMS(ideal trig mix)
             - relative energy error % = |ratio − 1| × 100 per (trial, α)
               (0 % = the decoded mix carries exactly the ideal constant-power
               energy at that α),
             - recon RMS err % = mean |RMS(recon)/RMS(orig) − 1| per stem —
               the model's own single-clip energy fidelity (the floor),
             - path flatness = std over α of the mean energy-ratio curve,
             - gap (lims convention) = mean |dec_norm − ideal_norm|.
  * mix fidelity (same 0-anchors as the separation study):
             - ΔSI-SDR = mix_sisdr − recon_sisdr dB (0 = mixing as good as
               own reconstruction),
             - MS-STFT inc % = (mix_ms − recon_ms)/recon_ms × 100
               (0 % = mixing error equals own reconstruction error).

"Interior" = α ∈ [0.1, 0.9] (the actual mixing path; endpoints α ∈ {0, 1} are
single-stem reconstruction identity checks).

Usage:
    python toy_evaluation/separation_compare/aggregate_energy.py \
        --results toy_evaluation/separation_compare/results_energy_same.json \
                  toy_evaluation/separation_compare/results_energy_same_l.json \
                  toy_evaluation/separation_compare/results_energy_cae.json \
                  toy_evaluation/separation_compare/results_energy_dac.json \
        --out toy_evaluation/separation_compare/energy_interp_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

PAIR_KEYS = ["drums/bass", "drums/other", "drums/vocals",
             "bass/other", "bass/vocals", "other/vocals"]


def _arr(trial: dict, key: str) -> np.ndarray:
    return np.asarray(trial[key], dtype=np.float64)


def interior_mask(alphas) -> np.ndarray:
    return (np.asarray(alphas) >= 0.1) & (np.asarray(alphas) <= 0.9)


def trial_scalars(trial: dict, mask: np.ndarray) -> dict:
    """Per-trial scalars (mean over the selected α window)."""
    er = _arr(trial, "energy_ratio")
    dsisdr = _arr(trial, "mix_sisdr") - trial["recon_sisdr"]
    recon_ms = trial["recon_ms"]
    ms_inc = (_arr(trial, "mix_ms") - recon_ms) / recon_ms * 100.0 if recon_ms > 0 else np.full_like(er, np.nan)
    dec_norm = _arr(trial, "dec_norm")
    ideal_norm = _arr(trial, "ideal_norm")
    return {
        "energy_err_all": float(np.nanmean(np.abs(er - 1.0))),           # all α
        "energy_err_int": float(np.nanmean(np.abs(er[mask] - 1.0))),     # interior α
        "energy_gap": float(np.nanmean(np.abs(dec_norm - ideal_norm))),  # lims gap, all α
        "dsisdr_all": float(np.nanmean(dsisdr)),
        "dsisdr_int": float(np.nanmean(dsisdr[mask])),
        "ms_inc_all": float(np.nanmean(ms_inc)),
        "ms_inc_int": float(np.nanmean(ms_inc[mask])),
    }


def model_stats(trials: list[dict], alphas: list[float]) -> dict:
    mask = interior_mask(alphas)
    n = len(trials)

    er = np.stack([_arr(t, "energy_ratio") for t in trials])          # (n, nα)
    dec_norm = np.stack([_arr(t, "dec_norm") for t in trials])
    ideal_norm = np.stack([_arr(t, "ideal_norm") for t in trials])
    mix_sisdr = np.stack([_arr(t, "mix_sisdr") for t in trials])
    mix_ms = np.stack([_arr(t, "mix_ms") for t in trials])
    recon_sisdr = np.array([t["recon_sisdr"] for t in trials])
    recon_ms = np.array([t["recon_ms"] for t in trials])
    recon_rms_err = np.array([t["recon_rms_err"] for t in trials])

    dsisdr = mix_sisdr - recon_sisdr[:, None]
    ms_inc = np.where(recon_ms[:, None] > 0,
                      (mix_ms - recon_ms[:, None]) / recon_ms[:, None] * 100.0,
                      np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        # mean and median ratio curves (median is robust to near-silent outliers)
        ratio_curve_mean = np.nanmean(er, axis=0)
        ratio_curve_std = np.nanstd(er, axis=0)
        ratio_curve_median = np.nanmedian(er, axis=0)
        dsisdr_curve = np.nanmean(dsisdr, axis=0)
        ms_inc_curve = np.nanmean(ms_inc, axis=0)
        dec_norm_curve = np.nanmean(dec_norm, axis=0)
        ideal_norm_curve = np.nanmean(ideal_norm, axis=0)

        err_all = np.abs(er - 1.0)
        err_int = np.abs(er[:, mask] - 1.0)
        gap_all = np.abs(dec_norm - ideal_norm)
        flatness_mean = float(np.nanstd(ratio_curve_mean))      # std of mean ratio curve
        flatness_median = float(np.nanstd(ratio_curve_median))  # std of median ratio curve
        mid = int(np.argmin(np.abs(np.asarray(alphas) - 0.5)))

    return {
        "n_trials": n,
        "alphas": alphas,
        # reconstruction anchors (the floor)
        "recon_sisdr_mean": float(np.nanmean(recon_sisdr)),
        "recon_ms_mean": float(np.nanmean(recon_ms)),
        "recon_rms_err_mean": float(np.nanmean(recon_rms_err)),
        "recon_rms_err_median": float(np.nanmedian(recon_rms_err)),
        # energy preservation
        "energy_ratio_curve_mean": ratio_curve_mean.tolist(),
        "energy_ratio_curve_median": ratio_curve_median.tolist(),
        "energy_ratio_curve_std": ratio_curve_std.tolist(),
        "energy_err_pct_all": float(np.nanmean(err_all) * 100.0),
        "energy_err_pct_all_median": float(np.nanmedian(err_all) * 100.0),
        "energy_err_pct_interior": float(np.nanmean(err_int) * 100.0),
        "energy_err_pct_interior_median": float(np.nanmedian(err_int) * 100.0),
        "energy_ratio_at_midpoint": float(ratio_curve_mean[mid]),
        "energy_ratio_at_midpoint_median": float(ratio_curve_median[mid]),
        "energy_path_flatness": flatness_mean,
        "energy_path_flatness_median": flatness_median,
        "energy_gap_mean": float(np.nanmean(gap_all)),
        "dec_norm_curve_mean": dec_norm_curve.tolist(),
        "ideal_norm_curve_mean": ideal_norm_curve.tolist(),
        # mix fidelity (anchored to own reconstruction)
        "dsisdr_db_mean": float(np.nanmean(dsisdr)),
        "dsisdr_db_median": float(np.nanmedian(dsisdr)),
        "dsisdr_db_interior_mean": float(np.nanmean(dsisdr[:, mask])),
        "dsisdr_db_interior_median": float(np.nanmedian(dsisdr[:, mask])),
        "dsisdr_curve_mean": dsisdr_curve.tolist(),
        "ms_inc_pct_mean": float(np.nanmean(ms_inc)),
        "ms_inc_pct_median": float(np.nanmedian(ms_inc)),
        "ms_inc_pct_interior_mean": float(np.nanmean(ms_inc[:, mask])),
        "ms_inc_pct_interior_median": float(np.nanmedian(ms_inc[:, mask])),
        "ms_inc_curve_mean": ms_inc_curve.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path,
                    default=[Path(__file__).resolve().parent / f"results_energy_{m}.json"
                             for m in ("same", "same_l", "cae", "dac")])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "energy_interp_summary.json")
    args = ap.parse_args()

    results = {}
    for path in args.results:
        data = json.loads(path.read_text())
        results[data["model"]] = data
    order = list(results)
    print(f"loaded {len(results)} models: {order}")

    summary = {
        "description": "Trigonometric (constant-power) interpolation energy preservation "
                       "on MUSDB18-HQ stems — 4 autoencoders, full α ∈ [0,1] path, "
                       "anchored to each model's own reconstruction of the same stems.",
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "interp": "z_mix = cos(α·π/2)·z_A + sin(α·π/2)·z_B; ideal = cos(α·π/2)·A + sin(α·π/2)·B",
            "pairs": "all 6 same-track stem pairs (drums/bass/other/vocals), 100 MUSDB tracks (seed 42)",
            "energy_anchor": "energy_ratio = RMS(D(z_mix))/RMS(ideal) (1.0 = perfect constant-power preservation)",
            "recon_anchor": "recon RMS err % = mean |RMS(recon)/RMS(orig) − 1| per stem (the model's single-clip energy floor)",
            "mix_fidelity_anchor": "ΔSI-SDR = mix_sisdr − recon_sisdr (0 dB = as good as own recon); "
                                   "MS inc % = (mix_ms − recon_ms)/recon_ms × 100 (0 % = as good as own recon)",
            "interior": "α ∈ [0.1, 0.9] excludes the reconstruction-identity endpoints",
        },
        "models": {},
        "per_pair_type": {},
        "paired": {},
    }

    # ---- per-model stats ----
    print("\n=== Energy preservation (full trig path; medians are robust to near-silent outliers) ===")
    print(f"{'model':<9} {'reconSISDR':>9} {'reconRMS%':>9} {'Erre%allM':>9} {'Erre%intM':>9} {'Erre%intX':>9} "
          f"{'flatM':>7} {'ratio@0.5M':>9} {'dSISDR med':>10} {'MSinc% med':>10}")
    for name in order:
        s = model_stats(results[name]["trials"], results[name]["alphas"])
        summary["models"][name] = s
        print(f"{name:<9} {s['recon_sisdr_mean']:>9.2f} {s['recon_rms_err_median']*100:>9.2f} "
              f"{s['energy_err_pct_all_median']:>9.2f} {s['energy_err_pct_interior_median']:>9.2f} "
              f"{s['energy_err_pct_interior']:>9.2f} "
              f"{s['energy_path_flatness_median']:>7.4f} {s['energy_ratio_at_midpoint_median']:>9.4f} "
              f"{s['dsisdr_db_median']:>10.2f} {s['ms_inc_pct_median']:>10.1f}")

    # ---- energy-ratio curve table (median | mean) ----
    alphas = results[order[0]]["alphas"]
    n_a = min(len(alphas), *(len(summary["models"][n]["energy_ratio_curve_median"]) for n in order))
    print("\nEnergy ratio by α  (median | mean over 600 trials; 1.000 = ideal constant power):")
    header = "  α".ljust(6) + "".join(f"{name:>22}" for name in order)
    print(header)
    for ai in range(n_a):
        a = alphas[ai]
        row = f"{a:>5.2f}  "
        for name in order:
            md = summary["models"][name]["energy_ratio_curve_median"][ai]
            mn = summary["models"][name]["energy_ratio_curve_mean"][ai]
            row += f"{md:>10.4f} | {mn:>10.4f}"
        print(row)

    # ---- per-pair-type (e.g. vocals-containing vs rhythmic) breakdown, medians ----
    print("\nPer pair type (interior energy error %, median):")
    print(f"{'pair':<14} " + "".join(f"{name:>12}" for name in order))
    for pk in PAIR_KEYS:
        summary["per_pair_type"][pk] = {}
        row = f"{pk:<14} "
        for name in order:
            sel = [t for t in results[name]["trials"] if t["pair"] == pk]
            s = model_stats(sel, alphas)
            summary["per_pair_type"][pk][name] = s["energy_err_pct_interior_median"]
            row += f"{s['energy_err_pct_interior_median']:>12.2f}"
        print(row)

    # ---- paired comparisons on aligned (track, pair) trials, interior window ----
    names = order
    print("\nPaired (same track+pair; interior α mean scalars; 0 % / 0 dB = as good as own recon):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ta = {t["track"] + "/" + t["pair"]: t for t in results[a]["trials"]}
            tb = {t["track"] + "/" + t["pair"]: t for t in results[b]["trials"]}
            keys = [k for k in ta if k in tb]
            mask = interior_mask(alphas)
            ea = np.array([trial_scalars(ta[k], mask)["energy_err_int"] for k in keys])
            eb = np.array([trial_scalars(tb[k], mask)["energy_err_int"] for k in keys])
            da = np.array([trial_scalars(ta[k], mask)["dsisdr_int"] for k in keys])
            db = np.array([trial_scalars(tb[k], mask)["dsisdr_int"] for k in keys])
            ma = np.array([trial_scalars(ta[k], mask)["ms_inc_int"] for k in keys])
            mb = np.array([trial_scalars(tb[k], mask)["ms_inc_int"] for k in keys])

            paired = {
                "model_a": a, "model_b": b, "n": len(keys),
                "mean_energy_err_pct_a": float(np.nanmean(ea) * 100),
                "mean_energy_err_pct_b": float(np.nanmean(eb) * 100),
                "median_energy_err_pct_a": float(np.nanmedian(ea) * 100),
                "median_energy_err_pct_b": float(np.nanmedian(eb) * 100),
                "frac_a_lower_energy_err": float(np.nanmean(ea < eb)),
                "mean_dsisdr_a": float(np.nanmean(da)),
                "mean_dsisdr_b": float(np.nanmean(db)),
                "frac_a_better_dsisdr": float(np.nanmean(da > db)),
                "mean_ms_inc_pct_a": float(np.nanmean(ma)),
                "mean_ms_inc_pct_b": float(np.nanmean(mb)),
                "frac_a_lower_ms_inc": float(np.nanmean(ma < mb)),
            }
            summary["paired"][f"{a}_vs_{b}"] = paired
            print(f"  {a} vs {b}: energy err% {paired['median_energy_err_pct_a']:.2f} vs "
                  f"{paired['median_energy_err_pct_b']:.2f} (frac A lower: {paired['frac_a_lower_energy_err']:.3f}) | "
                  f"ΔSI-SDR {paired['mean_dsisdr_a']:.2f} vs {paired['mean_dsisdr_b']:.2f} dB "
                  f"(frac A better: {paired['frac_a_better_dsisdr']:.3f})")

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
