#!/usr/bin/env python3
"""
Aggregate the algebraic source-separation results into an Exp.2 LaTeX table.

Reads the per-model trial JSONs produced by the workers and reports, as
mean ± std over the 400 (track, stem) trials:
  - SI-SDR for single-audio reconstruction      (recon_sisdr)
  - SI-SDR for algebraic source separation      (sep_sisdr)
  - MS-STFT MSE for single-audio reconstruction (recon_ms)
  - MS-STFT MSE for the separated track         (sep_ms)
plus the normalization anchors:
  - ΔSI-SDR = sep_sisdr - recon_sisdr   (0 dB = as good as own recon)
  - MS-STFT increment % = (sep_ms - recon_ms)/recon_ms × 100

Usage:
    python evaluation/separation_compare/aggregate_exp2.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent

MODELS = [
    ("best_univ_441", "results_univ_441.json", "\\textbf{Universal (44.1k)}"),
    ("trig_441", "results_same_441.json", "Trig mixup (44.1k)"),
    ("same-l", "results_same_l.json", "SAME-L"),
    ("lin-cae", "results_cae.json", "Lin-CAE"),
    ("dac", "results_dac.json", "DAC (RVQ)"),
]

STEMS = ["drums", "bass", "other", "vocals"]


def load(fn):
    p = D / fn
    if not p.exists():
        return None
    return json.loads(p.read_text())


def valid_trials(trials):
    """Drop trials with a silent target stem (SI-SDR undefined -> NaN).

    A silent stem (e.g. vocals in an instrumental) makes SI-SDR undefined and
    the relative MS-STFT increment meaningless (division by a ~1e-6 anchor).
    """
    out = []
    for t in trials:
        if np.isfinite(t["recon_sisdr"]) and np.isfinite(t["sep_sisdr"]):
            out.append(t)
    return out


def nan_stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v)), float(np.median(v)), int(v.size)


def fmt(m, s, prec=2):
    return f"{m:.{prec}f} $\\pm$ {s:.{prec}f}"


def main():
    rows = []
    per_stem = {}
    for mid, fn, disp in MODELS:
        d = load(fn)
        if d is None:
            print(f"  [skip] {fn} not found")
            continue
        raw = d["trials"]
        trials = valid_trials(raw)
        n_drop = len(raw) - len(trials)

        r_si_m, r_si_s, r_si_med, n = nan_stats([t["recon_sisdr"] for t in trials])
        s_si_m, s_si_s, s_si_med, _ = nan_stats([t["sep_sisdr"] for t in trials])
        r_ms_m, r_ms_s, r_ms_med, _ = nan_stats([t["recon_ms"] for t in trials])
        s_ms_m, s_ms_s, s_ms_med, _ = nan_stats([t["sep_ms"] for t in trials])
        d_si_m, d_si_s, d_si_med, _ = nan_stats(
            [t["sep_sisdr"] - t["recon_sisdr"] for t in trials])
        # MS increment: only where the recon anchor is not near-silent
        anchor = [t for t in trials if abs(t["recon_ms"]) > 1e-8]
        inc = [(t["sep_ms"] - t["recon_ms"]) / abs(t["recon_ms"]) * 100 for t in anchor]
        inc_m, inc_s, inc_med, _ = nan_stats(inc)

        rows.append({
            "id": mid, "disp": disp, "n": n, "n_dropped": n_drop,
            "recon_sisdr": (r_si_m, r_si_s, r_si_med),
            "sep_sisdr": (s_si_m, s_si_s, s_si_med),
            "recon_ms": (r_ms_m, r_ms_s, r_ms_med),
            "sep_ms": (s_ms_m, s_ms_s, s_ms_med),
            "delta_sisdr": (d_si_m, d_si_s, d_si_med),
            "ms_inc_pct": (inc_m, inc_s, inc_med),
        })

        # per-stem (median, robust)
        per_stem[disp] = {}
        for st in STEMS:
            st_trials = valid_trials([t for t in raw if t["stem"] == st])
            if not st_trials:
                continue
            ds = [t["sep_sisdr"] - t["recon_sisdr"] for t in st_trials]
            anchor_st = [t for t in st_trials if abs(t["recon_ms"]) > 1e-8]
            incs = [(t["sep_ms"] - t["recon_ms"]) / abs(t["recon_ms"]) * 100 for t in anchor_st]
            per_stem[disp][st] = {
                "n": len(st_trials),
                "delta_sisdr_median": float(np.median(ds)) if ds else float("nan"),
                "ms_inc_pct_median": float(np.median(incs)) if incs else float("nan"),
            }

    # ---- LaTeX ----
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Algebraic source separation on MUSDB18-HQ (100 tracks, "
                 r"400 (mix, stem) trials, seed 42, 44.1\,kHz). "
                 r"$z_{\mathrm{sep}} = z_{\mathrm{mix}} - \sum_{\mathrm{others}} z_{\mathrm{stem}}$, "
                 r"$\mathrm{sep} = D(z_{\mathrm{sep}})$. "
                 r"SI-SDR and MS-STFT columns are mean $\pm$ std over trials; "
                 r"$\Delta$SI-SDR and MS-inc\% are medians (they are heavy-tailed --- "
                 r"silent-stem trials are excluded and near-silent reconstruction anchors "
                 r"make per-trial ratios explode). "
                 r"Recon = the model's own reconstruction of the isolated stem (quality anchor). "
                 r"$\Delta$SI-SDR $=$ sep $-$ recon ($0$\,dB $=$ as good as the model's own "
                 r"reconstruction); MS-inc\% $=$ relative MS-STFT increase of separation vs "
                 r"own reconstruction.}")
    lines.append(r"\label{tab:exp2-separation}")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Recon SI-SDR$\uparrow$ & Sep SI-SDR$\uparrow$ & "
                 r"$\Delta$SI-SDR$\uparrow$ & Recon MS-STFT$\downarrow$ & "
                 r"Sep MS-STFT$\downarrow$ & MS-inc\%$\downarrow$ \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{r['disp']} & {fmt(*r['recon_sisdr'][:2])} & {fmt(*r['sep_sisdr'][:2])} & "
            f"{r['delta_sisdr'][2]:.2f} & {fmt(*r['recon_ms'][:2], 4)} & "
            f"{fmt(*r['sep_ms'][:2], 4)} & {r['ms_inc_pct'][2]:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out = D / "exp2_separation_table.tex"
    out.write_text(tex)
    print(tex)
    print(f"\nSaved: {out}")

    summary = {"rows": rows, "per_stem": per_stem}
    (D / "exp2_separation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {D / 'exp2_separation_summary.json'}")

    print("\nPer-stem medians (ΔSI-SDR dB / MS-inc %):")
    for disp, sts in per_stem.items():
        print(f"  {disp}:")
        for st in STEMS:
            if st in sts:
                print(f"    {st:<8} n={sts[st]['n']:<4} ΔSI-SDR={sts[st]['delta_sisdr_median']:+.2f} dB   "
                      f"MS-inc={sts[st]['ms_inc_pct_median']:+.1f}%")


if __name__ == "__main__":
    main()
