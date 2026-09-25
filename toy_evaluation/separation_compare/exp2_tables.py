#!/usr/bin/env python3
"""
Exp.2 tables: overall + PER-STEM separation and reconstruction, mean +- SEM.

Produces two full-width tables in the requested layout:
  * exp2_separation_stems.tex  -- separation fidelity (sep SI-SDR, sep MS-STFT)
  * exp2_reconstruction_stems.tex -- single-audio reconstruction (recon SI-SDR,
    recon MS-STFT)

Rows = models, columns = {Overall, Drums, Bass, Other, Vocals} x {SI-SDR, MS-STFT}.

Trial set: the trials where SI-SDR is defined (silent target stems excluded),
so every cell in both tables is computed on the SAME trials. Per-stem n is
reported. Values are mean +- SEM (std / sqrt(n)).

Usage:  python toy_evaluation/separation_compare/exp2_tables.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent

MODELS = [
    ("Universal (Ours)", "results_univ_441.json"),
    ("Trig mixup", "results_same_441.json"),
    ("SAME-L", "results_same_l.json"),
    ("Lin-CAE", "results_cae.json"),
    ("DAC", "results_dac.json"),
]
STEMS = ["drums", "bass", "other", "vocals"]


def load(fn):
    p = D / fn
    return json.loads(p.read_text()) if p.exists() else None


def valid(trials):
    """Keep trials where SI-SDR is defined, so both metrics share one trial set."""
    return [t for t in trials
            if np.isfinite(t.get("recon_sisdr", np.nan))
            and np.isfinite(t.get("sep_sisdr", np.nan))]


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), len(v)


def cell(mean, sem, prec):
    return f"${mean:.{prec}f} \\pm {sem:.{prec}f}$"


def build_table(rows, metric_pair, caption, label, prec_pair=(2, 4)):
    """rows: list of (display, dict) ; metric_pair: (sisdr_key, ms_key)."""
    ks, km = metric_pair
    ps, pm = prec_pair
    n_groups = 1 + len(STEMS)

    # ---- collect values and rank per column for bold/underline ----
    table = {}
    for disp, r in rows:
        table[disp] = {}
        for grp in ["overall"] + STEMS:
            g = r[grp]
            table[disp][grp] = (g[ks], g[km])

    best = {}
    for grp in ["overall"] + STEMS:
        for j, key in enumerate([ks, km]):
            vals = {d: table[d][grp][j][0] for d, _ in rows}
            order = sorted(vals, key=lambda d: -vals[d] if key == ks else vals[d])
            best[(grp, key)] = (order[0], order[1] if len(order) > 1 else None)

    def fmt_cell(disp, grp, key, prec):
        mean, sem, _ = table[disp][grp][0 if key == ks else 1]
        txt = cell(mean, sem, prec)
        b1, b2 = best[(grp, key)]
        if disp == b1:
            return r"\textbf{" + txt + "}"
        if disp == b2:
            return r"\underline{" + txt + "}"
        return txt

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l | c c | c c | c c | c c | c c}")
    lines.append(r"\hline")
    groups = ["Overall"] + [s.capitalize() for s in STEMS]
    hdr1 = [""]
    for gi, g in enumerate(groups):
        span = r"\multicolumn{2}{c|}" if gi < len(groups) - 1 else r"\multicolumn{2}{c}"
        hdr1.append(span + r"{\textbf{" + g + r"}}")
    lines.append(" & ".join(hdr1) + r" \\")
    hdr2 = [r"\textbf{Model}"]
    for _ in range(n_groups):
        hdr2.append(r"\textbf{SI-SDR $\uparrow$} & \textbf{MS-STFT $\downarrow$}")
    lines.append(" & ".join(hdr2) + r" \\")
    lines.append(r"\hline")
    for disp, _ in rows:
        cells = [disp]
        for grp in ["overall"] + STEMS:
            cells.append(fmt_cell(disp, grp, ks, ps))
            cells.append(fmt_cell(disp, grp, km, pm))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    # per-stem trial count row (same for every model -- the shared trial set)
    n_cells = [r"$n$ (trials)"]
    for grp in ["overall"] + STEMS:
        n_cells.append(f"{rows[0][1][grp]['n']}")
        n_cells.append("")
    lines.append(" & ".join(n_cells) + r" \\")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    data = {}
    for disp, fn in MODELS:
        d = load(fn)
        if d is None:
            print(f"  [skip] {fn}")
            continue
        tr = valid(d["trials"])
        groups = {"overall": tr}
        for st in STEMS:
            groups[st] = [t for t in tr if t["stem"] == st]
        entry = {}
        for grp, ts in groups.items():
            entry[grp] = {
                "sep_sisdr": ms([t["sep_sisdr"] for t in ts]),
                "sep_ms": ms([t["sep_ms"] for t in ts]),
                "recon_sisdr": ms([t["recon_sisdr"] for t in ts]),
                "recon_ms": ms([t["recon_ms"] for t in ts]),
                "n": len(ts),
            }
        data[disp] = entry

    rows = [(d, data[d]) for d, _ in MODELS if d in data]

    # ---------------- console summary ----------------
    print(f"\n{'='*104}")
    print("Per-stem means +- SEM  (sep = algebraic separation, recon = own reconstruction)")
    print(f"{'='*104}")
    for disp, r in rows:
        print(f"\n{disp}:")
        print(f"  {'group':<9}{'sep SI-SDR':>16}{'sep MS-STFT':>16}"
              f"{'recon SI-SDR':>16}{'recon MS-STFT':>16}{'n':>6}")
        for grp in ["overall"] + STEMS:
            g = r[grp]
            print(f"  {grp:<9}{g['sep_sisdr'][0]:>10.2f}+-{g['sep_sisdr'][1]:<4.2f}"
                  f"{g['sep_ms'][0]:>10.4f}+-{g['sep_ms'][1]:<4.4f}"
                  f"{g['recon_sisdr'][0]:>10.2f}+-{g['recon_sisdr'][1]:<4.2f}"
                  f"{g['recon_ms'][0]:>10.4f}+-{g['recon_ms'][1]:<4.4f}{g['n']:>6}")

    # ---------------- LaTeX ----------------
    sep = build_table(
        rows, ("sep_sisdr", "sep_ms"),
        caption=(r"Zero-shot algebraic source separation "
                 r"($z_{\mathrm{target}} = z_{\mathrm{mix}} - \sum_{k\neq\mathrm{target}} z_k$) "
                 r"on 100 held-out MUSDB18-HQ multi-tracks at 44.1\,kHz (seed 42). "
                 r"Fidelity of the separated stem against the ground-truth stem: SI-SDR (dB, higher better) "
                 r"and multi-scale STFT MSE (lower better), overall and per stem. "
                 r"Values are mean $\pm$ SEM over the trials where SI-SDR is defined "
                 r"(silent target stems excluded); per-stem $n$ is given in "
                 r"Table~\ref{tab:musdb_recon_slim}. Best in \textbf{bold}, second best \underline{underlined}."),
        label="tab:musdb_separation_slim", prec_pair=(2, 4))
    rec = build_table(
        rows, ("recon_sisdr", "recon_ms"),
        caption=(r"Single-audio reconstruction on the same MUSDB18-HQ stems and the same "
                 r"trial set as Table~\ref{tab:musdb_separation_slim}: each model reconstructs "
                 r"the isolated stem, $D(E(x))$. This is the quality anchor for the "
                 r"separation numbers: a model can only separate as well as it can code. "
                 r"Values are mean $\pm$ SEM; per-stem $n$ in the last row. "
                 r"Best in \textbf{bold}, second best \underline{underlined}."),
        label="tab:musdb_recon_slim", prec_pair=(2, 4))

    (D / "exp2_separation_stems.tex").write_text(sep + "\n")
    (D / "exp2_reconstruction_stems.tex").write_text(rec + "\n")
    print(f"\nWrote: {D/'exp2_separation_stems.tex'}")
    print(f"Wrote: {D/'exp2_reconstruction_stems.tex'}")

    # machine-readable
    (D / "exp2_stems_summary.json").write_text(json.dumps(data, indent=2))
    print(f"Wrote: {D/'exp2_stems_summary.json'}")


if __name__ == "__main__":
    main()
