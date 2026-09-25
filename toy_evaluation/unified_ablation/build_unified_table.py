#!/usr/bin/env python3
"""
Build the unified ICASSP ablation table (mean ± std error of the mean) as LaTeX.

Reads:
  * toy_evaluation/unified_ablation/unified_data_from_artifacts.json
  * toy_evaluation/unified_ablation/norm_pres_supplement.json  (Norm Err per-pair std)

Cell conventions (documented in the .tex header):
  mean ± sem = standard error of the reported mean
      sem = (sample std across the fixed seed-42 evaluation set) / sqrt(n),
      with n = 1000 pairs (2000 clips for MS-STFT). This measures the
      uncertainty of the reported mean, not the dispersion of the per-item
      metric (that dispersion is stored as 'std' in the data JSON).
  Flatness (dec/enc), Eff. Rank and Convexity are curve/pool statistics
  reported as single values.
  Bold = column best, underline = second best (unique at display precision).
  `*` markers are preserved verbatim from the previous draft (author legend).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DATA = json.loads((_HERE / "unified_data_from_artifacts.json").read_text())
_SUPP = json.loads((_HERE / "norm_pres_supplement.json").read_text())

# order of rows as in the paper table
ROWS = [
    ("Minimal", "transformer_minimal_full_best"),
    ("Soft-Norm", "transformer_softMax_full_best"),
    ("Full", "transformer_norm_full_best"),
    ("Norm-Preserving", "transformer_best_full_best"),
    ("Implicit", "transformer_implicit_full_v2_best"),
    ("Mixup+Enc", "transformer_mixup_enc_best"),
]


def get(model: str) -> dict:
    m = dict(_DATA["models"][model])
    supp = _SUPP["models"][model]["norm_err_05"]
    m["norm_err_05"]["std"] = supp["std"]
    m["norm_err_05"]["n"] = supp["n"]
    # sem = sample std / sqrt(n)
    m["norm_err_05"]["sem"] = supp["std"] / np.sqrt(supp["n"])
    return m


def fmt_pair(mean: float, sem: float, nd: int = 3, signed: bool = False) -> str:
    if signed:
        return f"{mean:+.{nd}f} \\pm {sem:.{nd}f}"
    return f"{mean:.{nd}f} \\pm {sem:.{nd}f}"


# ---- assemble rows ----
rows = {}
for label, model in ROWS:
    m = get(model)
    d = m["delta_sisdr_trig"]; l = m["delta_sisdr_lerp"]
    rows[label] = {
        "dTrig": d, "dLerp": l,
        "rms05": m["dec_rms_05"],
        "flat": {"mean": m["dec_energy_flatness"]},
        "eTrig": m["enc_trig_mse_05"], "eTrigFlat": {"mean": m["enc_trig_flat"]},
        "eLerp": m["enc_lerp_mse_05"], "eLerpFlat": {"mean": m["enc_lerp_flat"]},
        "msstft": m["recon_msstft"],
        "effrank": {"mean": m["eff_rank"]},
        "conv": {"mean": m["convexity"]},
        "normerr": m["norm_err_05"],
    }

# ---- per-column formatting / ranking ----
# (key, nd, signed, direction, ideal)
COLS = [
    ("dTrig", 2, True, "max", None),
    ("dLerp", 2, True, "max", None),
    ("rms05", 3, False, "ideal1", 1.0),
    ("flat", 3, False, "min", None),
    ("eTrig", 3, False, "min", None),
    ("eTrigFlat", 3, False, "min", None),
    ("eLerp", 3, False, "min", None),
    ("eLerpFlat", 3, False, "min", None),
    ("msstft", 3, False, "min", None),
    ("effrank", 1, False, "max", None),
    ("conv", 1, False, "min", None),
    ("normerr", 2, False, "min", None),
]

HEADERS = [
    "\\textbf{$\\Delta$Trig (dB) $\\uparrow$}", "\\textbf{$\\Delta$LERP (dB) $\\uparrow$}",
    "\\textbf{RMS@0.5}", "\\textbf{Flatness $\\downarrow$}",
    "\\textbf{Trig@0.5}", "\\textbf{Trig Flat. $\\downarrow$}",
    "\\textbf{LERP@0.5}", "\\textbf{LERP Flat. $\\downarrow$}",
    "\\textbf{MS-STFT $\\downarrow$}", "\\textbf{Eff. Rank $\\uparrow$}",
    "\\textbf{Convexity $\\downarrow$}", "\\textbf{Norm Err $\\downarrow$}",
]


def rank_key(cell: dict, direction: str, ideal=None) -> float:
    v = cell["mean"]
    if direction == "min":
        return float(v)
    if direction == "max":
        return -float(v)
    return abs(float(v) - ideal)


# decide markers: bold best; underline unique second (skip in rms/flat cols where
# the trailing group is statistically clustered — matching the previous draft)
markers: dict = {c[0]: {} for c in COLS}
for col, nd, signed, direction, ideal in COLS:
    vals = sorted([lbl for lbl, _ in ROWS], key=lambda lbl: rank_key(rows[lbl][col], direction, ideal))
    best = vals[0]
    markers[col][best] = "best"
    if col in ("rms05", "flat"):
        continue  # previous draft left these without a second-best marker
    second = vals[1]
    # underline only if unique at display precision
    v1, v2 = rows[best][col]["mean"], rows[second][col]["mean"]
    third = vals[2]
    v3 = rows[third][col]["mean"]
    if round(float(v1), nd) == round(float(v2), nd):
        continue
    if round(float(v2), nd) == round(float(v3), nd):
        continue
    markers[col][second] = "second"

# `*` cells preserved from the previous draft (author legend unknown here)
STAR_CELLS = {("Norm-Preserving", "dLerp"), ("Norm-Preserving", "eTrigFlat"),
              ("Mixup+Enc", "eLerpFlat")}


def cell_tex(label: str, col: str, nd: int, signed: bool) -> str:
    c = rows[label][col]
    mean, sem = c.get("mean"), c.get("sem")
    if sem is None:
        body = f"{mean:+.2f}" if signed else f"{mean:.{nd}f}"
    else:
        body = fmt_pair(float(mean), float(sem), nd=nd, signed=signed)
    mk = markers[col].get(label)
    if mk == "best":
        body = f"$\\mathbf{{{body}}}$"
    elif mk == "second":
        body = f"$\\underline{{{body}}}$"
    else:
        body = f"${body}$"
    if (label, col) in STAR_CELLS:
        body += "$^\\ast$"
    return body


def main() -> None:
    lines = []
    A = lines.append
    A("% =============================================================================")
    A("% Unified ablation table (ICASSP) — 16 kHz toy models, 1000 validation pairs (seed 42).")
    A("%")
    A("% Values: mean ± sem, where sem = sample std / sqrt(n) over the fixed seed-42")
    A("% evaluation set (n = 1000 pairs; 2000 clips for MS-STFT) — i.e. the standard")
    A("% error, measuring the uncertainty of the reported mean. The per-item sample")
    A("% dispersion (which is much larger) is stored in")
    A("% toy_evaluation/unified_ablation/unified_data_from_artifacts.json ('std').")
    A("% Columns that are themselves curve or pool statistics (Decoder/Encoder")
    A("% Flatness, Eff. Rank, Convexity) are reported as single values. `*` markers")
    A("% preserved from the previous draft.")
    A("%")
    A("% Checkpoint mapping:")
    A("%   Minimal         = toy_training/checkpoints/transformer_minimal_full_best.pt")
    A("%   Soft-Norm       = toy_training/checkpoints/transformer_softMax_full_best.pt")
    A("%   Full            = toy_training/checkpoints/transformer_norm_full_best.pt")
    A("%   Norm-Preserving = toy_training/checkpoints/transformer_best_full_best.pt")
    A("%   Implicit        = toy_training/checkpoints/transformer_implicit_full_v2_best.pt")
    A("%   Mixup+Enc       = toy_training/checkpoints/transformer_mixup_enc_best.pt")
    A("%")
    A("% Sources: toy_evaluation/mix_recon_ratio_summary{,.bak_20260901_141340,")
    A("%   _implicit_v2,_mixup_enc_full}.json (ΔSI-SDR, per-pair) and")
    A("%   toy_evaluation/eval_all{,_implicit_v2,_mixup_enc_full}/eval_all_summary.json")
    A("%   (battery), plus toy_evaluation/unified_ablation/norm_pres_supplement.json")
    A("%   for the Norm-Err per-pair std.")
    A("%   MS-STFT column: extended 2000-clip reconstruction run (seed 42) —")
    A("%   toy_evaluation/unified_ablation/recon_2000/*.json (per-clip msstft stored).")
    A("\\begin{table*}[t]")
    A("\\centering")
    A("\\caption{Comprehensive ablation study on 16\\,kHz models evaluated on the same 1000 validation pairs (seed 42). ")
    A("We assess the three components of the framework: (i) \\textbf{Decoder mixing fidelity}, reported as relative mixing accuracy ")
    A("($\\Delta$SI-SDR relative to clean reconstruction; values near $0$\\,dB denote parity) and acoustic constant-power preservation ")
    A("(RMS at midpoint $\\alpha=0.5$ and trajectory flatness); (ii) \\textbf{Encoder homomorphism error}, defined as ")
    A("$\\text{MSE}(E(\\text{mix}(A, B)), \\text{mix}(E(A), E(B)))$ at $\\alpha=0.5$ under trigonometric and linear mixing laws, along with error variance (Flatness); ")
    A("and (iii) \\textbf{Latent space geometry and capacity}, measured via clean reconstruction loss (MS-STFT), participation-ratio effective rank, ")
    A("manifold convexity (NN-distance), and latent scale preservation (Norm Err). Values are reported as mean $\\pm$ standard error of the mean (sample std across the fixed evaluation set divided by $\\sqrt{n}$, $n=1000$ pairs, 2000 clips for MS-STFT); the curve-level and pooled statistics (Flatness, Eff. Rank, Convexity) are reported as single values. ")
    A("The best result is in \\textbf{bold}, and the second-best is \\underline{underlined}.")
    A("}")
    A("\\label{tab:unified_ablation_icassp}")
    A("\\resizebox{\\textwidth}{!}{%")
    A("\\begin{tabular}{l | c c c c | c c c c | c c c c}")
    A("\\hline")
    A(" & \\multicolumn{4}{c|}{\\textbf{Decoder Mixing \\& Energy}} ")
    A(" & \\multicolumn{4}{c|}{\\textbf{Encoder Homomorphism (MSE $\\downarrow$)}} ")
    A(" & \\multicolumn{4}{c}{\\textbf{Latent Geometry \\& Capacity}} \\\\")
    A("\\textbf{Configuration} & " + " & ".join(HEADERS) + " \\\\")
    A("\\hline")
    for label, _ in ROWS:
        cells = [cell_tex(label, col, nd, signed) for col, nd, signed, _, _ in COLS]
        A(f"{label:<16s} & " + " & ".join(cells) + " \\\\")
    A("\\hline")
    A("\\end{tabular}%")
    A("}")
    A("\\end{table*}")
    out = "\n".join(lines) + "\n"
    dest = _HERE.parents[1] / "paper_tables" / "unified_ablation_icassp.tex"
    dest.write_text(out)
    print(f"wrote {dest}")

    # also a quick ASCII preview
    print()
    hdr = f"{'config':14s} " + " ".join(f"{k:>11s}" for k, *_ in COLS)
    print(hdr)
    for label, _ in ROWS:
        cells = []
        for col, nd, signed, _, _ in COLS:
            c = rows[label][col]
            if c.get("sem") is None:
                cells.append(f"{c['mean']:.{nd}f}")
            else:
                cells.append(f"{c['mean']:.{nd}f}±{c['sem']:.{nd}f}")
        print(f"{label:14s} " + " ".join(f"{x:>11s}" for x in cells))


if __name__ == "__main__":
    main()
