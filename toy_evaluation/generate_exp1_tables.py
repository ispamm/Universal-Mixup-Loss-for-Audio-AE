#!/usr/bin/env python3
"""
Generate LaTeX tables from the ICLR Exp.1 evaluation battery results.

Reads the eval_all_summary.json produced by evaluate_all.py and produces:
  - Table 1: Core additivity ablation (minimal, softnorm, trig, implicit, univ, univ_warm)
  - Table 2: Encoder alignment ablation (univ, mixup_enc, univ_enc, cycle, enc_anneal)
  - Table 3: CV-VAE / RAE verification (latent distribution stats)

All metrics reported as mean ± std where available.

Usage:
    python toy_evaluation/generate_exp1_tables.py \
        --summary toy_evaluation/eval_iclr_exp1_10s/eval_all_summary.json \
        --output_dir toy_evaluation/eval_iclr_exp1_10s
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


# ---- model display names ----
DISPLAY = {
    "transformer_minimal_full_best": "Minimal",
    "transformer_softMax_full_best": "Soft-Norm",
    "transformer_best_full_best": "Trig Mixup",
    "transformer_implicit_full_v2_best": "Implicit (Torres)",
    "transformer_univ_full_best": "Universal",
    "transformer_univ_warm_v2_best": "Universal (warm)",
    "transformer_mixup_enc_best": "Mixup+Enc",
    "transformer_univ_enc_best": "Univ+Enc",
    "univ_cycle_enc_best": "Univ+Cycle",
    "univ_enc_anneal_kl2_best": "Univ+Enc+Anneal",
}

CORE_ORDER = [
    "transformer_minimal_full_best", "transformer_softMax_full_best",
    "transformer_best_full_best", "transformer_implicit_full_v2_best",
    "transformer_univ_full_best", "transformer_univ_warm_v2_best",
]
ENC_ORDER = [
    "transformer_univ_full_best", "transformer_mixup_enc_best",
    "transformer_univ_enc_best", "univ_cycle_enc_best",
    "univ_enc_anneal_kl2_best",
]


def get_metric(ckpt_data, *path, default=None):
    """Navigate nested dict/list to get a metric value."""
    d = ckpt_data
    for p in path:
        if isinstance(d, dict):
            d = d.get(p, default)
        elif isinstance(d, list) and isinstance(p, int):
            d = d[p] if p < len(d) else default
        else:
            return default
        if d is None:
            return default
    return d


def fmt(mean, std=None, prec=4):
    """Format mean ± std."""
    if mean is None:
        return "—"
    if std is not None and std > 0:
        return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"
    return f"{mean:.{prec}f}"


def extract_row(ck_data):
    """Extract all metrics for one checkpoint."""
    prots = ck_data.get("protocols", {})
    ld = prots.get("latent_distance", {})
    lims = prots.get("lims", {})
    topo = prots.get("topology", {})
    recon = prots.get("reconstruction", {})

    # @0.5 is index 5 in the 11-step alpha array
    row = {}

    # Encoder homomorphism (MSE)
    row["enc_trig_mse@0.5"] = get_metric(ld, "mean_trig_mse", 5)
    row["enc_trig_mse@0.5_std"] = get_metric(ld, "std_trig_mse", 5)
    row["enc_lerp_mse@0.5"] = get_metric(ld, "mean_lerp_mse", 5)
    row["enc_lerp_mse@0.5_std"] = get_metric(ld, "std_lerp_mse", 5)

    # Decoder additivity (LSD dB)
    row["dec_trig_lsd@0.5"] = get_metric(lims, "mean_trig_lsd", 5)
    row["dec_trig_lsd@0.5_std"] = get_metric(lims, "std_trig_lsd", 5)
    row["dec_lerp_lsd@0.5"] = get_metric(lims, "mean_lerp_lsd", 5)
    row["dec_lerp_lsd@0.5_std"] = get_metric(lims, "std_lerp_lsd", 5)

    # Energy
    row["dec_trig_energy@0.5"] = get_metric(lims, "mean_trig_energy", 5)

    # Latent geometry
    row["effective_rank"] = get_metric(topo, "latent_effective_rank")
    row["latent_density"] = get_metric(topo, "latent_density")
    # convexity is nested dict: latent_convexity.trig
    conv = get_metric(topo, "latent_convexity")
    if isinstance(conv, dict):
        row["convexity_trig"] = conv.get("trig")
    else:
        row["convexity_trig"] = conv
    row["norm_pres_err@0.5"] = get_metric(ld, "mean_abs_norm_preservation", 5)

    # Path lengths
    row["path_length_trig"] = get_metric(ld, "mean_path_length_trig")
    row["path_length_lerp"] = get_metric(ld, "mean_path_length_lerp")

    # Anisotropy
    row["latent_anisotropy"] = get_metric(topo, "latent_anisotropy")

    # Reconstruction
    row["recon_msstft"] = get_metric(recon, "mean_msstft", default=get_metric(recon, "msstft"))
    row["recon_lsd"] = get_metric(recon, "mean_lsd")

    return row


def make_table(checkpoints_data, order, caption, label):
    """Generate a LaTeX table."""
    rows = []
    for ckpt_name in order:
        ck_data = checkpoints_data.get(ckpt_name, {})
        if not ck_data:
            continue
        r = extract_row(ck_data)
        display = DISPLAY.get(ckpt_name, ckpt_name)
        rows.append((display, r))

    if not rows:
        return f"% No data for {caption}\n"

    cols = [
        ("Config", None),
        ("$\\Delta$Trig$\\downarrow$", "dec_trig_lsd@0.5"),
        ("$\\Delta$LERP$\\downarrow$", "dec_lerp_lsd@0.5"),
        ("Enc MSE$\\downarrow$", "enc_trig_mse@0.5"),
        ("RMS@0.5", "dec_trig_energy@0.5"),
        ("MS-STFT$\\downarrow$", "recon_msstft"),
        ("Eff-Rank$\\uparrow$", "effective_rank"),
        ("Conv$\\downarrow$", "convexity_trig"),
        ("NormErr$\\downarrow$", "norm_pres_err@0.5"),
    ]

    n_cols = len(cols)
    col_fmt = "l" + "r" * (n_cols - 1)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")
    # header
    hdr = " & ".join(c[0] for c in cols)
    lines.append(hdr + r" \\")
    lines.append(r"\midrule")
    # rows
    for display, r in rows:
        vals = [display]
        for _, key in cols[1:]:
            if key:
                mean = r.get(key)
                std_key = key + "_std"
                std = r.get(std_key)
                vals.append(fmt(mean, std, prec=3 if "mse" in key or "msstft" in key else 2))
            else:
                vals.append("")
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, help="Path to eval_all_summary.json")
    p.add_argument("--output_dir", default=".")
    args = p.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    checkpoints_data = data["checkpoints"]

    # Table 1: Core ablation
    t1 = make_table(checkpoints_data, CORE_ORDER,
                    "Core additivity ablation (10s clips, 1000 pairs, 2000 recon clips, seed 42). "
                    "$\\Delta$Trig/LERP = decoder LSD at $\\alpha{=}0.5$ (lower = more additive). "
                    "Enc MSE = encoder homomorphism. "
                    "MS-STFT = reconstruction cost. "
                    "Eff-Rank = participation ratio. "
                    "Conv = latent convexity. "
                    "NormErr = $|\\|\\mathrm{trigMix}(Z)\\| - \\|Z_A\\||$ at $\\alpha{=}0.5$.",
                    "tab:exp1-core")

    # Table 2: Encoder alignment
    t2 = make_table(checkpoints_data, ENC_ORDER,
                    "Encoder alignment ablation (same metrics). "
                    "Univ = decoder-only universal (reference). "
                    "Mixup+Enc = direct encoder cone (collapse case). "
                    "Univ+Enc = universal + encoder cone. "
                    "Univ+Cycle = cycle-consistency (distribution caveat: "
                    "var 0.63, $|\\cos|{=}0.67$, norm ratio 1.27$\\times$). "
                    "Univ+Enc+Anneal = annealed encoder cone + boosted KL.",
                    "tab:exp1-enc")

    # Write output
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tex_file = out / "exp1_tables.tex"
    with open(tex_file, "w") as f:
        f.write("% Auto-generated by generate_exp1_tables.py\n")
        f.write(f"% Source: {args.summary}\n\n")
        f.write(t1)
        f.write("\n")
        f.write(t2)
    print(f"Tables written to {tex_file}")

    # Also print to stdout
    print("\n" + "="*80)
    print(t1)
    print(t2)


if __name__ == "__main__":
    main()
