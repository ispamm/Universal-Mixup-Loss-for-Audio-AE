#!/usr/bin/env python3
"""
Comparative interpolation-path analysis for two checkpoints over 1000 pairs.

Reuses evaluation.lims_test.evaluate_lims_checkpoint so the metrics are
identical to the standard LIMS protocol:

  * Path-following:  LSD( decoded(interp(Z_A, Z_B, α)) , ideal_mix(α) )
    for both Lerp (linear) and Trig (constant-power) schemes, at every α ∈
    {0.0, 0.1, ..., 1.0}.  Lower LSD = the decoded interpolation follows the
    ideal audio-space mix more closely along the WHOLE path, not only α=0.5.

  * Energy similarity:  normalised RMS( decoded(interp(α)) ) vs the normalised
    RMS of the real audio ideal mix at every α, for both Lerp and Trig.
    Smaller |decoded_energy − ideal_energy| = the model preserves the mix
    energy closer to the real audio mix energy along the whole path.

Both checkpoints are scored on the SAME 1000 deterministic pairs (seed 42) so
the comparison is paired and fair.

Outputs:
  evaluation/interp_path_compare_1000/
      best_full_vs_implicit_v2_1000.json   — raw aggregated curves
      REPORT.md                            — comparative analysis writeup
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
)
from evaluation.lims_test import evaluate_lims_checkpoint

# =============================================================================
#  Defaults
# =============================================================================

CKPT_A = "training/checkpoints/transformer_best_full_best.pt"
CKPT_B = "training/checkpoints/transformer_implicit_full_v2_best.pt"
LABEL_A = "best_full"
LABEL_B = "implicit_v2"
NUM_PAIRS = 1000
SEED = 42
INTERP_STEPS = 11
OUT_DIR = Path("evaluation/interp_path_compare_1000")


# =============================================================================
#  Aggregation helpers
# =============================================================================

def path_following_summary(res: dict) -> dict:
    """Per-α LSD for both schemes + whole-path aggregate stats."""
    alphas = res["alphas"]
    out = {
        "alphas": alphas,
        "lerp_lsd_mean": res["mean_lerp_lsd"],          # Lerp decoded vs Lerp ideal mix
        "lerp_lsd_std":  res["std_lerp_lsd"],
        "trig_lsd_mean": res["mean_trig_lsd"],          # Trig decoded vs Trig ideal mix
        "trig_lsd_std":  res["std_trig_lsd"],
        # Also the cross-target LSDs already computed by the protocol:
        "lerp_lsd_vs_trig_mean": res["mean_lerp_lsd_vs_trig"],
        "trig_lsd_vs_lerp_mean": res["mean_trig_lsd_vs_lerp"],
    }
    arr_lerp = np.asarray(res["mean_lerp_lsd"])
    arr_trig = np.asarray(res["mean_trig_lsd"])
    interior = slice(1, -1)  # α ∈ {0.1 .. 0.9} — the interpolation path itself
    out["lerp_interior_mean_lsd"] = float(arr_lerp[interior].mean())
    out["trig_interior_mean_lsd"] = float(arr_trig[interior].mean())
    out["lerp_path_lsd"] = float(arr_lerp.mean())          # whole path incl. endpoints
    out["trig_path_lsd"] = float(arr_trig.mean())
    out["lerp_curve_range"] = float(arr_lerp.max() - arr_lerp.min())
    out["trig_curve_range"] = float(arr_trig.max() - arr_trig.min())
    # Existing curve-smoothness metric from the protocol (variance of trig curve)
    out["trig_curve_smoothness"] = res.get("lsd_curve_smoothness_trig", float("nan"))
    return out


def energy_similarity_summary(res: dict) -> dict:
    """Per-α decoded energy vs ideal-mix energy + gap stats."""
    alphas = res["alphas"]
    gt_lerp = np.asarray(res["ground_truth_lerp_energy_mean"])
    gt_trig = np.asarray(res["ground_truth_trig_energy_mean"])
    dec_lerp = np.asarray(res["mean_lerp_energy"])
    dec_trig = np.asarray(res["mean_trig_energy"])
    out = {
        "alphas": alphas,
        "lerp_decoded_energy": dec_lerp.tolist(),
        "lerp_ideal_energy": gt_lerp.tolist(),
        "trig_decoded_energy": dec_trig.tolist(),
        "trig_ideal_energy": gt_trig.tolist(),
        "lerp_energy_gap": np.abs(dec_lerp - gt_lerp).tolist(),
        "trig_energy_gap": np.abs(dec_trig - gt_trig).tolist(),
    }
    interior = slice(1, -1)
    out["lerp_interior_mean_energy_gap"] = float(np.abs(dec_lerp - gt_lerp)[interior].mean())
    out["trig_interior_mean_energy_gap"] = float(np.abs(dec_trig - gt_trig)[interior].mean())
    out["lerp_whole_mean_energy_gap"] = float(np.abs(dec_lerp - gt_lerp).mean())
    out["trig_whole_mean_energy_gap"] = float(np.abs(dec_trig - gt_trig).mean())
    out["lerp_energy_curve_flatness"] = float(np.std(dec_lerp))
    out["trig_energy_curve_flatness"] = float(np.std(dec_trig))
    # Protocol's own trig gap metric
    out["trig_energy_gap_protocol"] = res.get("energy_curve_gap_trig", float("nan"))
    return out


# =============================================================================
#  Report
# =============================================================================

def _fmt(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a"
    return f"{x:.{n}f}"


def write_report(path_a_summary, path_b_summary, energy_a, energy_b, meta, out_md):
    """Write a Markdown comparative analysis."""
    L = [
        "# Interpolation-path comparative analysis (1000 audio pairs)",
        "",
        f"- **Generated:** {meta['timestamp']}",
        f"- **Pairs:** {meta['num_pairs']} (seed {meta['seed']}, mode random, "
        f"{meta['valid_pairs']} valid)",
        f"- **Interpolation steps:** {meta['interp_steps']} α ∈ "
        f"{[round(a,1) for a in meta['alphas']]}",
        f"- **Model A:** `{meta['ckpt_a']}`",
        f"- **Model B:** `{meta['ckpt_b']}`",
        "",
        "## 1. Path-following — LSD vs ideal mix along the whole path",
        "",
        "LSD (dB, lower = decoded interpolation matches the ideal audio mix "
        "more closely). The *interior* mean (α ∈ {0.1…0.9}) is the pure "
        "interpolation path; the *whole-path* mean also includes the endpoint "
        "reconstructions (α=0,1).",
        "",
        "| Metric | Model A (best_full) | Model B (implicit_v2) | Better |",
        "|---|---|---|---|",
    ]

    def row(label, va, vb, lower_better=True):
        if va is None or vb is None:
            return f"| {label} | {_fmt(va)} | {_fmt(vb)} | n/a |"
        diff = vb - va
        if abs(diff) < 1e-9:
            winner = "tie"
        elif (diff > 0) == lower_better:
            winner = "**A**"
        else:
            winner = "**B**"
        return f"| {label} | {_fmt(va)} | {_fmt(vb)} | {winner} |"

    L += [
        row("Lerp — interior mean LSD (↓)", path_a_summary["lerp_interior_mean_lsd"], path_b_summary["lerp_interior_mean_lsd"]),
        row("Lerp — whole-path mean LSD (↓)", path_a_summary["lerp_path_lsd"], path_b_summary["lerp_path_lsd"]),
        row("Trig — interior mean LSD (↓)", path_a_summary["trig_interior_mean_lsd"], path_b_summary["trig_interior_mean_lsd"]),
        row("Trig — whole-path mean LSD (↓)", path_a_summary["trig_path_lsd"], path_b_summary["trig_path_lsd"]),
        row("Lerp — curve range (max−min LSD) (↓)", path_a_summary["lerp_curve_range"], path_b_summary["lerp_curve_range"]),
        row("Trig — curve range (max−min LSD) (↓)", path_a_summary["trig_curve_range"], path_b_summary["trig_curve_range"]),
        row("Trig — curve smoothness (var of mean LSD) (↓)", path_a_summary["trig_curve_smoothness"], path_b_summary["trig_curve_smoothness"]),
        "",
        "### 1a. Per-α LSD curves",
        "",
        "| α | Lerp A | Lerp B | Trig A | Trig B |",
        "|---|---|---|---|---|",
    ]
    alphas = meta["alphas"]
    for i, a in enumerate(alphas):
        L.append(f"| {a:.1f} | {_fmt(path_a_summary['lerp_lsd_mean'][i])} | "
                 f"{_fmt(path_b_summary['lerp_lsd_mean'][i])} | "
                 f"{_fmt(path_a_summary['trig_lsd_mean'][i])} | "
                 f"{_fmt(path_b_summary['trig_lsd_mean'][i])} |")

    L += [
        "",
        "## 2. Energy similarity — decoded vs real audio mix energy",
        "",
        "Energy is normalised RMS(decoded) / mean_RMS(A,B). The *ideal* curves "
        "are computed directly from the raw waveform mixes. The energy **gap** "
        "= |decoded_energy − ideal_energy| at each α (lower = model preserves "
        "the real mix energy more closely).",
        "",
        "| Metric | Model A (best_full) | Model B (implicit_v2) | Better |",
        "|---|---|---|---|",
        row("Lerp — interior mean energy gap (↓)", energy_a["lerp_interior_mean_energy_gap"], energy_b["lerp_interior_mean_energy_gap"]),
        row("Lerp — whole-path mean energy gap (↓)", energy_a["lerp_whole_mean_energy_gap"], energy_b["lerp_whole_mean_energy_gap"]),
        row("Trig — interior mean energy gap (↓)", energy_a["trig_interior_mean_energy_gap"], energy_b["trig_interior_mean_energy_gap"]),
        row("Trig — whole-path mean energy gap (↓)", energy_a["trig_whole_mean_energy_gap"], energy_b["trig_whole_mean_energy_gap"]),
        row("Trig — protocol energy gap (↓)", energy_a["trig_energy_gap_protocol"], energy_b["trig_energy_gap_protocol"]),
        "",
        "### 2a. Per-α energy curves (decoded vs ideal)",
        "",
        "| α | Lerp A | Lerp ideal | Lerp B | Trig A | Trig ideal | Trig B |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, a in enumerate(alphas):
        L.append(
            f"| {a:.1f} | {_fmt(energy_a['lerp_decoded_energy'][i])} | "
            f"{_fmt(energy_a['lerp_ideal_energy'][i])} | "
            f"{_fmt(energy_b['lerp_decoded_energy'][i])} | "
            f"{_fmt(energy_a['trig_decoded_energy'][i])} | "
            f"{_fmt(energy_a['trig_ideal_energy'][i])} | "
            f"{_fmt(energy_b['trig_decoded_energy'][i])} |"
        )

    L += [
        "",
        "### 2b. Per-α energy gaps",
        "",
        "| α | Lerp gap A | Lerp gap B | Trig gap A | Trig gap B |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(alphas):
        L.append(
            f"| {a:.1f} | {_fmt(energy_a['lerp_energy_gap'][i])} | "
            f"{_fmt(energy_b['lerp_energy_gap'][i])} | "
            f"{_fmt(energy_a['trig_energy_gap'][i])} | "
            f"{_fmt(energy_b['trig_energy_gap'][i])} |"
        )

    L += [
        "",
        "## 3. Verdict",
        "",
    ]
    # Build a compact verdict from the interior mean stats.
    pf_lerp = "A" if path_a_summary["lerp_interior_mean_lsd"] <= path_b_summary["lerp_interior_mean_lsd"] else "B"
    pf_trig = "A" if path_a_summary["trig_interior_mean_lsd"] <= path_b_summary["trig_interior_mean_lsd"] else "B"
    en_lerp = "A" if energy_a["lerp_interior_mean_energy_gap"] <= energy_b["lerp_interior_mean_energy_gap"] else "B"
    en_trig = "A" if energy_a["trig_interior_mean_energy_gap"] <= energy_b["trig_interior_mean_energy_gap"] else "B"
    name = {"A": "best_full", "B": "implicit_v2"}
    L.append(f"- **Path-following — Lerp (linear):** Model **{pf_lerp}** ({name[pf_lerp]}) "
             f"(interior mean LSD {_fmt(path_a_summary['lerp_interior_mean_lsd'] if pf_lerp=='A' else path_b_summary['lerp_interior_mean_lsd'])} "
             f"vs {_fmt(path_b_summary['lerp_interior_mean_lsd'] if pf_lerp=='A' else path_a_summary['lerp_interior_mean_lsd'])})")
    L.append(f"- **Path-following — Trig (constant-power):** Model **{pf_trig}** ({name[pf_trig]}) "
             f"(interior mean LSD {_fmt(path_a_summary['trig_interior_mean_lsd'] if pf_trig=='A' else path_b_summary['trig_interior_mean_lsd'])} "
             f"vs {_fmt(path_b_summary['trig_interior_mean_lsd'] if pf_trig=='A' else path_a_summary['trig_interior_mean_lsd'])})")
    L.append(f"- **Energy similarity — Lerp (linear):** Model **{en_lerp}** ({name[en_lerp]}) "
             f"(interior mean gap {_fmt(energy_a['lerp_interior_mean_energy_gap'] if en_lerp=='A' else energy_b['lerp_interior_mean_energy_gap'])} "
             f"vs {_fmt(energy_b['lerp_interior_mean_energy_gap'] if en_lerp=='A' else energy_a['lerp_interior_mean_energy_gap'])})")
    L.append(f"- **Energy similarity — Trig (constant-power):** Model **{en_trig}** ({name[en_trig]}) "
             f"(interior mean gap {_fmt(energy_a['trig_interior_mean_energy_gap'] if en_trig=='A' else energy_b['trig_interior_mean_energy_gap'])} "
             f"vs {_fmt(energy_b['trig_interior_mean_energy_gap'] if en_trig=='A' else energy_a['trig_interior_mean_energy_gap'])})")
    L.append("")
    L.append("> Lower LSD and lower energy gap are better. “Interior” = α ∈ "
             "{0.1…0.9} (the interpolation path between the two endpoints); "
             "“whole-path” also counts the α=0/1 endpoint reconstructions.")

    out_md.write_text("\n".join(L))
    return pf_lerp, pf_trig, en_lerp, en_trig


# =============================================================================
#  Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_a", default=CKPT_A)
    p.add_argument("--ckpt_b", default=CKPT_B)
    p.add_argument("--num_pairs", type=int, default=NUM_PAIRS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Collecting validation stems …")
    stems = collect_validation_stems()
    print(f"  {len(stems)} validation stems")
    print(f"[2/4] Generating {args.num_pairs} pairs (seed {args.seed}) …")
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    print(f"  {len(pairs)} pairs")

    results = {}
    for label, ckpt in [("A", args.ckpt_a), ("B", args.ckpt_b)]:
        print(f"[3/4] Evaluating model {label}: {ckpt}")
        t0 = time.time()
        res = evaluate_lims_checkpoint(
            ckpt, stems, device,
            num_pairs=args.num_pairs, seed=args.seed,
            pairs=pairs, interp_steps=INTERP_STEPS,
        )
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s — valid_pairs={res['valid_pairs']}")
        results[label] = res

    meta = {
        "timestamp": datetime.now().isoformat(),
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "num_pairs": args.num_pairs,
        "seed": args.seed,
        "interp_steps": INTERP_STEPS,
        "alphas": results["A"]["alphas"],
        "valid_pairs": results["A"]["valid_pairs"],
    }

    path_a = path_following_summary(results["A"])
    path_b = path_following_summary(results["B"])
    en_a = energy_similarity_summary(results["A"])
    en_b = energy_similarity_summary(results["B"])

    json_path = out_dir / "best_full_vs_implicit_v2_1000.json"
    with open(json_path, "w") as fp:
        json.dump({
            "meta": meta,
            "model_A_path_following": path_a,
            "model_B_path_following": path_b,
            "model_A_energy": en_a,
            "model_B_energy": en_b,
        }, fp, indent=2)
    print(f"[4/4] JSON → {json_path}")

    md_path = out_dir / "REPORT.md"
    write_report(path_a, path_b, en_a, en_b, meta, md_path)
    print(f"      REPORT → {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
