#!/usr/bin/env python3
"""
Extract the unified-ablation numbers (mean ± std across the seed-42 eval set)
for the six ICASSP configurations from the *saved* evaluation artifacts.

Sources (all seed 42, deterministic pairs):
  * ΔSI-SDR (mix_recon_ratio, 1000 pairs; per-pair arrays stored):
      - 4 legacy rows : mix_recon_ratio_summary.bak_20260901_141340.json
      - Implicit(v2)  : mix_recon_ratio_summary_implicit_v2.json
      - Mixup+Enc     : mix_recon_ratio_summary_mixup_enc_full.json
  * eval_all battery (1000 pairs / 256 clips / 512 mixups / 200 recon clips):
      - Minimal, Soft-Norm, Full, Norm-Preserving : eval_all/eval_all_summary.bak_20260901_141340.json
      - Implicit(v2)  : eval_all_implicit_v2/eval_all_summary.json
      - Mixup+Enc     : eval_all_mixup_enc_full/eval_all_summary.json

Per-cell semantics:
  mean ± std = dispersion across the 1000 evaluation pairs (200 recon clips /
  256 topology clips) on the fixed seed-42 set.
  Columns that are themselves curve/pool statistics (dec/enc Flatness,
  Eff. Rank, Convexity) are reported as single values (no ±).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent  # toy_evaluation
ROOT = EVAL.parent

MIX_RECON = {
    "transformer_minimal_full_best": EVAL / "mix_recon_ratio_summary.bak_20260901_141340.json",
    "transformer_softMax_full_best": EVAL / "mix_recon_ratio_summary.bak_20260901_141340.json",
    "transformer_norm_full_best": EVAL / "mix_recon_ratio_summary.bak_20260901_141340.json",
    "transformer_best_full_best": EVAL / "mix_recon_ratio_summary.bak_20260901_141340.json",
    "transformer_implicit_full_v2_best": EVAL / "mix_recon_ratio_summary_implicit_v2.json",
    "transformer_mixup_enc_best": EVAL / "mix_recon_ratio_summary_mixup_enc_full.json",
}

BATTERY = {
    "transformer_minimal_full_best": EVAL / "eval_all/eval_all_summary.bak_20260901_141340.json",
    "transformer_softMax_full_best": EVAL / "eval_all/eval_all_summary.bak_20260901_141340.json",
    "transformer_norm_full_best": EVAL / "eval_all/eval_all_summary.bak_20260901_141340.json",
    "transformer_best_full_best": EVAL / "eval_all/eval_all_summary.bak_20260901_141340.json",
    "transformer_implicit_full_v2_best": EVAL / "eval_all_implicit_v2/eval_all_summary.json",
    "transformer_mixup_enc_best": EVAL / "eval_all_mixup_enc_full/eval_all_summary.json",
}


def load_ckpt(path: Path, stem: str) -> dict:
    d = json.loads(path.read_text())
    for container in ("checkpoints", "results"):
        cks = d.get(container, {})
        for k, v in cks.items():
            if k == stem or k.endswith(stem) or stem.endswith(k):
                return v
    raise KeyError(f"{stem} not in {path}")


def at_alpha(curve, alpha: float = 0.5):
    """Nearest-index value on the uniform α grid (index = round(α·(n-1)))."""
    arr = np.asarray(curve, dtype=np.float64)
    n = len(arr)
    return float(arr[int(round(alpha * (n - 1)))])


def curve_flatness(curve):
    """std over the interior α ∈ [0.1, 0.9] points of the pair-mean curve."""
    c = np.asarray(curve, dtype=np.float64)
    if len(c) < 11:
        return float("nan")
    return float(np.nanstd(c[1:10]))


def main() -> None:
    out: dict = {"note": "", "models": {}}

    for stem in MIX_RECON:
        rec = load_ckpt(MIX_RECON[stem], stem)
        pp = rec["per_pair"]
        d_trig = np.array([p["mix_trig_sisdr"] - p["recon_sisdr"] for p in pp])
        d_lerp = np.array([p["mix_lerp_sisdr"] - p["recon_sisdr"] for p in pp])
        n = int(len(d_trig))
        m = out["models"].setdefault(stem, {})
        m["delta_sisdr_trig"] = {"mean": float(np.mean(d_trig)), "std": float(np.std(d_trig)),
                                 "sem": float(np.std(d_trig) / np.sqrt(n)), "n": n}
        m["delta_sisdr_lerp"] = {"mean": float(np.mean(d_lerp)), "std": float(np.std(d_lerp)),
                                 "sem": float(np.std(d_lerp) / np.sqrt(n)), "n": n}

    for stem in BATTERY:
        rec = load_ckpt(BATTERY[stem], stem)
        prots = rec["protocols"]
        ld, lm, rc, tp = prots["latent_distance"], prots["lims"], prots["reconstruction"], prots["topology"]
        m = out["models"].setdefault(stem, {})
        n_ld = int(ld.get("valid_pairs") or 0)
        n_lims = int(lm.get("valid_pairs") or 0)
        n_recon = int(rc.get("num_clips_valid") or 0)

        def sem(sample_std: float, n: int) -> float:
            return float(sample_std / np.sqrt(n)) if (sample_std is not None and n > 0) else None

        # encoder homomorphism
        m["enc_trig_mse_05"] = {"mean": at_alpha(ld["mean_trig_mse"]),
                                "std": at_alpha(ld["std_trig_mse"]),
                                "sem": sem(at_alpha(ld["std_trig_mse"]), n_ld), "n": n_ld}
        m["enc_lerp_mse_05"] = {"mean": at_alpha(ld["mean_lerp_mse"]),
                                "std": at_alpha(ld["std_lerp_mse"]),
                                "sem": sem(at_alpha(ld["std_lerp_mse"]), n_ld), "n": n_ld}
        m["enc_trig_flat"] = curve_flatness(ld["mean_trig_mse"])
        m["enc_lerp_flat"] = curve_flatness(ld["mean_lerp_mse"])
        # decoder energy at α=0.5 (relative RMS) + flatness
        m["dec_rms_05"] = {"mean": at_alpha(lm["mean_trig_energy"]),
                           "std": at_alpha(lm["std_trig_energy"]),
                           "sem": sem(at_alpha(lm["std_trig_energy"]), n_lims), "n": n_lims}
        m["dec_energy_flatness"] = lm.get("energy_curve_flatness_trig")
        # reconstruction MS-STFT (per clip)
        m["recon_msstft"] = {"mean": rc.get("mean_msstft"), "std": rc.get("std_msstft"),
                             "sem": sem(rc.get("std_msstft"), n_recon), "n_clips": n_recon}
        # topology pool scalars
        m["eff_rank"] = tp.get("latent_effective_rank")
        m["convexity"] = (tp.get("latent_convexity") or {}).get("trig")
        m["topo_n_clips"] = tp.get("num_clips_valid")
        # norm preservation (mean |Δ| at α=0.5; sem filled by the supplemental run)
        m["norm_err_05"] = {"mean": at_alpha(ld["mean_abs_norm_preservation"]),
                            "std": None, "sem": None, "n": None,
                            "mean_signed": at_alpha(ld["mean_norm_preservation"]),
                            "std_signed": at_alpha(ld["std_norm_preservation"])}

    out["note"] = (
        "Means/sample-stds extracted from saved artifacts (seed 42). 'sem' = sample std / sqrt(n), "
        "i.e. the standard error of the reported mean. norm_err sem is filled by "
        "toy_evaluation/unified_ablation/norm_pres_supplement.py"
    )
    Path(__file__).parent.mkdir(exist_ok=True)
    json_path = Path(__file__).parent / "unified_data_from_artifacts.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {json_path}")

    # ---- human-readable check vs the current table ----
    order = ["transformer_minimal_full_best", "transformer_softMax_full_best",
             "transformer_norm_full_best", "transformer_best_full_best",
             "transformer_implicit_full_v2_best", "transformer_mixup_enc_best"]
    print(f"{'model':38s} {'dT':>7s} {'dL':>7s} {'RMS':>6s} {'dFlat':>6s} "
          f"{'eT':>6s} {'eTF':>6s} {'eL':>6s} {'eLF':>6s} {'MS':>6s} "
          f"{'ER':>6s} {'Conv':>7s} {'NE':>7s}")
    for s in order:
        m = out["models"][s]
        print(f"{s:38s} {m['delta_sisdr_trig']['mean']:7.2f} {m['delta_sisdr_lerp']['mean']:7.2f} "
              f"{m['dec_rms_05']['mean']:6.3f} {m['dec_energy_flatness']:6.3f} "
              f"{m['enc_trig_mse_05']['mean']:6.3f} {m['enc_trig_flat']:6.3f} "
              f"{m['enc_lerp_mse_05']['mean']:6.3f} {m['enc_lerp_flat']:6.3f} "
              f"{m['recon_msstft']['mean']:6.3f} {m['eff_rank']:6.1f} "
              f"{m['convexity']:7.1f} {m['norm_err_05']['mean']:7.2f}")


if __name__ == "__main__":
    main()
