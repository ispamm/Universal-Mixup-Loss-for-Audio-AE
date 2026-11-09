#!/usr/bin/env python3
"""
Generate a human-readable .txt report (tables) from separation_summary_v2.json.

Reads the aggregated v2 summary and writes a fixed-width text file with tables
for: reconstruction, algebraic separation, ΔSI-SDR, trigonometric/linear latent
mix (midpoint), and — most importantly — **path-following** metrics across the
full α ∈ [0,1] interpolation trajectory (9 steps): path mean, path flatness
(std across α), and path range (max−min across α).

All values: mean ± SEM (standard error = sample std / sqrt(n), ddof=1).

Usage:
    python evaluation/separation_compare/report_v2.py \
        --summary evaluation/separation_compare/separation_summary_v2.json \
        --out    evaluation/separation_compare/REPORT_v2.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STEMS = ["drums", "bass", "other", "vocals"]
PAIRS = ["drums+bass", "drums+other", "drums+vocals", "bass+other", "bass+vocals", "other+vocals"]


def fms(v: float) -> str:
    """Format an MS-STFT MSE value compactly."""
    if abs(v) < 0.001:
        return f"{v*1e6:.2f}e-6"
    if abs(v) < 1.0:
        return f"{v:.5f}"
    return f"{v:.3f}"


def _sisdr(d: dict) -> str:
    return f"{d['mean']:7.2f} ± {d['sem']:6.2f}"


def _ms(d: dict) -> str:
    return f"{fms(d['mean']):>10} ± {fms(d['sem']):>10}"


def _sisdr_tight(d: dict) -> str:
    return f"{d['mean']:7.2f}±{d['sem']:5.2f}"


def _ms_tight(d: dict) -> str:
    return f"{fms(d['mean']):>9}±{fms(d['sem']):>9}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s = json.loads(args.summary.read_text())
    order = s["model_order"]
    models = s["models"]
    alphas = models[order[0]].get("alphas", [])
    alphas_str = ", ".join(f"{a:.3f}" for a in alphas) if alphas else "?"

    L: list[str] = []
    A = L.append

    A("=" * 108)
    A("Algebraic Source Separation & Latent Mixing on MUSDB18-HQ")
    A("=" * 108)
    A("")
    A("Models (all evaluated at 44.1 kHz on the same 100 MUSDB18-HQ tracks, seed 42):")
    A("  SAME-441  : transformer (norm-preserving) checkpoint")
    A("              training/checkpoints/transformer_best_full_441_best.pt")
    A("  SAME-L    : original SAME autoencoder (same-l), stable_audio_3")
    A("  Lin-CAE   : linear consistency autoencoder (lin-cae)")
    A("  DAC       : Descript Audio Codec, 44 kHz, full RVQ-quantized pipeline")
    A("")
    A("Protocol:")
    A("  100 MUSDB18-HQ tracks (seed 42; 70 train / 30 test), 2 s centre mono chunks.")
    A("  mixture = drums + bass + other + vocals (to ~1e-4).")
    A("  - Reconstruction : recon = decode(encode(stem))                     [400 trials]")
    A("  - Separation    : z_sep = z_mix - sum(z_other_stems); decode        [400 trials]")
    A("  - Trig/Lerp mix : decode(cos(θ)·z_A + sin(θ)·z_B) and decode((1-α)z_A + αz_B),")
    A(f"                    α swept over {len(alphas)} steps: [{alphas_str}]        [600 trials]")
    A("")
    A("Metrics:")
    A("  SI-SDR  : scale-invariant SDR (dB, higher = better)")
    A("  MS-STFT : multi-scale STFT magnitude MSE (7 FFT sizes 32-2048, lower = better)")
    A("  ΔSI-SDR : sep_sisdr - recon_sisdr (dB; 0 = separation as good as own recon)")
    A("  Path-following (mix tasks, full α trajectory):")
    A("    path_mean  : per-trial mean of the metric across α → mean ± SEM across trials")
    A("                 (how well the whole path is followed on average; ↑ better for SI-SDR)")
    A("    path_std   : per-trial std across α → mean ± SEM  (path flatness; ↓ = more uniform)")
    A("    path_range : per-trial (max−min) across α → mean ± SEM (worst-case deviation; ↓ better)")
    A("  All values: mean ± SEM (std/sqrt(n), ddof=1). NaN SI-SDR (silent targets) dropped.")
    A("")
    A("=" * 108)
    A("")

    # ---- 1. Reconstruction (overall) ----
    A("-" * 108)
    A("TABLE 1. Reconstruction quality (overall, 400 trials)")
    A("-" * 108)
    A(f"{'Model':<14}{'SI-SDR (dB) mean±SEM':>24}{'MS-STFT MSE mean±SEM':>30}{'n':>8}")
    for name in order:
        m = models[name]["recon_overall"]
        A(f"{name:<14}{_sisdr(m['sisdr']):>24}{_ms(m['ms']):>30}{m['sisdr']['n']:>8}")
    A("")

    # ---- 2. Separation (overall) ----
    A("-" * 108)
    A("TABLE 2. Algebraic source separation (overall, 400 trials)")
    A("-" * 108)
    A(f"{'Model':<14}{'sep SI-SDR mean±SEM':>24}{'sep MS-STFT mean±SEM':>30}{'ΔSI-SDR mean±SEM':>24}")
    for name in order:
        m = models[name]["sep_overall"]
        A(f"{name:<14}{_sisdr(m['sisdr']):>24}{_ms(m['ms']):>30}{_sisdr(m['delta_sisdr']):>24}")
    A("")

    # ---- 3. Separation per stem ----
    A("-" * 108)
    A("TABLE 3. Algebraic separation — per stem (ΔSI-SDR = sep - recon, dB)")
    A("-" * 108)
    for stem in STEMS:
        A(f"  Stem: {stem}")
        A(f"{'Model':<14}{'sep SI-SDR':>18}{'ΔSI-SDR':>18}{'sep MS-STFT':>28}")
        for name in order:
            d = models[name]["sep_per_stem"][stem]
            A(f"{name:<14}{_sisdr_tight(d['sisdr']):>18}{_sisdr_tight(d['delta_sisdr']):>18}"
              f"{_ms_tight(d['ms']):>28}")
        A("")

    # ---- 4. Trig mix midpoint (overall + per pair) ----
    A("-" * 108)
    A("TABLE 4. Trigonometric latent mix — midpoint (α=0.5)")
    A("-" * 108)
    A(f"{'Model':<14}{'SI-SDR (dB) mean±SEM':>24}{'MS-STFT MSE mean±SEM':>30}{'n':>8}")
    for name in order:
        m = models[name]["trig_overall"]
        A(f"{name:<14}{_sisdr(m['sisdr']):>24}{_ms(m['ms']):>30}{m['sisdr']['n']:>8}")
    A("")
    A("  Per stem-pair (midpoint α=0.5):")
    for pair in PAIRS:
        A(f"    {pair}:")
        A(f"{'Model':<14}{'SI-SDR (dB)':>18}{'MS-STFT MSE':>24}")
        for name in order:
            d = models[name]["trig_per_pair"][pair]
            A(f"{name:<14}{_sisdr_tight(d['sisdr']):>18}{_ms_tight(d['ms']):>24}")
        A("")

    # ---- 5. Lerp mix midpoint (overall + per pair) ----
    A("-" * 108)
    A("TABLE 5. Linear latent mix — midpoint (α=0.5)")
    A("-" * 108)
    A(f"{'Model':<14}{'SI-SDR (dB) mean±SEM':>24}{'MS-STFT MSE mean±SEM':>30}{'n':>8}")
    for name in order:
        m = models[name]["lerp_overall"]
        A(f"{name:<14}{_sisdr(m['sisdr']):>24}{_ms(m['ms']):>30}{m['sisdr']['n']:>8}")
    A("")
    A("  Per stem-pair (midpoint α=0.5):")
    for pair in PAIRS:
        A(f"    {pair}:")
        A(f"{'Model':<14}{'SI-SDR (dB)':>18}{'MS-STFT MSE':>24}")
        for name in order:
            d = models[name]["lerp_per_pair"][pair]
            A(f"{name:<14}{_sisdr_tight(d['sisdr']):>18}{_ms_tight(d['ms']):>24}")
        A("")

    # ---- 6. Reconstruction per stem (reference) ----
    A("-" * 108)
    A("TABLE 6. Reconstruction quality — per stem")
    A("-" * 108)
    for stem in STEMS:
        A(f"  Stem: {stem}")
        A(f"{'Model':<14}{'SI-SDR (dB)':>18}{'MS-STFT MSE':>24}")
        for name in order:
            d = models[name]["recon_per_stem"][stem]
            A(f"{name:<14}{_sisdr_tight(d['sisdr']):>18}{_ms_tight(d['ms']):>24}")
        A("")

    # ---- 7. Trig mix path-following (overall) ----
    A("-" * 108)
    A("TABLE 7. Trigonometric latent mix — PATH-FOLLOWING (full α ∈ [0,1] trajectory, 9 steps)")
    A("-" * 108)
    A("  SI-SDR (dB): path_mean ↑ better | path_std ↓ = flatter | path_range ↓ = smaller worst-case")
    A(f"{'Model':<14}{'path_mean':>22}{'path_std (flat)':>22}{'path_range':>22}")
    for name in order:
        p = models[name]["trig_overall"]["path_sisdr"]
        A(f"{name:<14}{_sisdr(p['path_mean']):>22}{_sisdr(p['path_std']):>22}{_sisdr(p['path_range']):>22}")
    A("")
    A("  MS-STFT MSE: path_mean ↓ better | path_std ↓ = flatter | path_range ↓ = smaller worst-case")
    A(f"{'Model':<14}{'path_mean':>24}{'path_std (flat)':>24}{'path_range':>24}")
    for name in order:
        p = models[name]["trig_overall"]["path_ms"]
        A(f"{name:<14}{_ms(p['path_mean']):>24}{_ms(p['path_std']):>24}{_ms(p['path_range']):>24}")
    A("")
    A("  Per stem-pair (SI-SDR path-following):")
    for pair in PAIRS:
        A(f"    {pair}:")
        A(f"{'Model':<14}{'path_mean':>18}{'path_std':>18}{'path_range':>18}")
        for name in order:
            p = models[name]["trig_per_pair"][pair]["path_sisdr"]
            A(f"{name:<14}{_sisdr_tight(p['path_mean']):>18}{_sisdr_tight(p['path_std']):>18}"
              f"{_sisdr_tight(p['path_range']):>18}")
        A("")

    # ---- 8. Lerp mix path-following (overall) ----
    A("-" * 108)
    A("TABLE 8. Linear latent mix — PATH-FOLLOWING (full α ∈ [0,1] trajectory, 9 steps)")
    A("-" * 108)
    A("  SI-SDR (dB): path_mean ↑ better | path_std ↓ = flatter | path_range ↓ = smaller worst-case")
    A(f"{'Model':<14}{'path_mean':>22}{'path_std (flat)':>22}{'path_range':>22}")
    for name in order:
        p = models[name]["lerp_overall"]["path_sisdr"]
        A(f"{name:<14}{_sisdr(p['path_mean']):>22}{_sisdr(p['path_std']):>22}{_sisdr(p['path_range']):>22}")
    A("")
    A("  MS-STFT MSE: path_mean ↓ better | path_std ↓ = flatter | path_range ↓ = smaller worst-case")
    A(f"{'Model':<14}{'path_mean':>24}{'path_std (flat)':>24}{'path_range':>24}")
    for name in order:
        p = models[name]["lerp_overall"]["path_ms"]
        A(f"{name:<14}{_ms(p['path_mean']):>24}{_ms(p['path_std']):>24}{_ms(p['path_range']):>24}")
    A("")
    A("  Per stem-pair (SI-SDR path-following):")
    for pair in PAIRS:
        A(f"    {pair}:")
        A(f"{'Model':<14}{'path_mean':>18}{'path_std':>18}{'path_range':>18}")
        for name in order:
            p = models[name]["lerp_per_pair"][pair]["path_sisdr"]
            A(f"{name:<14}{_sisdr_tight(p['path_mean']):>18}{_sisdr_tight(p['path_std']):>18}"
              f"{_sisdr_tight(p['path_range']):>18}")
        A("")

    A("=" * 108)
    A("Notes:")
    A("  - Lin-CAE decode is a stochastic one-step consistency pass (seeded per decode).")
    A("  - DAC uses the full RVQ-quantized pipeline (bypass_quantizer=False).")
    A("  - SAME-L is fed a stereo copy of the mono chunk (matches lims_test_same.py).")
    A("  - All 4 models run at 44.1 kHz on identical chunks; ΔSI-SDR additionally")
    A("    anchors each model to its own reconstruction quality.")
    A("  - SEM = sample std (ddof=1) / sqrt(n); it measures the uncertainty of the")
    A("    reported mean, not the dispersion of individual trials.")
    A("  - Path-following: for each (track, stem-pair) trial, the metric is computed at")
    A("    each α step; path_mean/std/range are the per-trial mean, std, and (max−min)")
    A("    across α, then aggregated across trials as mean ± SEM.")
    A("=" * 108)

    out = "\n".join(L) + "\n"
    args.out.write_text(out)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
