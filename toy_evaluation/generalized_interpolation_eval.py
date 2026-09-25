#!/usr/bin/env python3
"""
Generalized latent interpolation laws — non-linear weight profiles
==================================================================

Verifies whether the continuous trigonometric latent-mixup regularisation
(training objective: ``D(cosθ·Z_A + sinθ·Z_B) ≈ cosθ·A + sinθ·B`` for
θ = α·π/2, α ~ U(0,1)) induces a **general positive-weight homomorphism cone**
rather than memorising only the trigonometric / linear (LERP) curves it was
exposed to during training.

For three non-linear weight laws  (w1(t), w2(t)) ∈ [0,1]² on the interior grid
t ∈ [0.1, 0.9] we measure, per pair (A, B) and per t:

  * Encoder homomorphism error
        MSE( Enc(w1·A + w2·B),  w1·Enc(A) + w2·Enc(B) )
    — the encoding of the *ideal acoustic weighted mix* vs the *weighted
    combination of the encodings* (the RHS is also the latent path we decode).
  * Decoder mixing quality
        ΔSI-SDR = SI-SDR(w1·A + w2·B, D(w1·Z_A + w2·Z_B)) − recon_SI-SDR
        MS-STFT ratio = MS-STFT(D(w1·Z_A + w2·Z_B), w1·A + w2·B) / recon_MS-STFT
    where recon_* is the mean over the two endpoint single-clip
    reconstructions (same pairs, internally consistent — same convention as
    ``mix_recon_ratio.py``).

Interpolation laws (interior t ∈ [0.1, 0.9]; w1(t) = w2(1−t) by symmetry):

  1. logarithmic / audio fader
        w2(t) = log10(1 + 9t)              w1(t) = log10(1 + 9(1−t))
  2. parabolic / quadratic
        w2(t) = t²                         w1(t) = (1−t)²
  3. exponential
        w2(t) = (10^(2t) − 1) / 99         w1(t) = (10^(2(1−t)) − 1) / 99

Note these laws are NOT convex (w1 + w2 ≠ 1) and not energy-normalised
(w1² + w2² ≠ 1) in general — exactly the out-of-distribution weight pairs the
homomorphism-cone hypothesis must generalise to.

Usage:
    python toy_evaluation/generalized_interpolation_eval.py \
        --checkpoints toy_training/checkpoints/transformer_best_full_best.pt \
                       toy_training/checkpoints/transformer_minimal_full_best.pt \
        --num_pairs 100 --seed 42 --device cuda:3

Outputs:
    toy_evaluation/generalized_interpolation_summary.json
    (plus a Markdown comparison table printed to stdout)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss

CKPTS = [
    "toy_training/checkpoints/transformer_best_full_best.pt",     # Norm-Preserving / Full Mixup
    "toy_training/checkpoints/transformer_minimal_full_best.pt",  # Minimal / Unregularized baseline
]

T_MIN, T_MAX = 0.1, 0.9  # interior grid — never touch the pure-endpoint t ∈ {0, 1}


# =============================================================================
#  Interpolation laws (each returns (w1, w2) for a given t)
# =============================================================================

def law_logarithmic(t: float) -> tuple[float, float]:
    """Audio-fader law: log10-scaled crossfade, w(t) = w(1−t)."""
    w2 = float(np.log10(1.0 + 9.0 * t))
    w1 = float(np.log10(1.0 + 9.0 * (1.0 - t)))
    return w1, w2


def law_parabolic(t: float) -> tuple[float, float]:
    """Quadratic law: w2 = t², w1 = (1−t)²."""
    w1 = (1.0 - t) ** 2
    w2 = t ** 2
    return float(w1), float(w2)


def law_exponential(t: float) -> tuple[float, float]:
    """Exponential law: w2 = (10^(2t) − 1)/99, w1 = (10^(2(1−t)) − 1)/99."""
    w2 = (10.0 ** (2.0 * t) - 1.0) / 99.0
    w1 = (10.0 ** (2.0 * (1.0 - t)) - 1.0) / 99.0
    return float(w1), float(w2)


LAWS = {
    "logarithmic": law_logarithmic,
    "parabolic": law_parabolic,
    "exponential": law_exponential,
}


# =============================================================================
#  Metrics (identical conventions to toy_evaluation/mix_recon_ratio.py)
# =============================================================================

def ms_stft_of(pred_np: np.ndarray, tgt_np: np.ndarray, loss_fn, device) -> float:
    pred_t = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, mag = loss_fn(pred_t, tgt_t, use_phase=False, return_components=True)
    return float(mag.item())


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019). Higher = better."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = np.dot(ref, ref)
    if denom < 1e-12:
        return float("nan")
    alpha = np.dot(est, ref) / denom
    e_target = alpha * ref
    e_noise = est - e_target
    snr = np.dot(e_target, e_target) / (np.dot(e_noise, e_noise) + 1e-12)
    return float(10.0 * np.log10(snr))


def encoder_homomorphism_mse(z_mix_enc: torch.Tensor, z_mix_lin: torch.Tensor) -> float:
    """MSE(Enc(w1·A + w2·B), w1·Enc(A) + w2·Enc(B)) over the full latent."""
    return float(F.mse_loss(z_mix_enc, z_mix_lin).item())


# =============================================================================
#  Per-checkpoint evaluation
# =============================================================================

def evaluate_checkpoint(
    ckpt_path: str,
    device: torch.device,
    stems,
    pairs,
    clip_samples: int,
    loss_fn,
    t_grid: np.ndarray,
    label: str | None = None,
) -> dict:
    """Run the generalized-interpolation battery for ONE checkpoint.

    Returns a dict keyed by law name; each law entry contains per-t mean
    curves (encoder homomorphism MSE, ΔSI-SDR, MS-STFT ratio), the midpoint
    and mean aggregates, path flatness, and per-pair records for CI work.
    """
    model = ToyModelWrapper(ckpt_path, device)
    name = label or Path(ckpt_path).stem
    n_t = len(t_grid)
    print(f"=== {name} ===")
    print(f"    model_type={model.model_type}  device={device}  pairs={len(pairs)}  t-grid={t_grid.tolist()}")

    # accumulators: law -> per-(pair, t) arrays
    acc = {law: {
        "enc_mse": np.full((len(pairs), n_t), np.nan),
        "mix_sisdr": np.full((len(pairs), n_t), np.nan),
        "mix_msstft": np.full((len(pairs), n_t), np.nan),
    } for law in LAWS}
    recon_sisdr = np.full((len(pairs),), np.nan)
    recon_msstft = np.full((len(pairs),), np.nan)

    for p_idx, (i, j) in enumerate(pairs):
        a = stem_to_numpy(stems[i][0], clip_samples, centre=True)
        b = stem_to_numpy(stems[j][0], clip_samples, centre=True)
        n = len(a)
        z_a = model.encode(a)
        z_b = model.encode(b)
        if z_a.shape[-1] != z_b.shape[-1]:
            mlen = min(z_a.shape[-1], z_b.shape[-1])
            z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]

        # ---- endpoint reconstruction fidelity (same clips as the mixes) ----
        ra = model.decode(z_a, n)
        rb = model.decode(z_b, n)
        recon_sisdr[p_idx] = float(np.nanmean([
            si_sdr(a, ra), si_sdr(b, rb),
        ]))
        recon_msstft[p_idx] = float(np.nanmean([
            ms_stft_of(ra, a, loss_fn, device),
            ms_stft_of(rb, b, loss_fn, device),
        ]))

        for law, fn in LAWS.items():
            for t_idx, t in enumerate(t_grid):
                w1, w2 = fn(t)
                ideal = w1 * a + w2 * b                      # ideal acoustic weighted mix
                z_lin = w1 * z_a + w2 * z_b                  # RHS: weighted combination of encodings
                z_enc = model.encode(ideal)                  # LHS: encode the ideal mix
                if z_enc.shape[-1] != z_lin.shape[-1]:
                    mlen = min(z_enc.shape[-1], z_lin.shape[-1])
                    z_enc, z_lin = z_enc[..., :mlen], z_lin[..., :mlen]
                acc[law]["enc_mse"][p_idx, t_idx] = encoder_homomorphism_mse(z_enc, z_lin)

                audio_mix = model.decode(z_lin, n)           # decoded latent weighted mix
                acc[law]["mix_sisdr"][p_idx, t_idx] = si_sdr(ideal, audio_mix)
                acc[law]["mix_msstft"][p_idx, t_idx] = ms_stft_of(audio_mix, ideal, loss_fn, device)

        if (p_idx + 1) % 25 == 0 or p_idx == len(pairs) - 1:
            print(f"    pair {p_idx + 1}/{len(pairs)}  "
                  f"recon-SI-SDR {np.nanmean(recon_sisdr[:p_idx + 1]):.2f} dB")

    # ---- aggregate per law ----
    def _nanmean(x):
        return float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else float("nan")

    t_mid = float(t_grid[np.argmin(np.abs(t_grid - 0.5))])

    results = {}
    for law in LAWS:
        enc_curve = np.nanmean(acc[law]["enc_mse"], axis=0)          # per-t mean over pairs
        enc_curve_std = np.nanstd(acc[law]["enc_mse"], axis=0)
        d_sisdr = np.nanmean(acc[law]["mix_sisdr"] - recon_sisdr[:, None], axis=0)
        d_sisdr_std = np.nanstd(acc[law]["mix_sisdr"] - recon_sisdr[:, None], axis=0)
        mix_ms = acc[law]["mix_msstft"]
        # per-(pair,t) ratio uses the per-pair recon ceiling (internally consistent)
        ms_ratio_pt = mix_ms / recon_msstft[:, None]
        ms_ratio_curve = np.nanmean(ms_ratio_pt, axis=0)
        ms_ratio_curve_std = np.nanstd(ms_ratio_pt, axis=0)

        mid_idx = int(np.argmin(np.abs(t_grid - t_mid)))
        results[law] = {
            "law_definition": _LAW_DESCRIPTIONS[law],
            "t_grid": t_grid.tolist(),
            "weights_at_midpoint": {"w1": float(LAWS[law](t_mid)[0]), "w2": float(LAWS[law](t_mid)[1])},
            # ---- encoder homomorphism ----
            "enc_mse_midpoint": float(enc_curve[mid_idx]),
            "enc_mse_mean": float(np.nanmean(enc_curve)),
            "enc_mse_std": float(np.nanstd(enc_curve)),
            "enc_mse_path_flatness_std": float(np.nanstd(enc_curve)),
            "enc_mse_per_t_mean": enc_curve.tolist(),
            "enc_mse_per_t_std": enc_curve_std.tolist(),
            # ---- decoder mixing quality ----
            "delta_sisdr_mean": float(np.nanmean(d_sisdr)),
            "delta_sisdr_per_t_mean": d_sisdr.tolist(),
            "delta_sisdr_per_t_std": d_sisdr_std.tolist(),
            "mix_sisdr_mean": _nanmean(acc[law]["mix_sisdr"]),
            "mix_msstft_mean": _nanmean(mix_ms),
            "msstft_ratio_of_means": (
                float(np.nanmean(mix_ms) / np.nanmean(recon_msstft))
                if np.nanmean(recon_msstft) else float("nan")
            ),
            "msstft_mean_of_ratios": _nanmean(ms_ratio_pt),
            "msstft_ratio_per_t_mean": ms_ratio_curve.tolist(),
            "msstft_ratio_per_t_std": ms_ratio_curve_std.tolist(),
            # ---- per-pair records (CI / significance work) ----
            "per_pair": {
                "enc_mse": acc[law]["enc_mse"].tolist(),
                "mix_sisdr": acc[law]["mix_sisdr"].tolist(),
                "mix_msstft": mix_ms.tolist(),
                "recon_sisdr": recon_sisdr.tolist(),
                "recon_msstft": recon_msstft.tolist(),
            },
        }
        print(f"  [{law:12s}] enc-MSE@0.5={results[law]['enc_mse_midpoint']:.4f} "
              f"mean={results[law]['enc_mse_mean']:.4f} flat={results[law]['enc_mse_path_flatness_std']:.4f} | "
              f"ΔSI-SDR={results[law]['delta_sisdr_mean']:+.2f} dB | "
              f"MS-STFT ratio={results[law]['msstft_ratio_of_means']:.3f}")

    return {"name": name, "checkpoint": ckpt_path, "results": results}


_LAW_DESCRIPTIONS = {
    "logarithmic": "w2=log10(1+9t), w1=log10(1+9(1-t))  [audio fader]",
    "parabolic": "w2=t^2, w1=(1-t)^2",
    "exponential": "w2=(10^(2t)-1)/99, w1=(10^(2(1-t))-1)/99",
}


# =============================================================================
#  Markdown table
# =============================================================================

def print_markdown_table(model_results: list[dict], t_mid: float) -> None:
    print("\n\n## Generalized latent interpolation — summary (t ∈ [0.1, 0.9], 9 interior points)\n")
    header = ("| Law | Model | Enc MSE @t=0.5 | Enc MSE mean | Path flatness (std) | "
              "ΔSI-SDR (dB) | MS-STFT ratio |")
    sep = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for mr in model_results:
        for law in LAWS:
            r = mr["results"][law]
            print(f"| {law} | `{mr['name']}` | {r['enc_mse_midpoint']:.4f} | "
                  f"{r['enc_mse_mean']:.4f} | {r['enc_mse_path_flatness_std']:.4f} | "
                  f"{r['delta_sisdr_mean']:+.2f} | {r['msstft_ratio_of_means']:.3f} |")
    print()


# =============================================================================
#  CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generalized latent interpolation laws (log / parabolic / exp) "
                    "— encoder homomorphism cone + decoder mixing quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS,
                    help="Checkpoints to evaluate (default: best/full-mixup + minimal/baseline).")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional display labels for the checkpoints (same order).")
    ap.add_argument("--num_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=9,
                    help="Number of interior t points on [0.1, 0.9] (default: 9).")
    ap.add_argument("--device", type=str, default="cuda:3")
    ap.add_argument("--json", type=Path, default=_EVAL_DIR / "generalized_interpolation_summary.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t_grid = np.linspace(T_MIN, T_MAX, args.steps)

    stems = collect_validation_stems()
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)
    loss_fn = SAMELoss().to(device)
    print(f"pairs={len(pairs)} seed={args.seed} device={device} steps={args.steps} "
          f"t ∈ [{T_MIN}, {T_MAX}]")
    print("laws:", ", ".join(LAWS))

    labels = args.labels or [None] * len(args.checkpoints)
    model_results = []
    for ckpt, label in zip(args.checkpoints, labels):
        model_results.append(evaluate_checkpoint(
            ckpt, device, stems, pairs, clip_samples, loss_fn, t_grid, label=label,
        ))

    t_mid = float(t_grid[np.argmin(np.abs(t_grid - 0.5))])
    out = {
        "description": "Generalized latent interpolation laws (log / parabolic / exp) "
                       "— positive-weight homomorphism cone vs memorised trig/lerp curves",
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "seed": args.seed,
        "device": str(device),
        "t_grid": t_grid.tolist(),
        "laws": _LAW_DESCRIPTIONS,
        "metrics": {
            "enc_mse": "MSE(Enc(w1 A + w2 B), w1 Enc(A) + w2 Enc(B)) — lower = more homomorphic",
            "enc_mse_path_flatness_std": "std of mean enc-MSE across t — lower = flatter path",
            "delta_sisdr": "mean(SI-SDR(mix) - SI-SDR(recon)) over pairs and t — positive = mixing beats reconstruction",
            "msstft_ratio_of_means": "mean(mix MS-STFT)/mean(recon MS-STFT) — <1 = mixing as good as reconstruction",
        },
        "models": model_results,
    }
    args.json.write_text(json.dumps(out, indent=2))
    print(f"\n📄 JSON -> {args.json}")

    print_markdown_table(model_results, t_mid)


if __name__ == "__main__":
    main()
