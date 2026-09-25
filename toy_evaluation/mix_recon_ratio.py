#!/usr/bin/env python3
"""
Mix-vs-reconstruction ratio study (internally consistent, same pairs)
=====================================================================

For the SAME 100 pairs (seed 42) used by the LIMS battery, computes at α=0.5:

  * mix completeness:  LSD / LSD-active / MS-STFT of
        D(trigMix(Z_A,Z_B)) vs ideal trigMix audio, and the lerp analogue,
  * reconstruction fidelity of the pair endpoints (same model, same clips):
        recon LSD / LSD-active / energy-weighted LSD / MS-STFT
        (mean of the two endpoints),

so the "mix / recon" RATIO is computed on identical data — no mixing of
protocols, and no silence-inflated reconstruction LSD in the denominator
(plain LSD is reported alongside for reference).

Usage:
    python toy_evaluation/mix_recon_ratio.py \
        --checkpoints toy_training/checkpoints/transformer_*_full_best.pt \
        --num_pairs 100 --seed 42 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

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
from toy_evaluation.lims_test_toy import (
    compute_lsd,
    compute_lsd_decomposed,
    lerp,
    trig_mix,
)
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss

CKPTS = [
    "toy_training/checkpoints/transformer_best_full_best.pt",
    "toy_training/checkpoints/transformer_norm_full_best.pt",
    "toy_training/checkpoints/transformer_softMax_full_best.pt",
    "toy_training/checkpoints/transformer_minimal_full_best.pt",
]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--num_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--json", type=Path, default=_EVAL_DIR / "mix_recon_ratio_summary.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stems = collect_validation_stems()
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)
    loss_fn = SAMELoss().to(device)
    print(f"pairs={len(pairs)} seed={args.seed} device={device}")

    results = {}
    for ckpt in args.checkpoints:
        name = Path(ckpt).stem
        print(f"\n=== {name} ===")
        model = ToyModelWrapper(ckpt, device)

        mix_trig_lsd, mix_trig_act, mix_trig_ew, mix_trig_ms = [], [], [], []
        mix_lerp_lsd, mix_lerp_act, mix_lerp_ew, mix_lerp_ms = [], [], [], []
        mix_trig_sisdr, mix_lerp_sisdr = [], []
        recon_lsd, recon_act, recon_ew, recon_ms, recon_sisdr = [], [], [], [], []
        recon_sparsity = []
        per_pair = []  # one dict per pair for CI computation

        for (i, j) in pairs:
            a = stem_to_numpy(stems[i][0], clip_samples, centre=True)
            b = stem_to_numpy(stems[j][0], clip_samples, centre=True)
            n = len(a)
            z_a = model.encode(a)
            z_b = model.encode(b)

            # --- endpoint reconstruction (same clips as the mix) ---
            ra = model.decode(z_a, n)
            rb = model.decode(z_b, n)
            pair_recon = {"lsd": [], "act": [], "ew": [], "ms": [], "sp": [], "sisdr": []}
            for audio, recon in ((a, ra), (b, rb)):
                d = compute_lsd_decomposed(audio, recon)
                pair_recon["lsd"].append(d["lsd"])
                pair_recon["act"].append(d["lsd_active"])
                pair_recon["ew"].append(d["lsd_energy_weighted"])
                pair_recon["sp"].append(d["sparsity"])
                pair_recon["ms"].append(ms_stft_of(recon, audio, loss_fn, device))
                pair_recon["sisdr"].append(si_sdr(audio, recon))
            # per-pair values = mean over the two endpoints (same scale as mix)
            recon_lsd.append(np.mean(pair_recon["lsd"]))
            recon_act.append(np.mean(pair_recon["act"]))
            recon_ew.append(np.mean(pair_recon["ew"]))
            recon_sparsity.append(np.mean(pair_recon["sp"]))
            recon_ms.append(np.mean(pair_recon["ms"]))
            recon_sisdr.append(np.mean(pair_recon["sisdr"]))

            # --- trig mix @ α=0.5 ---
            ideal_trig = trig_mix(torch.from_numpy(a), torch.from_numpy(b), 0.5).numpy()
            z_trig = trig_mix(z_a, z_b, 0.5)
            audio_trig = model.decode(z_trig, n)
            td = compute_lsd_decomposed(ideal_trig, audio_trig)
            mix_trig_lsd.append(td["lsd"])
            mix_trig_act.append(td["lsd_active"])
            mix_trig_ew.append(td["lsd_energy_weighted"])
            mix_trig_ms.append(ms_stft_of(audio_trig, ideal_trig, loss_fn, device))
            mix_trig_sisdr.append(si_sdr(ideal_trig, audio_trig))

            # --- lerp mix @ α=0.5 ---
            ideal_lerp = 0.5 * a + 0.5 * b
            z_lerp = lerp(z_a, z_b, 0.5)
            audio_lerp = model.decode(z_lerp, n)
            ld = compute_lsd_decomposed(ideal_lerp, audio_lerp)
            mix_lerp_lsd.append(ld["lsd"])
            mix_lerp_act.append(ld["lsd_active"])
            mix_lerp_ew.append(ld["lsd_energy_weighted"])
            mix_lerp_ms.append(ms_stft_of(audio_lerp, ideal_lerp, loss_fn, device))
            mix_lerp_sisdr.append(si_sdr(ideal_lerp, audio_lerp))

            per_pair.append({
                "mix_trig_lsd_active": mix_trig_act[-1],
                "mix_trig_msstft": mix_trig_ms[-1],
                "mix_trig_sisdr": mix_trig_sisdr[-1],
                "mix_lerp_lsd_active": mix_lerp_act[-1],
                "mix_lerp_msstft": mix_lerp_ms[-1],
                "mix_lerp_sisdr": mix_lerp_sisdr[-1],
                "recon_lsd_active": recon_act[-1],
                "recon_msstft": recon_ms[-1],
                "recon_sisdr": recon_sisdr[-1],
            })

        def m(x):
            return float(np.mean(x))

        # Ratio of MEANS (standard for LSD tables) and mean of per-pair ratios.
        def ratios(mix, recon):
            return {
                "ratio_of_means": m(mix) / m(recon) if m(recon) else float("nan"),
                "mean_of_ratios": m(np.array(mix) / np.array(recon)),
            }

        res = {
            "valid_pairs": len(pairs),
            "mix_trig_lsd": m(mix_trig_lsd),
            "mix_lerp_lsd": m(mix_lerp_lsd),
            "mix_trig_lsd_active": m(mix_trig_act),
            "mix_trig_msstft": m(mix_trig_ms),
            "mix_lerp_msstft": m(mix_lerp_ms),
            "recon_lsd": m(recon_lsd),
            "recon_lsd_active": m(recon_act),
            "recon_lsd_energy_weighted": m(recon_ew),
            "recon_msstft": m(recon_ms),
            "recon_sisdr": m(recon_sisdr),
            "mix_trig_sisdr": m(mix_trig_sisdr),
            "mix_lerp_sisdr": m(mix_lerp_sisdr),
            "delta_sisdr_trig": m(
                np.array(mix_trig_sisdr) - np.array(recon_sisdr)
            ),  # ΔdB = mixing − reconstruction (positive → mixing better)
            "delta_sisdr_lerp": m(
                np.array(mix_lerp_sisdr) - np.array(recon_sisdr)
            ),
            "recon_sparsity": m(recon_sparsity),
            "ratio_trig_lsd_vs_recon_lsd": ratios(mix_trig_lsd, recon_lsd),
            "ratio_trig_lsd_active_vs_recon_lsd_active": ratios(mix_trig_act, recon_act),
            "ratio_trig_msstft_vs_recon_msstft": ratios(mix_trig_ms, recon_ms),
            "ratio_lerp_lsd_active_vs_recon_lsd_active": ratios(mix_lerp_act, recon_act),
            "ratio_lerp_msstft_vs_recon_msstft": ratios(mix_lerp_ms, recon_ms),
            "per_pair": per_pair,
        }
        results[name] = res
        print(f"  trig LSD ratio (LSD-active): {res['ratio_trig_lsd_active_vs_recon_lsd_active']['ratio_of_means']:.3f}"
              f"  (mix {res['mix_trig_lsd_active']:.2f} / recon {res['recon_lsd_active']:.2f})")
        print(f"  trig MS-STFT ratio:          {res['ratio_trig_msstft_vs_recon_msstft']['ratio_of_means']:.3f}"
              f"  (mix {res['mix_trig_msstft']:.3f} / recon {res['recon_msstft']:.3f})")
        print(f"  [ref] trig LSD ratio vs PLAIN recon LSD: {res['ratio_trig_lsd_vs_recon_lsd']['ratio_of_means']:.3f}"
              f"  (mix {res['mix_trig_lsd']:.2f} / recon-plain {res['recon_lsd']:.2f})")

    out = {
        "description": "Mix-vs-reconstruction ratios, internally consistent (same pairs, seed 42)",
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "seed": args.seed,
        "results": results,
    }
    args.json.write_text(json.dumps(out, indent=2))
    print(f"\n📄 JSON -> {args.json}")


if __name__ == "__main__":
    main()
