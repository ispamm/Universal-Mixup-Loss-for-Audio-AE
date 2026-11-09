#!/usr/bin/env python3
"""
AE — evaluate_all: full evaluation battery over arbitrary checkpoints
==========================================================================

Runs a fixed suite of evaluation protocols over ANY number of checkpoints and
emits a per-checkpoint summary plus a comparison table, so that the whole
checkpoint lineage (baseline / minimal / norm / softMax / hparam winners / …)
can be compared on identical evaluation sets.

Protocols included (each with a single-checkpoint core in its own script):

  1. Reconstruction fidelity         → eval_reconstruction.py
       multi-resolution STFT reconstruction loss (MS-STFT, SAME mag-loss —
       the exact training-time spectral objective)
  2. Encoder homomorphism            → latent_distance.py
       MSE of interp(E(A),E(B)) vs E(mix)  (Lerp, Trig; Slerp retained in the
       summary curves), plus trigMix norm-preservation error and the
       Lerp/Trig interpolation path lengths
  3. Decoder additivity (LIMS)       → lims_test.py
       LSD and energy-drop curves of decoded latent interps, plus trigMix
       LSD-smoothness / energy-flatness curve-level metrics
  4. Latent topology                 → latent_topology.py
       latent density, latent convexity (Lerp/Trig midpoints), latent
       anisotropy (covariance singular-value ratio)

The suite is deliberately focused on the **latent-space architecture**: how
the trigMix additivity loss shapes geometry (density / anisotropy /
convexity), homomorphism (encoder MSE), constant-power behaviour (norm
preservation, energy flatness) and interpolation quality (decoder LSD,
path lengths).  Layer-by-layer and attention-level probes are available
standalone (``layer_linearity_eval.py``, ``attention_map_eval.py``) but are
deliberately NOT part of this battery — they are too granular / too
architecture-specific for the cross-checkpoint comparison.

All protocols evaluate on the SAME held-out validation set (the exact
train/val split of ``training/train.py`` over VCTK + FSD50K +
FMA-small, loaded from the preprocessed 16 kHz cache) and the SAME
deterministic pairs / clip indices, so cross-checkpoint numbers are directly
comparable.

Usage
-----
    python evaluation/evaluate_all.py \\
        --checkpoints training/checkpoints/transformer_best_full_best.pt \\
                      training/checkpoints/transformer_minimal_full_best.pt \\
                      training/checkpoints/hparam_search/c-adv005-fine_best.pt \\
        --num_pairs 100 --device cuda:0

Outputs (in ``--output_dir``, default ``evaluation/eval_all``)
-------------------------------------------------------------------
  eval_all_summary.json      full per-checkpoint results (all curves)
  eval_all_comparison.md     markdown table, checkpoint × headline metric
  eval_all_comparison.csv    same table as CSV
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import (
    CLIP_DURATION,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIRS,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
)
from evaluation.model_utils import detect_model_type

# Import the per-protocol cores.
from evaluation.eval_reconstruction import evaluate_reconstruction_checkpoint
from evaluation.latent_distance import evaluate_latent_distance_checkpoint
from evaluation.lims_test import evaluate_lims_checkpoint
from evaluation.latent_topology import evaluate_topology_checkpoint


# =============================================================================
#  Helpers
# =============================================================================

def _val_at_alpha(curve: List[float], alpha: float = 0.5) -> Optional[float]:
    """Return the curve value closest to *alpha* (or None for empty curves)."""
    if not curve:
        return None
    arr = np.asarray(curve, dtype=np.float64)
    if np.isnan(arr).all():
        return None
    # Assumes a uniform α grid: nearest index to alpha.
    n = len(arr)
    idx = int(round(alpha * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return float(arr[idx])


def _safe(fn, *args, **kwargs):
    """Run *fn*, returning (result, error) — never raises."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 — a failing protocol must not kill the run
        return None, str(exc)


# =============================================================================
#  Main
# =============================================================================

def run_battery(
    checkpoint_paths: List[str],
    stems,
    device: torch.device,
    *,
    num_pairs: int = 100,
    pair_mode: str = "random",
    num_clips: int = 256,
    num_mixups: int = 512,
    num_recon_clips: int = 200,
    seed: int = 42,
    sample_rate: int = MODEL_SAMPLE_RATE,
    save_audio: bool = False,
    audio_root: Optional[Path] = None,
) -> Dict:
    """Run the battery over every checkpoint.

    Pairs / clip indices are generated ONCE from *stems* and reused for every
    checkpoint, so all models are compared on identical evaluation data.
    """
    results: Dict[str, Dict] = {}

    pairs = generate_pairs(stems, num_pairs, seed=seed, mode=pair_mode)
    rng = random.Random(seed)
    n_avail = len(stems)
    recon_indices = rng.sample(range(n_avail), min(num_recon_clips, n_avail))
    topo_indices = rng.sample(range(n_avail), min(num_clips, n_avail))

    print(f"[Battery] {len(stems)} validation stems, {len(pairs)} LIMS/latent "
          f"pairs, {len(recon_indices)} recon clips, {len(topo_indices)} topo clips")

    for ckpt in checkpoint_paths:
        name = Path(ckpt).stem
        print(f"\n{'='*80}\n  Checkpoint: {ckpt}\n{'='*80}")
        per_ckpt: Dict = {"checkpoint": ckpt, "model_type": None,
                          "protocols": {}, "errors": {}}

        try:
            model_type = detect_model_type(ckpt, device)
        except Exception as exc:
            per_ckpt["errors"]["detect"] = str(exc)
            results[name] = per_ckpt
            print(f"  ⚠ Failed to detect model type: {exc}")
            continue
        per_ckpt["model_type"] = model_type
        print(f"  Model type: {model_type}")

        # ---- 1. Reconstruction fidelity ----
        print("\n[1/4] Reconstruction fidelity …")
        res, err = _safe(
            evaluate_reconstruction_checkpoint, ckpt, stems, device,
            num_clips=len(recon_indices), seed=seed, model_type=model_type,
            sample_rate=sample_rate, clip_duration=CLIP_DURATION,
            clip_indices=recon_indices,
            save_wav_dir=(audio_root / name / "recon") if save_audio else None,
        )
        if err:
            per_ckpt["errors"]["reconstruction"] = err
            print(f"  ⚠ {err}")
        else:
            per_ckpt["protocols"]["reconstruction"] = res

        # ---- 2. Encoder homomorphism ----
        print("\n[2/4] Encoder homomorphism (latent distance) …")
        res, err = _safe(
            evaluate_latent_distance_checkpoint, ckpt, stems, device,
            num_pairs=len(pairs), seed=seed, pair_mode=pair_mode,
            model_type=model_type, sample_rate=sample_rate,
            clip_duration=CLIP_DURATION, pairs=pairs,
        )
        if err:
            per_ckpt["errors"]["latent_distance"] = err
            print(f"  ⚠ {err}")
        else:
            per_ckpt["protocols"]["latent_distance"] = res

        # ---- 3. Decoder additivity (LIMS) ----
        print("\n[3/4] Decoder additivity (LIMS) …")
        res, err = _safe(
            evaluate_lims_checkpoint, ckpt, stems, device,
            num_pairs=len(pairs), seed=seed, pair_mode=pair_mode,
            model_type=model_type, sample_rate=sample_rate,
            clip_duration=CLIP_DURATION, pairs=pairs,
            save_wav_dir=(audio_root / name / "lims") if save_audio else None,
        )
        if err:
            per_ckpt["errors"]["lims"] = err
            print(f"  ⚠ {err}")
        else:
            per_ckpt["protocols"]["lims"] = res

        # ---- 4. Latent topology ----
        print("\n[4/4] Latent topology (density / convexity / anisotropy) …")
        res, err = _safe(
            evaluate_topology_checkpoint, ckpt, stems, device,
            num_clips=len(topo_indices), num_mixups=num_mixups, seed=seed,
            model_type=model_type, sample_rate=sample_rate,
            clip_duration=CLIP_DURATION, clip_indices=topo_indices,
        )
        if err:
            per_ckpt["errors"]["topology"] = err
            print(f"  ⚠ {err}")
        else:
            per_ckpt["protocols"]["topology"] = res

        results[name] = per_ckpt

    return results


# =============================================================================
#  Comparison table builders
# =============================================================================

# (column, protocol key, extractor) — extractor(res) → float | None
#
# The comparison table is deliberately focused on the trigMix architectural
# narrative:
#   * baseline reconstruction sanity check,
#   * encoder homomorphism for trigMix vs lerp (MSE captures angle AND scale),
#   * end-to-end interpolation quality (decoder LSD) and constant-power
#     energy preservation for trigMix,
#   * latent-space geometry (density, convexity, anisotropy),
#   * trigMix-specific probes (norm preservation, path length, curve
#     smoothness / flatness).
TABLE_COLUMNS = [
    ("model_type", None, None),
    # ---- Baseline reconstruction quality (MS-STFT, sanity check) ----
    ("recon_msstft", "reconstruction",
     lambda r: r.get("mean_msstft")),
    # ---- Encoder homomorphism (trigMix vs lerp trade-off) ----
    ("enc_trig_mse@0.5", "latent_distance",
     lambda r: _val_at_alpha(r.get("mean_trig_mse"))),
    ("enc_lerp_mse@0.5", "latent_distance",
     lambda r: _val_at_alpha(r.get("mean_lerp_mse"))),
    # ---- Decoder additivity (LIMS) ----
    ("dec_trig_lsd@0.5", "lims",
     lambda r: _val_at_alpha(r.get("mean_trig_lsd"))),
    ("dec_lerp_lsd@0.5", "lims",
     lambda r: _val_at_alpha(r.get("mean_lerp_lsd"))),
    ("dec_trig_energy@0.5", "lims",
     lambda r: _val_at_alpha(r.get("mean_trig_energy"))),
    # ---- Latent space geometry ----
    ("latent_density", "topology",
     lambda r: r.get("latent_density")),
    ("convexity_trig", "topology",
     lambda r: (r.get("latent_convexity") or {}).get("trig")),
    ("convexity_lerp", "topology",
     lambda r: (r.get("latent_convexity") or {}).get("lerp")),
    # ---- trigMix-specific probes ----
    ("norm_preservation_error@0.5", "latent_distance",
     lambda r: _val_at_alpha(r.get("mean_abs_norm_preservation"))),
    ("path_length_trig", "latent_distance",
     lambda r: r.get("mean_path_length_trig")),
    ("path_length_lerp", "latent_distance",
     lambda r: r.get("mean_path_length_lerp")),
    ("latent_anisotropy", "topology",
     lambda r: r.get("latent_anisotropy")),
    ("lsd_curve_smoothness_trig", "lims",
     lambda r: r.get("lsd_curve_smoothness_trig")),
    ("energy_curve_flatness_trig", "lims",
     lambda r: r.get("energy_curve_flatness_trig")),
]


def build_rows(results: Dict[str, Dict]) -> List[Dict]:
    """Flatten per-checkpoint results into table rows keyed by column name."""
    rows: List[Dict] = []
    for name, per_ckpt in results.items():
        row: Dict = {"checkpoint": name}
        for col, proto, extractor in TABLE_COLUMNS:
            if proto is None:
                row[col] = per_ckpt.get("model_type")
                continue
            res = per_ckpt.get("protocols", {}).get(proto)
            row[col] = extractor(res) if res is not None else None
        rows.append(row)
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1e-3:
            return f"{v:.4f}"
        return f"{v:.3e}"
    return str(v)


def render_markdown(rows: List[Dict]) -> str:
    cols = ["checkpoint"] + [c for c, _, _ in TABLE_COLUMNS]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: List[Dict]) -> str:
    import csv
    import io
    cols = ["checkpoint"] + [c for c, _, _ in TABLE_COLUMNS]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for row in rows:
        writer.writerow(["" if row.get(c) is None else row.get(c) for c in cols])
    return buf.getvalue()


# =============================================================================
#  CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AE — full evaluation battery over arbitrary checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoints", required=True, nargs="+",
                   help="One or more checkpoint paths, e.g. "
                        "training/checkpoints/transformer_best_full_best.pt "
                        "training/checkpoints/transformer_minimal_full_best.pt")
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                   nargs="+", default=None,
                   help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                        "training set used by train.py.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Preprocessed cache base dir "
                        "(default: /data/preprocessed_cache).")
    p.add_argument("--num_pairs", type=int, default=100,
                   help="Pairs for LIMS + latent-distance (default: 100).")
    p.add_argument("--pair_mode", type=str, default="random",
                   choices=["random", "intra", "inter"],
                   help="Pair mode (default: random).")
    p.add_argument("--num_clips", type=int, default=256,
                   help="Clips for latent topology (default: 256).")
    p.add_argument("--num_mixups", type=int, default=512,
                   help="Interpolations for convexity (default: 512).")
    p.add_argument("--num_recon_clips", type=int, default=200,
                   help="Clips for reconstruction fidelity (default: 200).")
    p.add_argument("--save-audio", action="store_true", default=False,
                   help="Save LIMS interpolations and reconstructions per "
                        "checkpoint under the output dir.")
    p.add_argument("--device", default="cuda",
                   help="Device (default: cuda).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--sample_rate", type=int, default=MODEL_SAMPLE_RATE,
                   help=f"Target sample rate in Hz (default: {MODEL_SAMPLE_RATE}).")
    p.add_argument("--output_dir", default="evaluation/eval_all",
                   help="Output directory (default: evaluation/eval_all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Data dirs] {args.data_dirs or DEFAULT_DATA_DIRS}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root = out_dir / "audio" if args.save_audio else None

    # ---- Shared validation stems (exact training split over all datasets) ----
    print("[Setup] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(args.data_dirs, args.cache_dir,
                                     sample_rate=args.sample_rate)
    print(f"[Setup] {len(stems)} validation stems")

    # ---- Run battery ----
    results = run_battery(
        args.checkpoints, stems, device,
        num_pairs=args.num_pairs, pair_mode=args.pair_mode,
        num_clips=args.num_clips, num_mixups=args.num_mixups,
        num_recon_clips=args.num_recon_clips,
        seed=args.seed, sample_rate=args.sample_rate,
        save_audio=args.save_audio, audio_root=audio_root,
    )

    # ---- Build tables ----
    rows = build_rows(results)
    md = render_markdown(rows)
    csv_text = render_csv(rows)

    summary = {
        "description": "AE evaluation battery — full comparison",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "data_dirs": list(args.data_dirs) if args.data_dirs else DEFAULT_DATA_DIRS,
        "num_pairs": args.num_pairs,
        "num_clips": args.num_clips,
        "num_mixups": args.num_mixups,
        "num_recon_clips": args.num_recon_clips,
        "seed": args.seed,
        "checkpoints": {k: v for k, v in results.items()},
    }

    json_path = out_dir / "eval_all_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n📄 JSON → {json_path}")

    md_path = out_dir / "eval_all_comparison.md"
    md_path.write_text(md)
    print(f"📄 Markdown table → {md_path}")

    csv_path = out_dir / "eval_all_comparison.csv"
    csv_path.write_text(csv_text)
    print(f"📄 CSV table → {csv_path}")

    # ---- Print the comparison table ----
    print("\n" + "=" * 100)
    print("  Comparison table")
    print("=" * 100)
    print(md)
    print("Legend: enc = encoder homomorphism MSE, dec = decoder additivity "
          "(LIMS) LSD/energy,")
    print("convexity = NN-distance of interp midpoints (lower = more filled-in),")
    print("norm_preservation_error = |‖trigMix(Z)‖ − ‖Z_A‖| at α=0.5 (0 = "
          "constant-power),")
    print("path_length = chord-sum of the interpolation path (shorter = more "
          "direct),")
    print("latent_anisotropy = σ_max/σ_min of latent covariance (1.0 = "
          "isotropic),")
    print("lsd_curve_smoothness / energy_curve_flatness = consistency of trigMix "
          "quality across α.")
    print("— = protocol N/A or failed for that checkpoint (see JSON 'errors').")


if __name__ == "__main__":
    main()
