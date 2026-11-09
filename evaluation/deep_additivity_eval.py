#!/usr/bin/env python3
"""
AE — Deep Additivity Evaluation (layer-wise trigMix commutation + mix-at-any-depth)
=========================================================================================

The trigMix additivity loss trains the *bottleneck* commutation
``E(trigMix(A, B)) ≈ trigMix(E(A), E(B))``.  This experiment asks the deeper
architectural question: **is the model additivity-preserving at EVERY layer of
the encoder and decoder — i.e. non-linear but *deeply* additive?**

Two complementary protocols:

Protocol 1 — Layer-wise trigMix commutation (encoder AND decoder)
-----------------------------------------------------------------
For each pair (A, B) and every intermediate layer l:
    h_mix^(l)  = Layer_l( trigMix_audio(A, B) )     # encode the audio mix
    h_trig^(l) = trigMix( Layer_l(A), Layer_l(B) )  # mix the representations
and measure how well the two commute:

    comm_rel_err = ‖h_mix − trigMix(h_A, h_B)‖_F / ‖h_mix‖_F   (lower = additive)
    comm_mse     = mean((h_mix − trigMix(h_A, h_B))²)           (bottleneck ↔ enc_trig_mse)

For contrast we also report the *direct* algebraic linearity (the old
f(A+B) ≈ f(A)+f(B) probe):
    lin_rel_err = ‖h_mix − (h_A + h_B)‖_F / ‖h_mix‖_F
If ``lin_rel_err ≫ comm_rel_err`` at every layer, the model is **non-linear
but deeply additivity-preserving**: the layers are not linear maps, yet the
trigMix operation commutes with every one of them.

Reading the incremental error: layers decompose into three regimes —
(1) bias-free linear maps (patch/embed projections) commute *exactly*
(≈ 0); (2) affine layers with bias (``enc_to_latent``, ``latent_to_dec``,
soft-norm) have a fixed additive floor ``‖(1−cosθ−sinθ)·b‖`` because trigMix
weights sum to √2 ≠ 1 — small but non-zero; (3) the non-linear transformer
blocks (RoPE attention + leaky-ReLU/snake FFN) dominate the error.  So the
per-layer error is a genuine *additivity budget*: how much of the mix is
preserved when passing through each layer in isolation.

Protocol 2 — "Complete mix at any depth" (end-to-end)
-----------------------------------------------------
For every layer l along the encoder→decoder path, inject
``z_mix = trigMix(Layer_l(A), Layer_l(B))`` back into the network at that
depth (pre-hook on the next module) and decode the rest of the model to
audio.  If the model is deeply additive, the result should be a *complete*
mix regardless of where the mix is injected:

    LSD( decode(rest(z_mix at depth l)), trigMix_audio(A, B) )

The bottleneck injection (after the soft-norm) is exactly the mix the
training loss optimises — it is the reference "canonical" depth.

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``evaluation.data``):
validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.
All models are compared on the SAME deterministic pairs (seed 42).

Usage
-----
    python evaluation/deep_additivity_eval.py \\
        --checkpoints training/checkpoints/transformer_best_full_best.pt \\
                      training/checkpoints/transformer_norm_full_best.pt \\
        --num_pairs 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.data import (
    CLIP_DURATION,
    DEFAULT_DATA_DIRS,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from evaluation.latent_distance import trig_mix
from evaluation.lims_test import compute_lsd
from evaluation.layer_linearity_eval import (
    LayerActivationCapture,
    discover_layers,
)
from evaluation.model_utils import detect_model_type, load_model

# =============================================================================
#  Constants
# =============================================================================

MIX_ALPHA = 0.5  # trigMix α = 0.5  →  cos(π/4)·A + sin(π/4)·B (training headline)


def _trig_audio(audio_a: np.ndarray, audio_b: np.ndarray) -> np.ndarray:
    """Audio-domain trigMix at α = 0.5 (the constant-power mix of training)."""
    theta = MIX_ALPHA * (np.pi / 2.0)
    return np.cos(theta) * audio_a + np.sin(theta) * audio_b


# =============================================================================
#  Injection points (ordered along the forward graph)
# =============================================================================

def build_injection_points(layer_modules) -> List[Tuple[str, str, str, nn.Module]]:
    """Return ``[(display_name, source_name, next_name, next_module)]``.

    ``layer_modules`` is the OrderedDict from ``discover_layers`` (insertion
    order == forward order for both supported architectures).  Each point
    mixes the *source* layer's representation of (A, B) and re-injects it as
    the input of the *next* module via a forward pre-hook.
    """
    names = list(layer_modules.keys())
    points: List[Tuple[str, str, str, nn.Module]] = []
    for k in range(len(names) - 1):
        src, nxt = names[k], names[k + 1]
        points.append((f"after:{src}", src, nxt, layer_modules[nxt]))
    return points


def run_with_injection(
    model: nn.Module,
    layer_modules,
    z_mix: torch.Tensor,
    next_module: nn.Module,
    x: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Run the full model with ``z_mix`` substituted as the input of
    ``next_module`` (forward pre-hook).

    Returns ``(all_layer_outputs, decoded_audio)``: the layer outputs let us
    read off the *injected* forward's result at the target module, which is
    exactly ``Layer_target(trigMix(inputs))`` with a perfectly-commuting input
    — i.e. the isolated (incremental) additivity error of the target layer.
    """
    current: Dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                current[name] = out.detach().clone()
        return hook_fn

    hooks = [m.register_forward_hook(_make_hook(n))
             for n, m in layer_modules.items()]
    handle = next_module.register_forward_pre_hook(
        lambda mod, args: (z_mix.detach(),)
    )
    try:
        with torch.no_grad():
            out = model(x)
    finally:
        for h in hooks:
            h.remove()
        handle.remove()
    return current, out


# =============================================================================
#  Single-checkpoint core
# =============================================================================

def evaluate_deep_additivity_checkpoint(
    ckpt_path: str,
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_pairs: int = 32,
    seed: int = 42,
    pair_mode: str = "random",
    model_type: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    clip_duration: float = CLIP_DURATION,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    """Run both deep-additivity protocols for ONE checkpoint."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_samples = int(clip_duration * sample_rate)

    if model_type is None:
        model_type = detect_model_type(ckpt_path, device)
        print(f"  [Auto-detect] Model type: {model_type}")

    model = load_model(ckpt_path, device, model_type)
    print(f"  Model: {Path(ckpt_path).stem}  ({model_type})")

    layer_modules, layer_info = discover_layers(model, model_type)
    injection_points = build_injection_points(layer_modules)
    capture = LayerActivationCapture(model, layer_modules, device)

    if pairs is None:
        pairs = generate_pairs(stems, num_pairs, seed=seed, mode=pair_mode)
    print(f"  {len(pairs)} pairs, {len(layer_info)} layers, "
          f"{len(injection_points)} injection depths")

    # ---- Accumulators ----
    # Protocol 1: per-layer commutation / linearity
    comm = {name: {"rel": [], "mse": [], "cos": []} for name in layer_modules}
    lin = {name: {"rel": [], "cos": []} for name in layer_modules}
    # Incremental additivity: per-target-layer error given a perfectly
    # commuting (injected) input — filled by Protocol 2's injected forwards.
    incr = {name: {"rel": []} for name in layer_modules}
    # Protocol 2: per-injection-depth end-to-end quality
    inj = {disp: {"lsd": [], "wave_mse": []} for disp, _, _, _ in injection_points}

    for pair_idx, (i, j) in enumerate(tqdm(pairs, desc="Pairs", unit="pair")):
        path_a, _sa, _ta = stems[i]
        path_b, _sb, _tb = stems[j]
        try:
            audio_a = stem_to_numpy(path_a, clip_samples, centre=True)
            audio_b = stem_to_numpy(path_b, clip_samples, centre=True)
        except Exception as exc:
            print(f"  ⚠ Skip pair {pair_idx}: load error – {exc}")
            continue

        audio_mix = _trig_audio(audio_a, audio_b).astype(np.float32)
        orig_len = len(audio_a)
        try:
            acts_a = capture.capture(audio_a)
            acts_b = capture.capture(audio_b)
            acts_mix = capture.capture(audio_mix)
        except Exception as exc:
            print(f"  ⚠ Skip pair {pair_idx}: capture error – {exc}")
            continue

        # =====================================================================
        #  Protocol 1 — layer-wise trigMix commutation
        # =====================================================================
        for name in layer_modules:
            if name not in acts_a or name not in acts_b or name not in acts_mix:
                continue
            h_a = acts_a[name].reshape(-1).float()
            h_b = acts_b[name].reshape(-1).float()
            h_mix = acts_mix[name].reshape(-1).float()

            h_trig = trig_mix(acts_a[name], acts_b[name], MIX_ALPHA).reshape(-1).float()

            norm_mix = torch.norm(h_mix)
            if norm_mix > 1e-12:
                comm[name]["rel"].append(
                    float((torch.norm(h_mix - h_trig) / norm_mix).cpu()))
            comm[name]["mse"].append(
                float(((h_mix - h_trig) ** 2).mean().cpu()))
            comm[name]["cos"].append(float(
                torch.nn.functional.cosine_similarity(
                    h_mix.unsqueeze(0), h_trig.unsqueeze(0)).item()))

            h_lin = h_a + h_b
            if norm_mix > 1e-12:
                lin[name]["rel"].append(
                    float((torch.norm(h_mix - h_lin) / norm_mix).cpu()))
            lin[name]["cos"].append(float(
                torch.nn.functional.cosine_similarity(
                    h_mix.unsqueeze(0), h_lin.unsqueeze(0)).item()))

        # =====================================================================
        #  Protocol 2 — complete mix at any depth (end-to-end)
        # =====================================================================
        dummy = torch.from_numpy(audio_a).unsqueeze(0).unsqueeze(0).to(device)
        for disp, src, nxt_name, nxt in injection_points:
            if src not in acts_a or src not in acts_b:
                continue
            z_mix = trig_mix(acts_a[src], acts_b[src], MIX_ALPHA)
            # The forward graph denormalises the soft-norm latent *before*
            # latent_to_dec (inside decode()); replicate that transform so the
            # injected representation lives in the space the next module expects.
            if src == "soft_norm" and getattr(model, "soft_norm", None) is not None:
                z_mix = model.soft_norm.reverse(z_mix)
            z_mix = z_mix.to(device)
            try:
                acts_inj, out = run_with_injection(
                    model, layer_modules, z_mix, nxt, dummy)
            except Exception as exc:
                print(f"  ⚠ Pair {pair_idx} inject@{disp}: {exc}")
                continue
            audio_out = out.squeeze().cpu().numpy()
            if len(audio_out) != orig_len:
                audio_out = audio_out[:orig_len]
            inj[disp]["lsd"].append(compute_lsd(audio_mix, audio_out))
            inj[disp]["wave_mse"].append(
                float(np.mean((audio_mix[:orig_len] - audio_out[:orig_len]) ** 2)))

            # Incremental additivity of the TARGET layer: its input is now the
            # exactly-commuting injected mix, so its own output error isolates
            # the layer's contribution:
            #   Layer_t(trigMix(in_A, in_B))  vs  trigMix(Layer_t(in_A), Layer_t(in_B))
            if nxt_name in acts_inj and nxt_name in acts_a and nxt_name in acts_b:
                h_inj = acts_inj[nxt_name].reshape(-1).float()
                h_trig = trig_mix(
                    acts_a[nxt_name], acts_b[nxt_name], MIX_ALPHA
                ).reshape(-1).float()
                if torch.norm(h_inj) > 1e-12:
                    incr[nxt_name]["rel"].append(
                        float((torch.norm(h_inj - h_trig) / torch.norm(h_inj)).cpu()))

    # ---- Aggregate: Protocol 1 ----
    def _agg(store: Dict[str, Dict[str, list]], key: str) -> Tuple[List, List]:
        vals = [np.mean(store[n][key]) if store[n][key] else float("nan")
                for n in store]
        stds = [np.std(store[n][key]) if store[n][key] else float("nan")
                for n in store]
        return vals, stds

    layers_out: List[Dict] = []
    for mt, name, idx in layer_info:
        if not comm[name]["rel"]:
            continue
        layers_out.append({
            "module": mt,
            "layer_name": name,
            "layer_idx": idx,
            "n_pairs": len(comm[name]["rel"]),
            "comm_rel_err": float(np.mean(comm[name]["rel"])),
            "comm_rel_err_std": float(np.std(comm[name]["rel"])),
            "comm_mse": float(np.mean(comm[name]["mse"])),
            "comm_cos": float(np.mean(comm[name]["cos"])),
            "incr_comm_rel_err": (float(np.mean(incr[name]["rel"]))
                                  if incr[name]["rel"] else float("nan")),
            "lin_rel_err": float(np.mean(lin[name]["rel"])),
            "lin_rel_err_std": float(np.std(lin[name]["rel"])),
            "lin_cos": float(np.mean(lin[name]["cos"])),
        })

    # ---- Aggregate: Protocol 2 ----
    inj_out: List[Dict] = []
    for disp, src, _nxt_name, _nxt in injection_points:
        lsd = inj[disp]["lsd"]
        if not lsd:
            continue
        wm = inj[disp]["wave_mse"]
        inj_out.append({
            "depth": disp,
            "source_layer": src,
            "is_bottleneck": src in ("soft_norm", "enc_to_latent"),
            "n_pairs": len(lsd),
            "lsd_mean": float(np.mean(lsd)),
            "lsd_std": float(np.std(lsd)),
            "wave_mse_mean": float(np.mean(wm)),
        })

    enc_rows = [r for r in layers_out if r["module"] == "Encoder"]
    dec_rows = [r for r in layers_out if r["module"] == "Decoder"]
    bn_rows = [r for r in layers_out if r["layer_name"] in ("soft_norm", "enc_to_latent")]
    enc_incr = [r["incr_comm_rel_err"] for r in enc_rows
                if not np.isnan(r["incr_comm_rel_err"])]
    dec_incr = [r["incr_comm_rel_err"] for r in dec_rows
                if not np.isnan(r["incr_comm_rel_err"])]

    return {
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "mix_alpha": MIX_ALPHA,
        "num_pairs_requested": len(pairs),
        "layers": layers_out,
        "injections": inj_out,
        "summary": {
            "encoder_comm_rel_err": float(np.mean([r["comm_rel_err"] for r in enc_rows])),
            "decoder_comm_rel_err": float(np.mean([r["comm_rel_err"] for r in dec_rows])),
            "bottleneck_comm_mse": float(np.mean([r["comm_mse"] for r in bn_rows])),
            "encoder_incr_comm_rel_err": float(np.mean(enc_incr)) if enc_incr else float("nan"),
            "decoder_incr_comm_rel_err": float(np.mean(dec_incr)) if dec_incr else float("nan"),
            "encoder_lin_rel_err": float(np.mean([r["lin_rel_err"] for r in enc_rows])),
            "decoder_lin_rel_err": float(np.mean([r["lin_rel_err"] for r in dec_rows])),
            "bottleneck_inject_lsd": float(np.mean(
                [r["lsd_mean"] for r in inj_out if r["is_bottleneck"]])),
            "encoder_inject_lsd": float(np.mean(
                [r["lsd_mean"] for r in inj_out
                 if r["source_layer"].startswith(("patch_embed", "enc_"))])),
            "decoder_inject_lsd": float(np.mean(
                [r["lsd_mean"] for r in inj_out
                 if r["source_layer"].startswith(("latent_to_dec", "dec_"))])),
            "worst_depth_lsd": float(np.max([r["lsd_mean"] for r in inj_out])),
            "best_depth_lsd": float(np.min([r["lsd_mean"] for r in inj_out])),
        },
    }


# =============================================================================
#  Multi-checkpoint runner + tables
# =============================================================================

def print_deep_additivity_tables(results: Dict[str, Dict]) -> None:
    for ck, res in results.items():
        sep = "=" * 110
        print(f"\n{sep}\n  {ck}  —  deep additivity\n{sep}")
        print(f"{'module':<10} {'layer':<16} {'comm_rel':>9} {'incr_rel':>9} "
              f"{'comm_mse':>11} {'comm_cos':>9} | {'lin_rel':>9} {'lin_cos':>9}")
        print("-" * 110)
        for r in res["layers"]:
            idx = "" if r["layer_idx"] < 0 else f"#{r['layer_idx']}"
            incr_s = f"{r['incr_comm_rel_err']:.4f}" if not np.isnan(r["incr_comm_rel_err"]) else "—"
            print(f"{r['module']:<10} {r['layer_name']+idx:<16} "
                  f"{r['comm_rel_err']:>9.4f} {incr_s:>9} "
                  f"{r['comm_mse']:>11.5f} {r['comm_cos']:>9.4f} | "
                  f"{r['lin_rel_err']:>9.4f} {r['lin_cos']:>9.4f}")
        print(f"\n  mix-at-any-depth (LSD vs ideal trigMix audio, lower = more complete mix):")
        print(f"{'depth':<26} {'source':<16} {'LSD(dB)':>10} {'wave_mse':>12}")
        print("-" * 110)
        for r in res["injections"]:
            tag = "  ◀ bottleneck (training mix)" if r["is_bottleneck"] else ""
            print(f"{r['depth']:<26} {r['source_layer']:<16} "
                  f"{r['lsd_mean']:>10.4f} {r['wave_mse_mean']:>12.6f}{tag}")
        s = res["summary"]
        print(f"\n  summary: enc comm {s['encoder_comm_rel_err']:.4f} | "
              f"dec comm {s['decoder_comm_rel_err']:.4f} | "
              f"enc incr {s['encoder_incr_comm_rel_err']:.4f} | "
              f"dec incr {s['decoder_incr_comm_rel_err']:.4f} | "
              f"enc lin {s['encoder_lin_rel_err']:.4f} | dec lin {s['decoder_lin_rel_err']:.4f} | "
              f"bottleneck inject LSD {s['bottleneck_inject_lsd']:.3f} | "
              f"worst-depth LSD {s['worst_depth_lsd']:.3f}")


# =============================================================================
#  CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AE Deep Additivity — layer-wise trigMix commutation "
                    "+ mix-at-any-depth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoints", required=True, nargs="+",
                   help="One or more checkpoint paths.")
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                   nargs="+", default=None,
                   help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                        "training set.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Preprocessed cache base dir.")
    p.add_argument("--num_pairs", type=int, default=32,
                   help="Number of audio pairs (A, B) to test (default: 32).")
    p.add_argument("--pair_mode", type=str, default="random",
                   choices=["random", "intra", "inter"],
                   help="Pair mode (default: random).")
    p.add_argument("--output_dir", default="evaluation/eval_deep_additivity",
                   help="Directory to save summary JSON.")
    p.add_argument("--device", default="cuda",
                   help="Device (default: cuda).")
    p.add_argument("--sample_rate", type=int, default=MODEL_SAMPLE_RATE,
                   help=f"Target sample rate in Hz (default: {MODEL_SAMPLE_RATE}).")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["conv", "transformer"],
                   help="Model architecture. Auto-detected from checkpoint if omitted.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    print("[Setup] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(args.data_dirs, args.cache_dir,
                                     sample_rate=args.sample_rate)
    print(f"[Setup] {len(stems)} validation stems")

    pairs = generate_pairs(stems, args.num_pairs, seed=42, mode=args.pair_mode)

    results: Dict[str, Dict] = {}
    for ck in args.checkpoints:
        name = Path(ck).stem
        print(f"\n{'='*100}\n  {name}\n{'='*100}")
        results[name] = evaluate_deep_additivity_checkpoint(
            ck, stems, device, num_pairs=len(pairs), seed=42,
            pair_mode=args.pair_mode, model_type=args.model_type,
            sample_rate=args.sample_rate, clip_duration=CLIP_DURATION,
            pairs=pairs,
        )

    print_deep_additivity_tables(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "AE deep additivity — layer-wise trigMix "
                       "commutation + mix-at-any-depth completeness",
        "timestamp": datetime.now().isoformat(),
        "clip_duration_s": CLIP_DURATION,
        "mix_alpha": MIX_ALPHA,
        "num_pairs": len(pairs),
        "pair_mode": args.pair_mode,
        "seed": 42,
        "data_dirs": list(args.data_dirs) if args.data_dirs else DEFAULT_DATA_DIRS,
        "checkpoints": results,
    }
    json_path = out_dir / "deep_additivity_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n📄 JSON → {json_path}")
    print("✓ Deep additivity evaluation complete.")


if __name__ == "__main__":
    main()
