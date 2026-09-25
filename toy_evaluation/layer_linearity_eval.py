#!/usr/bin/env python3
"""
Toy AE — Algebraic Tensor Linearity Evaluation  (Layer-by-Layer)
=================================================================

Evaluates **strict algebraic linearity**  f(A + B) ≈ f(A) + f(B)  directly
on raw internal activation tensors at every layer of both the Encoder and
Decoder stacks, without passing through the rest of the decoder.

Methodology
-----------
1. Load a pair of evaluation audio signals  x_A  and  x_B.
2. Compute the ground-truth audio sum  x_mix = x_A + x_B
   (or a trigonometric sum:  cos(θ)·x_A + sin(θ)·x_B).
3. Register forward hooks on every intermediate layer l in both
   Encoder and Decoder.
4. Run all three inputs (x_A, x_B, x_mix) through the model, capturing:
     - h_A^(l)  = Layer_l(x_A)
     - h_B^(l)  = Layer_l(x_B)
     - h_mix^(l) = Layer_l(x_mix)
5. Compute the linear combination:  h_lin^(l) = h_A^(l) + h_B^(l)
6. For each layer l, compute:
     - Relative L2 Linearity Error:  ||h_mix - h_lin||_2 / ||h_mix||_2
     - Tensor Cosine Similarity:     cos_sim(h_mix, h_lin)

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``toy_evaluation.data``):
validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.

Usage
-----
    python toy_evaluation/layer_linearity_eval.py \\
        --checkpoint_path toy_training/checkpoints/my_model.pt \\
        --num_pairs 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict as TOrderedDict, Tuple

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

from toy_evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from toy_evaluation.model_utils import detect_model_type, load_model

# =============================================================================
#  Constants
# =============================================================================


# =============================================================================
#  Mix computation
# =============================================================================

def compute_mix(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    mix_type: str = "direct",
    theta: float | None = None,
) -> np.ndarray:
    """Compute the mixture signal from two source audios.

    mix_type "direct": x_mix = x_A + x_B ; "trig": cos(θ)·x_A + sin(θ)·x_B
    (θ defaults to π/4, equal weighting).
    """
    if mix_type == "direct":
        return (audio_a.astype(np.float64) + audio_b.astype(np.float64)).astype(np.float32)
    if theta is None:
        theta = np.pi / 4.0
    c = np.cos(theta)
    s = np.sin(theta)
    return (c * audio_a.astype(np.float64) + s * audio_b.astype(np.float64)).astype(np.float32)


# =============================================================================
#  Hook-based Activation Capture
# =============================================================================

class LayerActivationCapture:
    """Registers forward hooks on target modules and captures their outputs."""

    def __init__(
        self,
        model: nn.Module,
        layer_modules: TOrderedDict,  # {display_name: nn.Module}
        device: torch.device,
    ):
        self.model = model
        self.layer_modules = layer_modules
        self.device = device
        self._hooks: list = []
        self._current: Dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._current[name] = output.detach().clone()
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self._current[name] = item.detach().clone()
                        break
        return hook_fn

    def _register(self):
        for name, module in self.layer_modules.items():
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def capture(self, audio_np: np.ndarray) -> Dict[str, torch.Tensor]:
        self._current = {}
        self._register()
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        _ = self.model(x)
        self._remove()
        return dict(self._current)


# =============================================================================
#  Layer discovery  (model → {name: module})
# =============================================================================

def discover_layers(
    model: nn.Module,
    model_type: str,
) -> Tuple[TOrderedDict, List[Tuple[str, str, int]]]:
    """Return (layer_modules_dict, layer_info_list).

    layer_modules_dict: {display_name: nn.Module} for hook registration.
    layer_info_list: [(module_type, display_name, layer_idx), ...] for table.
    """
    layer_modules: TOrderedDict = OrderedDict()
    layer_info: List[Tuple[str, str, int]] = []

    if model_type == "transformer":
        m = model
        num_layers = len(m.encoder_transformer.layers)

        layer_modules["patch_embed"] = m.patch_to_embed
        layer_info.append(("Encoder", "patch_embed", -1))

        for i in range(num_layers):
            name = f"enc_L{i}"
            layer_modules[name] = m.encoder_transformer.layers[i]
            layer_info.append(("Encoder", name, i))

        if m.encoder_transformer.norm is not None:
            layer_modules["enc_final_norm"] = m.encoder_transformer.norm
            layer_info.append(("Encoder", "enc_final_norm", num_layers))

        layer_modules["enc_to_latent"] = m.enc_to_latent
        layer_info.append(("Encoder", "enc_to_latent", num_layers + 1))

        if m.soft_norm is not None:
            layer_modules["soft_norm"] = m.soft_norm
            layer_info.append(("Bottleneck", "soft_norm", -1))

        layer_modules["latent_to_dec"] = m.latent_to_dec
        layer_info.append(("Decoder", "latent_to_dec", -1))

        for i in range(num_layers):
            name = f"dec_L{i}"
            layer_modules[name] = m.decoder_transformer.layers[i]
            layer_info.append(("Decoder", name, i))

        if m.decoder_transformer.norm is not None:
            layer_modules["dec_final_norm"] = m.decoder_transformer.norm
            layer_info.append(("Decoder", "dec_final_norm", num_layers))

        layer_modules["embed_to_patch"] = m.embed_to_patch
        layer_info.append(("Decoder", "embed_to_patch", num_layers + 1))
    else:
        m = model
        for i, block in enumerate([m.enc_block1, m.enc_block2, m.enc_block3, m.enc_block4]):
            name = f"enc_L{i}"
            layer_modules[name] = block
            layer_info.append(("Encoder", name, i))
        layer_modules["enc_bottleneck"] = m.enc_bottleneck
        layer_info.append(("Encoder", "enc_bottleneck", 4))
        for i, block in enumerate([m.dec_expand, m.dec_block1, m.dec_block2, m.dec_block3]):
            name = f"dec_L{i}"
            layer_modules[name] = block
            layer_info.append(("Decoder", name, i))
        layer_modules["dec_out"] = m.dec_block4
        layer_info.append(("Decoder", "dec_out", 4))

    return layer_modules, layer_info


# =============================================================================
#  Linearity metrics
# =============================================================================

def compute_linearity_metrics(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    h_mix: torch.Tensor,
) -> Tuple[float, float]:
    """(relative_l2_error, cosine_similarity) for f(A+B) ≈ f(A)+f(B)."""
    a = h_a.reshape(-1).float()
    b = h_b.reshape(-1).float()
    mix = h_mix.reshape(-1).float()
    h_lin = a + b

    norm_mix = torch.norm(mix, p=2)
    if norm_mix < 1e-12:
        rel_error = 0.0
    else:
        rel_error = float((torch.norm(mix - h_lin, p=2) / norm_mix).cpu())

    dot = torch.dot(mix, h_lin)
    norm_lin = torch.norm(h_lin, p=2)
    if norm_mix < 1e-12 or norm_lin < 1e-12:
        cos_sim = 1.0
    else:
        cos_sim = float((dot / (norm_mix * norm_lin)).cpu())

    return rel_error, cos_sim


# =============================================================================
#  Single-checkpoint core (used by evaluate_all.py and the CLI)
# =============================================================================

def evaluate_linearity_checkpoint(
    ckpt_path: str,
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_pairs: int = 32,
    seed: int = 42,
    mix_type: str = "direct",
    trig_theta: float | None = None,
    model_type: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
    clip_duration: float = CLIP_DURATION,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    """Run the layer-by-layer algebraic linearity test for ONE checkpoint.

    Returns a dict with per-layer results plus stage-aggregated summaries
    (input stage, encoder, bottleneck, decoder, output stage).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_samples = int(clip_duration * sample_rate)

    if model_type is None:
        model_type = detect_model_type(ckpt_path, device)
        print(f"  [Auto-detect] Model type: {model_type}")

    model = load_model(ckpt_path, device, model_type)
    ckpt_name = Path(ckpt_path).stem
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {ckpt_name}  ({model_type})  —  {total_params:,} params")

    layer_modules, layer_info = discover_layers(model, model_type)
    capture = LayerActivationCapture(model, layer_modules, device)

    if pairs is None:
        pairs = generate_pairs(stems, num_pairs, seed=seed, mode="random")

    accum: Dict[str, Dict[str, list]] = {
        name: {"rel_errors": [], "cos_sims": []}
        for _, name, _ in layer_info
    }

    mix_types = ["direct", "trig"] if mix_type == "both" else [mix_type]

    for pair_idx, (i, j) in enumerate(tqdm(pairs, desc="Pairs", unit="pair")):
        path_a, _sa, _ta = stems[i]
        path_b, _sb, _tb = stems[j]
        try:
            audio_a = stem_to_numpy(path_a, clip_samples, centre=True)
            audio_b = stem_to_numpy(path_b, clip_samples, centre=True)
        except Exception as exc:
            print(f"  ⚠ Skip pair {pair_idx}: load error – {exc}")
            continue

        for mt in mix_types:
            audio_mix = compute_mix(audio_a, audio_b, mix_type=mt, theta=trig_theta)
            try:
                acts_a = capture.capture(audio_a)
                acts_b = capture.capture(audio_b)
                acts_mix = capture.capture(audio_mix)
            except Exception as exc:
                print(f"  ⚠ Skip pair {pair_idx} ({mt}): capture error – {exc}")
                continue

            for layer_name in accum:
                if layer_name not in acts_a or layer_name not in acts_b or layer_name not in acts_mix:
                    continue
                rel_err, cos_sim = compute_linearity_metrics(
                    acts_a[layer_name], acts_b[layer_name], acts_mix[layer_name],
                )
                accum[layer_name]["rel_errors"].append(rel_err)
                accum[layer_name]["cos_sims"].append(cos_sim)

    # ---- Per-layer results ----
    layers_out: List[Dict] = []
    for mt, name, idx in layer_info:
        rels = accum[name]["rel_errors"]
        coss = accum[name]["cos_sims"]
        layers_out.append({
            "module": mt,
            "layer_name": name,
            "layer_idx": idx,
            "mean_rel_error": float(np.mean(rels)) if rels else float("nan"),
            "std_rel_error": float(np.std(rels)) if rels else float("nan"),
            "mean_cos_sim": float(np.mean(coss)) if coss else float("nan"),
            "std_cos_sim": float(np.std(coss)) if coss else float("nan"),
            "n_samples": len(rels),
        })

    # ---- Stage aggregation ----
    def _stage(name_filter, layers=layers_out):
        rows = [l for l in layers if name_filter(l["layer_name"]) and l["n_samples"] > 0]
        if not rows:
            return {"mean_rel_error": float("nan"), "mean_cos_sim": float("nan"),
                    "n_layers": 0}
        return {
            "mean_rel_error": float(np.mean([r["mean_rel_error"] for r in rows])),
            "mean_cos_sim": float(np.mean([r["mean_cos_sim"] for r in rows])),
            "n_layers": len(rows),
        }

    if model_type == "transformer":
        stages = {
            "input_stage": _stage(lambda n: n == "patch_embed"),
            "encoder": _stage(lambda n: n.startswith("enc_L") or n == "enc_final_norm"),
            "bottleneck": _stage(lambda n: n in ("enc_to_latent", "soft_norm")),
            "decoder": _stage(lambda n: n == "latent_to_dec" or n.startswith("dec_L")
                              or n == "dec_final_norm"),
            "output_stage": _stage(lambda n: n == "embed_to_patch"),
        }
    else:
        stages = {
            "encoder": _stage(lambda n: n.startswith("enc_L")),
            "bottleneck": _stage(lambda n: n == "enc_bottleneck"),
            "decoder": _stage(lambda n: n.startswith("dec_L") or n == "dec_out"),
        }

    all_rel = [r["mean_rel_error"] for r in layers_out if r["n_samples"] > 0]
    return {
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "mix_type": mix_type,
        "trig_theta_rad": trig_theta if mix_type in ("trig", "both") else None,
        "num_pairs_requested": len(pairs),
        "num_pairs_valid": len(pairs),  # per-layer n_samples is authoritative
        "stages": stages,
        "overall_rel_error": float(np.mean(all_rel)) if all_rel else float("nan"),
        "overall_cos_sim": float(np.mean(
            [r["mean_cos_sim"] for r in layers_out if r["n_samples"] > 0]
        )) if all_rel else float("nan"),
        "layers": layers_out,
    }


# =============================================================================
#  CLI wrapper
# =============================================================================

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    print("[0/4] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(args.data_dirs, args.cache_dir,
                                     sample_rate=args.sample_rate)
    print(f"  Found {len(stems)} validation stems")

    results = evaluate_linearity_checkpoint(
        args.checkpoint_path, stems, device,
        num_pairs=args.num_pairs, seed=42,
        mix_type=args.mix_type, trig_theta=args.trig_theta,
        model_type=args.model_type, sample_rate=args.sample_rate,
        clip_duration=CLIP_DURATION,
    )

    # ---- Print table ----
    print_linearity_table(results["layers"])

    # ---- Save JSON ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "Toy AE Algebraic Tensor Linearity — f(A+B) ≈ f(A)+f(B)",
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint_path,
        "model_type": results["model_type"],
        "mix_type": args.mix_type,
        "trig_theta_rad": results["trig_theta_rad"],
        "clip_duration_s": CLIP_DURATION,
        "num_pairs_requested": args.num_pairs,
        "stages": results["stages"],
        "overall_rel_error": results["overall_rel_error"],
        "overall_cos_sim": results["overall_cos_sim"],
        "layers": results["layers"],
    }
    json_path = out_dir / "linearity_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  📄 JSON → {json_path}")
    print("\n✓ Algebraic tensor linearity evaluation complete.")


def print_linearity_table(results_layers: List[Dict]) -> None:
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  Algebraic Tensor Linearity — Layer-by-Layer Results  (f(A+B) ≈ f(A)+f(B))")
    print(sep)
    print(f"{'Module':<12} {'Layer':<18} {'Idx':<5} {'Rel L2 Err':<12} {'Tensor CosSim':<15}")
    print("-" * 100)
    for r in results_layers:
        display_idx = f"{r['layer_idx']}" if r["layer_idx"] >= 0 else "—"
        print(f"{r['module']:<12} {r['layer_name']:<18} {display_idx:<5} "
              f"{r['mean_rel_error']:>10.6f}   {r['mean_cos_sim']:>13.6f}")
    print(f"{sep}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy AE Algebraic Tensor Linearity — Layer-by-Layer f(A+B)≈f(A)+f(B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint_path", required=True,
                   help="Path to the .pt model checkpoint.")
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                   nargs="+", default=None,
                   help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                        "training set.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Preprocessed cache base dir.")
    p.add_argument("--num_pairs", type=int, default=32,
                   help="Number of audio pairs (A, B) to test (default: 32).")
    p.add_argument("--mix_type", type=str, default="direct",
                   choices=["direct", "trig", "both"],
                   help="Mixture type: 'direct' = A+B, 'trig' = cos(θ)A+sin(θ)B, "
                        "'both' = run both (default: direct).")
    p.add_argument("--trig_theta", type=float, default=None,
                   help="Angle θ in radians for trigonometric mix (default: π/4).")
    p.add_argument("--output_dir", default="toy_evaluation/eval_linearity",
                   help="Directory to save summary JSON.")
    p.add_argument("--device", default="cuda:0",
                   help="Device (default: cuda:0).")
    p.add_argument("--sample_rate", type=int, default=MODEL_SAMPLE_RATE,
                   help=f"Target sample rate in Hz (default: {MODEL_SAMPLE_RATE}).")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["conv", "transformer"],
                   help="Model architecture. Auto-detected from checkpoint if omitted.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
