#!/usr/bin/env python3
"""
Toy AE — Self-Attention Linearity & Dynamic Behavior Evaluation
================================================================

Extracts the raw Self-Attention weight matrices across all heads and layers
to evaluate if attention routing behaves linearly under signal mixing.

Methodology
-----------
1. Load audio x_A, x_B, and their mixture:
     x_mix = cos(θ)·x_A + sin(θ)·x_B   (default θ = π/4, equal weighting)
2. Register attention-capture patches on all ``RoPETransformerEncoderLayer``
   sub-modules to intercept the softmax-normalised attention weights
   A^(l,h) for each layer l and head h.
3. Run all three inputs (A, B, mix) through the model and capture:
     - A_A^(l,h)   — attention map for source A
     - A_B^(l,h)   — attention map for source B
     - A_mix^(l,h) — attention map for mixture
4. Compute the expected linear combination (attention mass scales with
   signal *energy*, i.e. with amplitude²):
     A_lin^(l,h) = cos²(θ)·A_A^(l,h) + sin²(θ)·A_B^(l,h)
5. For every (layer, head), compute:
     - Frobenius norm distance:  ||A_mix - A_lin||_F / ||A_mix||_F
     - Cosine Similarity between flattened attention maps

Data & split
------------
Uses the shared multi-dataset evaluation data layer (``toy_evaluation.data``):
validation stems are the exact held-out tracks of the training split
(VCTK + FSD50K + FMA-small), loaded from the preprocessed 16 kHz cache.
(This fixes the previous version, which sampled stems without any
train/validation restriction.)

Outputs
-------
- ``attention_summary.json`` — aggregated per-layer × per-head metrics.
- ``attention_linearity_maps.png`` — heatmap grid of attention maps.
- Printed summary table at end of run.

Usage
-----
    python toy_evaluation/attention_map_eval.py \\
        --checkpoint_path toy_training/checkpoints/my_model.pt \\
        --output_dir toy_evaluation/eval_attention
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_model.transformer_autoencoder import RoPETransformerEncoderLayer
from toy_evaluation.data import (
    CLIP_DURATION,
    MODEL_SAMPLE_RATE,
    collect_validation_stems,
    generate_pairs,
    stem_to_numpy,
)
from toy_evaluation.model_utils import detect_model_type, load_model

# =============================================================================
#  Mix computation
# =============================================================================

def compute_mix(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    mix_type: str = "trig",
    theta: float | None = None,
) -> np.ndarray:
    """Compute the mixture signal from two source audios."""
    if mix_type == "direct":
        return (audio_a.astype(np.float64) + audio_b.astype(np.float64)).astype(np.float32)
    if theta is None:
        theta = np.pi / 4.0
    c = np.cos(theta)
    s = np.sin(theta)
    return (c * audio_a.astype(np.float64) + s * audio_b.astype(np.float64)).astype(np.float32)


def get_linear_combination_weights(mix_type: str, theta: float | None = None) -> Tuple[float, float]:
    """Return (w_A, w_B) for the expected linear combination of attention maps.

    Attention maps are softmax-normalised distributions; the expected
    combination uses squared coefficients because attention is proportional to
    signal energy, which scales with the square of amplitude.
    """
    if mix_type == "direct":
        return 0.5, 0.5
    if theta is None:
        theta = np.pi / 4.0
    c2 = np.cos(theta) ** 2
    s2 = np.sin(theta) ** 2
    total = c2 + s2
    return c2 / total, s2 / total


# =============================================================================
#  Attention Capture via Method Patching
# =============================================================================

def _make_patched_sa_block(original_method, capture_dict: dict, layer_key: str):
    """Create a patched ``_sa_block`` that stores attention weights."""
    def patched_sa_block(self, x, attn_mask, key_padding_mask):
        B, L, E = x.shape
        H = self.nhead
        D = self.head_dim

        Q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        K = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        Q = self.rope(Q)
        K = self.rope(K)

        scale = D ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"),
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        capture_dict[layer_key] = attn_weights.detach().clone()

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(attn_out)

    return patched_sa_block


class AttentionCaptureContext:
    """Context manager that temporarily patches RoPETransformerEncoderLayer
    sub-modules to capture multi-head self-attention weight matrices."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._originals: Dict[int, Callable] = {}
        self._capture_dict: Dict[str, torch.Tensor] = {}
        self._patched_modules: List[nn.Module] = []

    def _collect_transformer_layers(self) -> List[Tuple[str, nn.Module]]:
        layers: List[Tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if isinstance(module, RoPETransformerEncoderLayer):
                layers.append((name, module))
        return layers

    def _patch_all(self):
        self._capture_dict.clear()
        self._originals.clear()
        self._patched_modules.clear()
        for name, module in self._collect_transformer_layers():
            mod_id = id(module)
            self._originals[mod_id] = module._sa_block
            patched = _make_patched_sa_block(module._sa_block, self._capture_dict, name)
            module._sa_block = patched.__get__(module, type(module))
            self._patched_modules.append(module)

    def _restore_all(self):
        for name, module in self._collect_transformer_layers():
            mod_id = id(module)
            if mod_id in self._originals:
                module._sa_block = self._originals[mod_id]
        self._originals.clear()
        self._patched_modules.clear()

    def __enter__(self):
        self._patch_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_all()
        return False

    @torch.no_grad()
    def capture(self, audio_np: np.ndarray) -> Dict[str, torch.Tensor]:
        self._capture_dict.clear()
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return {k: v.clone() for k, v in self._capture_dict.items()}


# =============================================================================
#  Attention Linearity Metrics
# =============================================================================

def compute_attention_linearity(
    attn_a: torch.Tensor,    # (nhead, L, L)
    attn_b: torch.Tensor,    # (nhead, L, L)
    attn_mix: torch.Tensor,  # (nhead, L, L)
    w_a: float,
    w_b: float,
) -> Dict[str, np.ndarray]:
    """Per-head linearity metrics between mixed and expected attention."""
    nhead = attn_a.shape[0]
    frob_rel = np.zeros(nhead, dtype=np.float64)
    cos_sims = np.zeros(nhead, dtype=np.float64)

    attn_lin = w_a * attn_a + w_b * attn_b
    attn_diff = torch.abs(attn_mix - attn_lin)

    for h in range(nhead):
        mix_h = attn_mix[h].reshape(-1).float()
        lin_h = attn_lin[h].reshape(-1).float()
        norm_mix = torch.norm(mix_h, p=2)
        if norm_mix < 1e-12:
            frob_rel[h] = 0.0
        else:
            frob_rel[h] = float((torch.norm(mix_h - lin_h, p=2) / norm_mix).cpu())
        dot = torch.dot(mix_h, lin_h)
        norm_lin = torch.norm(lin_h, p=2)
        if norm_mix < 1e-12 or norm_lin < 1e-12:
            cos_sims[h] = 1.0
        else:
            cos_sims[h] = float((dot / (norm_mix * norm_lin)).cpu())

    return {
        "frobenius_relative": frob_rel,
        "cosine_similarity": cos_sims,
        "attn_lin": attn_lin,
        "attn_diff": attn_diff,
    }


# =============================================================================
#  Single-checkpoint core (used by evaluate_all.py and the CLI)
# =============================================================================

def evaluate_attention_checkpoint(
    ckpt_path: str,
    stems: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    num_pairs: int = 32,
    seed: int = 42,
    mix_type: str = "trig",
    theta: float | None = None,
    pairs: Optional[List[Tuple[int, int]]] = None,
    save_plots: bool = False,
    output_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Run the self-attention linearity test for ONE checkpoint.

    Returns a dict with per-layer × per-head metrics and overall means, or
    ``None`` when the checkpoint is a conv model (no self-attention).
    """
    model_type = detect_model_type(ckpt_path, device)
    if model_type == "conv":
        print("  ⚠ Conv autoencoder detected — self-attention evaluation N/A.")
        return None

    model = load_model(ckpt_path, device, model_type)
    print(f"  Model: {Path(ckpt_path).stem} (transformer)")

    layer_keys: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, RoPETransformerEncoderLayer):
            layer_keys.append(name)
    print(f"  [Layers] {len(layer_keys)} RoPETransformerEncoderLayer instances")

    if pairs is None:
        pairs = generate_pairs(stems, num_pairs, seed=seed, mode="random")
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)
    w_a, w_b = get_linear_combination_weights(mix_type, theta)
    n_layers = len(layer_keys)
    nhead = model.encoder_transformer.layers[0].nhead

    frob_sum = np.zeros((n_layers, nhead), dtype=np.float64)
    cos_sum = np.zeros((n_layers, nhead), dtype=np.float64)
    valid_count = np.zeros((n_layers, nhead), dtype=np.int32)
    first_pair_attn: Dict[str, Dict[str, torch.Tensor]] = {}

    for pair_idx, (i, j) in enumerate(tqdm(pairs, desc="Evaluating pairs", unit="pair")):
        path_a, _, _ = stems[i]
        path_b, _, _ = stems[j]
        try:
            audio_a = stem_to_numpy(path_a, clip_samples, centre=True)
            audio_b = stem_to_numpy(path_b, clip_samples, centre=True)
        except Exception as e:
            print(f"  ⚠ Skipping pair {pair_idx}: {e}")
            continue

        audio_mix = compute_mix(audio_a, audio_b, mix_type, theta)

        with AttentionCaptureContext(model, device) as cap:
            attn_maps_a = cap.capture(audio_a)
        with AttentionCaptureContext(model, device) as cap:
            attn_maps_b = cap.capture(audio_b)
        with AttentionCaptureContext(model, device) as cap:
            attn_maps_mix = cap.capture(audio_mix)

        if pair_idx == 0:
            first_pair_attn = {
                "A": {k: v.cpu().clone() for k, v in attn_maps_a.items()},
                "B": {k: v.cpu().clone() for k, v in attn_maps_b.items()},
                "mix": {k: v.cpu().clone() for k, v in attn_maps_mix.items()},
            }

        for layer_idx, lkey in enumerate(layer_keys):
            if lkey not in attn_maps_a or lkey not in attn_maps_b or lkey not in attn_maps_mix:
                continue
            a_map = attn_maps_a[lkey].squeeze(0)
            b_map = attn_maps_b[lkey].squeeze(0)
            mix_map = attn_maps_mix[lkey].squeeze(0)
            try:
                metrics = compute_attention_linearity(a_map, b_map, mix_map, w_a, w_b)
                for h in range(nhead):
                    frob_sum[layer_idx, h] += metrics["frobenius_relative"][h]
                    cos_sum[layer_idx, h] += metrics["cosine_similarity"][h]
                    valid_count[layer_idx, h] += 1
            except Exception as e:
                print(f"  ⚠ Error computing metrics for {lkey}: {e}")

    valid_mask = valid_count > 0
    frob_avg = np.where(valid_mask, frob_sum / np.maximum(valid_count, 1), np.nan)
    cos_avg = np.where(valid_mask, cos_sum / np.maximum(valid_count, 1), np.nan)

    layers_out = []
    for layer_idx, lkey in enumerate(layer_keys):
        layers_out.append({
            "layer": lkey,
            "frobenius_relative": {f"h{h}": float(frob_avg[layer_idx, h])
                                   if valid_mask[layer_idx, h] else None
                                   for h in range(nhead)},
            "cosine_similarity": {f"h{h}": float(cos_avg[layer_idx, h])
                                  if valid_mask[layer_idx, h] else None
                                  for h in range(nhead)},
            "frobenius_relative_mean": float(np.nanmean(frob_avg[layer_idx])),
            "cosine_similarity_mean": float(np.nanmean(cos_avg[layer_idx])),
        })

    results = {
        "checkpoint": ckpt_path,
        "model_type": model_type,
        "mix_type": mix_type,
        "theta": theta if theta is not None else "pi/4",
        "num_pairs": len(pairs),
        "linear_weights": {"w_A": w_a, "w_B": w_b},
        "num_layers": n_layers,
        "num_heads": nhead,
        "frobenius_relative_mean": float(np.nanmean(frob_avg)),
        "cosine_similarity_mean": float(np.nanmean(cos_avg)),
        "layers": layers_out,
    }

    if save_plots and output_dir is not None and first_pair_attn:
        _save_plots(first_pair_attn, w_a, w_b, frob_avg, cos_avg, layer_keys,
                    valid_mask, nhead, output_dir)

    return results


def _save_plots(first_pair_attn, w_a, w_b, frob_avg, cos_avg, layer_keys,
                valid_mask, nhead, out_dir: Path) -> None:
    """Heatmap grids for representative layers + summary heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_layers = layer_keys[:min(4, len(layer_keys))]
    dec_layers = [k for k in layer_keys if "decoder" in k.lower() or "dec_" in k.lower()]
    if dec_layers:
        rep_layers = rep_layers[:2] + dec_layers[:min(2, len(dec_layers))]

    for lkey in rep_layers:
        if lkey not in first_pair_attn["A"]:
            continue
        a_map = first_pair_attn["A"][lkey].squeeze(0)
        b_map = first_pair_attn["B"][lkey].squeeze(0)
        mix_map = first_pair_attn["mix"][lkey].squeeze(0)
        lin_map = w_a * a_map + w_b * b_map
        diff_map = torch.abs(mix_map - lin_map)
        safe_name = lkey.replace(".", "_").replace("/", "_")
        plot_attention_heatmaps(
            a_map, b_map, mix_map, lin_map, diff_map,
            [0, 1, 2, 3], lkey, out_dir / f"attention_maps_{safe_name}.png",
        )

    plot_summary_heatmap(frob_avg, cos_avg, layer_keys, valid_mask,
                         out_dir / "attention_linearity_summary.png")


# =============================================================================
#  Plotting
# =============================================================================

def plot_attention_heatmaps(
    attn_a, attn_b, attn_mix, attn_lin, attn_diff,
    heads_to_show: List[int], layer_label: str, output_path: Path,
) -> None:
    n_heads = len(heads_to_show)
    n_cols = 5
    fig, axes = plt.subplots(n_heads, n_cols, figsize=(n_cols * 3.5, n_heads * 3.0),
                             squeeze=False)
    fig.suptitle(f"Attention Linearity — {layer_label}", fontsize=13, fontweight="bold")
    col_labels = ["A_A (source)", "A_B (source)", "A_mix (actual)",
                  "A_lin (expected)", "|A_mix − A_lin|"]
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=9, fontweight="bold")
    for row_idx, head_idx in enumerate(heads_to_show):
        maps = [
            attn_a[head_idx].cpu().numpy(), attn_b[head_idx].cpu().numpy(),
            attn_mix[head_idx].cpu().numpy(), attn_lin[head_idx].cpu().numpy(),
            attn_diff[head_idx].cpu().numpy(),
        ]
        titles = [f"A (h={head_idx})", f"B (h={head_idx})", f"mix (h={head_idx})",
                  f"lin (h={head_idx})", f"diff (h={head_idx})"]
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            data = maps[col_idx]
            vmax_val = np.percentile(data, 99) if col_idx < 4 else np.percentile(data, 95)
            ax.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=max(vmax_val, 1e-8))
            ax.set_title(titles[col_idx], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(frob_avg, cos_avg, layer_keys, valid_mask, output_path: Path) -> None:
    n_layers, nhead = frob_avg.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, n_layers * 0.6)))
    frob_masked = np.where(valid_mask, frob_avg, np.nan)
    cos_masked = np.where(valid_mask, cos_avg, np.nan)
    short_names = [k.replace("encoder_transformer.layers.", "enc_")
                    .replace("decoder_transformer.layers.", "dec_")
                   for k in layer_keys]
    im1 = ax1.imshow(frob_masked, aspect="auto", cmap="YlOrRd", vmin=0)
    ax1.set_title("Frobenius Relative Error (↓ linear)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_xticks(range(nhead))
    ax1.set_xticklabels([f"h{h}" for h in range(nhead)], fontsize=8)
    ax1.set_yticks(range(n_layers))
    ax1.set_yticklabels(short_names, fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    im2 = ax2.imshow(cos_masked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_title("Cosine Similarity (↑ linear)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_xticks(range(nhead))
    ax2.set_xticklabels([f"h{h}" for h in range(nhead)], fontsize=8)
    ax2.set_yticks(range(n_layers))
    ax2.set_yticklabels(short_names, fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    fig.suptitle("Attention Linearity — Layer × Head Summary", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Attention Linearity & Dynamic Behavior Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--data_dir", "--data_dirs", dest="data_dirs", type=str,
                        nargs="+", default=None,
                        help="Dataset root(s). Defaults to the VCTK+FSD50K+FMA "
                             "training set.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Preprocessed cache base dir.")
    parser.add_argument("--output_dir", type=str, default="toy_evaluation/eval_attention",
                        help="Directory for output plots and JSON.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda, cpu, mps).")
    parser.add_argument("--num_pairs", type=int, default=32,
                        help="Number of audio pairs to evaluate (default 32).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mix_type", type=str, default="trig", choices=["direct", "trig"],
                        help="Mixture type: 'direct' (sum) or 'trig' (cos·A + sin·B).")
    parser.add_argument("--theta", type=float, default=None,
                        help="Angle in radians for trigonometric mix (default π/4).")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    print("[0/3] Collecting validation stems (multi-dataset training split) …")
    stems = collect_validation_stems(args.data_dirs, args.cache_dir)
    print(f"  Found {len(stems)} validation stems")

    results = evaluate_attention_checkpoint(
        args.checkpoint_path, stems, device,
        num_pairs=args.num_pairs, seed=args.seed,
        mix_type=args.mix_type, theta=args.theta,
        save_plots=True, output_dir=Path(args.output_dir),
    )

    if results is None:
        print("  ⚠ Attention evaluation skipped (conv model).")
        return

    _print_summary_table(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(results)
    summary["timestamp"] = datetime.now().isoformat()
    json_path = out_dir / "attention_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n  ✓ Saved summary → {json_path}")
    print(f"\n{'='*60}\n  Attention Linearity Evaluation — Complete\n{'='*60}")


def _print_summary_table(results: Dict) -> None:
    print(f"\n{'='*90}")
    print(f"  Attention Linearity Summary  (n_pairs={results['num_pairs']}, "
          f"mix={results['mix_type']})")
    print(f"{'='*90}")
    print(f"  {'Layer':<24} {'Heads':>6}  {'Frob Rel ↓':>12}  {'CosSim ↑':>12}")
    print(f"  {'-'*24} {'-'*6}  {'-'*12}  {'-'*12}")
    for layer in results["layers"]:
        print(f"  {layer['layer']:<24} {results['num_heads']:>6}  "
              f"{layer['frobenius_relative_mean']:>12.6f}  "
              f"{layer['cosine_similarity_mean']:>12.6f}")
    print(f"  Overall: Frob rel = {results['frobenius_relative_mean']:.6f}, "
          f"CosSim = {results['cosine_similarity_mean']:.6f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
