#!/usr/bin/env python3
"""
Shared model loading / auto-detection for the toy evaluation suite
===================================================================

Every evaluation script reconstructs the correct toy autoencoder architecture
from a checkpoint *without* an external config file:

- ``detect_model_type`` distinguishes the CNN baseline
  (``ToyConvAutoencoder``) from the Transformer (``ToyTransformerAutoencoder``)
  by inspecting the state-dict keys.
- ``detect_transformer_config`` recovers the init parameters, most importantly
  the **scale-sensitive bottleneck** flags (``encoder_final_norm=False`` /
  ``decoder_final_norm=False``) that the additivity loss flips — the presence
  or absence of the ``encoder_transformer.norm`` / ``decoder_transformer.norm``
  keys records this in the checkpoint itself.  A ``model_config`` dict saved by
  the training loop is used preferentially when present.
- ``load_model`` instantiates the right class and loads the state dict,
  handling both bare state dicts and wrapped checkpoint dicts.

Centralising this logic guarantees every protocol evaluates the same model
reconstruction, and fixes the previous per-script duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---- Path wrangling ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_model.autoencoder import ToyConvAutoencoder
from toy_model.transformer_autoencoder import ToyTransformerAutoencoder


# =============================================================================
#  Model type auto-detection
# =============================================================================

def detect_model_type(ckpt_path: str, device: Optional[torch.device] = None) -> str:
    """Auto-detect 'conv' or 'transformer' from checkpoint keys."""
    if device is None:
        device = torch.device("cpu")
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        sd = raw.get("model_state_dict", raw.get("model", raw))
    else:
        sd = raw
    sample = " ".join(list(sd.keys())[:20])
    if any(kw in sample for kw in ("patch_to_embed", "encoder_transformer", "_snake_act")):
        return "transformer"
    return "conv"


def detect_transformer_config(ckpt_path: str, device: Optional[torch.device] = None) -> dict:
    """Inspect a transformer checkpoint and return the correct init parameters.

    The trig-loss (latent mixup) model disables the final LayerNorm
    (``encoder_final_norm=False, decoder_final_norm=False``) to keep the
    bottleneck scale-sensitive.  Also detects the base ``activation`` and
    ``decoder_num_snake_layers`` (final decoder layers using Snake), and
    ``use_soft_norm`` (soft-norm bottleneck).
    """
    if device is None:
        device = torch.device("cpu")
    raw = torch.load(ckpt_path, map_location=device)

    # ---- Preferred: explicit config saved by train_toy.py ----
    if isinstance(raw, dict) and isinstance(raw.get("model_config"), dict):
        cfg = dict(raw["model_config"])
        cfg.setdefault("encoder_final_norm", False)
        cfg.setdefault("decoder_final_norm", False)
        cfg.setdefault("activation", "leaky_relu")
        cfg.setdefault("decoder_num_snake_layers", 0)
        cfg.setdefault("use_soft_norm", False)
        cfg.setdefault("latent_dim", 64)
        cfg.setdefault("patch_size", 256)
        cfg.setdefault("d_model", 512)
        return cfg

    if isinstance(raw, dict):
        sd = raw.get("model_state_dict", raw.get("model", raw))
    else:
        sd = raw

    encoder_has_final_norm = any("encoder_transformer.norm" in k for k in sd)
    decoder_has_final_norm = any("decoder_transformer.norm" in k for k in sd)
    use_soft_norm = any("soft_norm." in k for k in sd)

    # ---- Number of transformer layers (encoder/decoder are symmetric) ----
    num_layers = 6
    layer_indices: set = set()
    for k in sd:
        prefix = "encoder_transformer.layers."
        if k.startswith(prefix):
            idx = k[len(prefix):].split(".", 1)[0]
            if idx.isdigit():
                layer_indices.add(int(idx))
    if layer_indices:
        num_layers = max(layer_indices) + 1

    # ---- Snake layers: check `_snake_act.a` presence per decoder layer ----
    dec_snake = {
        i for i in range(num_layers)
        if f"decoder_transformer.layers.{i}._snake_act.a" in sd
    }
    enc_snake = any(
        f"encoder_transformer.layers.{i}._snake_act.a" in sd
        for i in range(num_layers)
    )

    if enc_snake:
        # ---- Legacy (pre-hybrid) checkpoint: `_snake_act` existed in EVERY
        # layer, so key presence is not meaningful.  Fall back to the value
        # heuristic (trained away from 1.0 ⇒ the model used global Snake).
        learned = any(
            "_snake_act.a" in k and not torch.allclose(v, torch.ones_like(v))
            for k, v in sd.items()
        )
        if learned:
            # Old global-Snake model → hybrid equivalent: base activation for
            # the Encoder, Snake for every Decoder layer.
            activation, decoder_num_snake_layers = "leaky_relu", num_layers
        else:
            # Old leaky model: Snake modules present but never trained.
            activation, decoder_num_snake_layers = "leaky_relu", 0
    else:
        # ---- New hybrid checkpoint: Snake forms a contiguous suffix of the
        # Decoder stack.  The base activation (silu vs leaky_relu) is not
        # recoverable from parameter keys, so it defaults to "leaky_relu".
        activation = "leaky_relu"
        decoder_num_snake_layers = 0
        for i in range(num_layers - 1, -1, -1):
            if i in dec_snake:
                decoder_num_snake_layers += 1
            else:
                break

    return {
        "encoder_final_norm": encoder_has_final_norm,
        "decoder_final_norm": decoder_has_final_norm,
        "activation": activation,
        "decoder_num_snake_layers": decoder_num_snake_layers,
        "use_soft_norm": use_soft_norm,
        "latent_dim": 64,
        "patch_size": 256,
        "d_model": 512,
    }


# =============================================================================
#  Model instantiation + state dict loading
# =============================================================================

def load_model(
    ckpt_path: str,
    device: torch.device,
    model_type: Optional[str] = None,
) -> torch.nn.Module:
    """Instantiate the toy autoencoder of the detected (or given) type and
    load the checkpoint state dict into it.

    Supports both bare ``state_dict`` legacy checkpoints and the dict format
    ``{"model_state_dict": ..., "model_config": ...}`` saved by the GAN
    training loop.  Returns the model in eval mode.
    """
    if model_type is None:
        model_type = detect_model_type(ckpt_path, device)

    if model_type == "transformer":
        cfg = detect_transformer_config(ckpt_path, device)
        model = ToyTransformerAutoencoder(
            latent_dim=cfg["latent_dim"], d_model=cfg["d_model"], nhead=8,
            num_layers=6, dim_feedforward=2048,
            dropout=0.0, patch_size=cfg["patch_size"],
            activation=cfg["activation"],
            decoder_num_snake_layers=cfg["decoder_num_snake_layers"],
            encoder_final_norm=cfg["encoder_final_norm"],
            decoder_final_norm=cfg["decoder_final_norm"],
            use_soft_norm=cfg["use_soft_norm"],
        ).to(device)
    else:
        model = ToyConvAutoencoder(latent_dim=64, dropout=0.0).to(device)

    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            model.load_state_dict(raw["model_state_dict"], strict=False)
        elif "model" in raw:
            model.load_state_dict(raw["model"], strict=False)
        else:
            model.load_state_dict(raw, strict=False)
    else:
        model.load_state_dict(raw, strict=False)
    model.eval()
    return model


class ToyModelWrapper:
    """Thin wrapper around a toy autoencoder for encode/decode.

    Normalises the latent shape to ``(1, D, T_latent)`` regardless of the
    underlying model (transformer returns ``(B, T, D)``, conv ``(B, D, T)``),
    so all downstream metric code remains shape-agnostic.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        model_type: Optional[str] = None,
    ):
        self.device = device
        self.name = Path(ckpt_path).stem
        self.model_type = model_type or detect_model_type(ckpt_path, device)
        self.model = load_model(ckpt_path, device, self.model_type)

    @torch.no_grad()
    def encode(self, audio_np) -> torch.Tensor:
        """Mono numpy (T,) → latent (1, D, T_latent)."""
        t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        z = self.model.encode(t)
        if self.model_type == "transformer":
            z = z.permute(0, 2, 1)  # (1, T, D) → (1, D, T)
        return z  # (1, D, T_latent)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, orig_len: int) -> np.ndarray:
        """Latent (1, D, T) → mono numpy (T,)."""
        z_in = z
        if self.model_type == "transformer":
            z_in = z.permute(0, 2, 1)  # (1, D, T) → (1, T, D)
        audio = self.model.decode(z_in, original_length=orig_len)
        return audio.squeeze().cpu().numpy()
