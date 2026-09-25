#!/usr/bin/env python3
"""
Toy AE — Layer-wise Feature Probing via Spectrogram Analysis
=============================================================

Intercepts intermediate feature representations at every layer of both the
Encoder and Decoder, passes them through the continuation pipeline to decode
them into audio, and generates comparative spectrogram visualizations.

Core idea
---------
For a single test audio sample x (or a pair x_A, x_B):
  1. For every layer l of the Encoder (patch_embed → enc_L5):
     a. Extract H_e^(l) by running the encoder up to layer l.
     b. Continue through the remaining encoder layers, bottleneck, and full
        decoder to produce audio.
  2. For every layer l of the Decoder (latent_to_dec → dec_L5):
     a. Fully encode x → Z, then run the decoder up to layer l.
     b. Continue through the remaining decoder layers to produce audio.
  3. Compute the STFT log-magnitude spectrogram for each layer's decoded
     output and plot a grid arranged chronologically by layer depth.

Outputs
-------
- ``layer_spectrograms.png`` — grid of spectrograms per interception layer.
- ``layer_spectrograms_{stem}.wav`` — optional per-layer reconstructed audio.

Usage
-----
    python toy_evaluation/layer_spectrogram_eval.py \\
        --checkpoint_path toy_training/checkpoints/my_model.pt \\
        --data_dir /path/to/vctk/test \\
        --output_dir toy_evaluation/eval_spectrograms

    # Or specify a single audio file directly:
    python toy_evaluation/layer_spectrogram_eval.py \\
        --checkpoint_path toy_training/checkpoints/my_model.pt \\
        --audio_path /path/to/sample.wav \\
        --output_dir toy_evaluation/eval_spectrograms
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from scipy.io import wavfile

# ---- Path wrangling ----
_THIS_FILE = Path(__file__).resolve()
_EVAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _EVAL_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_model.autoencoder import ToyConvAutoencoder
from toy_model.transformer_autoencoder import ToyTransformerAutoencoder

# =============================================================================
#  Constants
# =============================================================================

VALID_STEMS = {".wav", ".flac"}
CLIP_DURATION = 2.0
MODEL_SAMPLE_RATE = 16000

# Spectrogram parameters
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048


# =============================================================================
#  0.  Model type auto-detection & config
# =============================================================================


def detect_model_type(ckpt_path: str, device: torch.device | None = None) -> str:
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


def detect_transformer_config(ckpt_path: str, device: torch.device | None = None) -> dict:
    """Inspect a transformer checkpoint and return the correct init parameters."""
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
    }


# =============================================================================
#  1.  Audio helpers
# =============================================================================


def load_audio(path: str, target_sr: int = MODEL_SAMPLE_RATE) -> np.ndarray:
    """Load any audio file to mono float32 at *target_sr*."""
    audio_tensor, orig_sr = torchaudio.load(path)
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=orig_sr, new_freq=target_sr,
        )
    return audio_tensor.squeeze(0).numpy().astype(np.float32)


def load_mono_clip(
    audio_path: str,
    clip_samples: int,
    centre: bool = True,
    target_sr: int = MODEL_SAMPLE_RATE,
) -> np.ndarray:
    """Load audio, convert to mono, resample, extract clip."""
    data = load_audio(audio_path, target_sr=target_sr)
    total = len(data)
    if total < clip_samples:
        data = np.pad(data, (0, clip_samples - total))
    else:
        if centre:
            start = (total - clip_samples) // 2
        else:
            import random
            start = random.randint(0, total - clip_samples)
        data = data[start:start + clip_samples]
    return data.astype(np.float32)


# =============================================================================
#  2.  Spectrogram computation
# =============================================================================


def compute_log_mel_or_stft(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
    win: int = WIN_LENGTH,
    sample_rate: int = MODEL_SAMPLE_RATE,
    log_eps: float = 1e-6,
) -> np.ndarray:
    """Compute log-magnitude STFT spectrogram.

    Parameters
    ----------
    audio : np.ndarray  shape (T,), mono float32
    n_fft, hop, win : int — STFT parameters.
    sample_rate : int
    log_eps : float — small value to avoid log(0).

    Returns
    -------
    np.ndarray  shape (n_fft//2 + 1, n_frames) — log-magnitude in dB.
    """
    # Ensure float64 for STFT
    signal = audio.astype(np.float64)
    window = np.hanning(win)

    n_frames = (len(signal) - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 1))

    n_bins = n_fft // 2 + 1
    spec = np.zeros((n_bins, n_frames), dtype=np.float64)

    for t in range(n_frames):
        start = t * hop
        frame = signal[start:start + n_fft] * window
        spec[:, t] = np.abs(np.fft.rfft(frame))

    # Convert to log-magnitude (dB)
    spec_db = 20.0 * np.log10(spec + log_eps)

    return spec_db.astype(np.float32)


# =============================================================================
#  3.  Layer Probing Wrapper
# =============================================================================


class LayerProbingWrapper:
    """Loads a toy autoencoder and exposes per-layer feature extraction
    with continuation through the remaining pipeline.

    Supports both ``ToyTransformerAutoencoder`` and ``ToyConvAutoencoder``.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        model_type: str = "conv",
    ):
        self.device = device
        self.name = Path(ckpt_path).stem
        self.model_type = model_type

        if model_type == "transformer":
            cfg = detect_transformer_config(ckpt_path, device)
            self.model = ToyTransformerAutoencoder(
                latent_dim=64, d_model=512, nhead=8,
                num_layers=6, dim_feedforward=2048,
                dropout=0.0, patch_size=256,
                activation=cfg["activation"],
                decoder_num_snake_layers=cfg["decoder_num_snake_layers"],
                encoder_final_norm=cfg["encoder_final_norm"],
                decoder_final_norm=cfg["decoder_final_norm"],
                use_soft_norm=cfg["use_soft_norm"],
            ).to(device)
        else:
            self.model = ToyConvAutoencoder(latent_dim=64, dropout=0.0).to(device)

        raw = torch.load(ckpt_path, map_location=device)
        if isinstance(raw, dict):
            if "model_state_dict" in raw:
                self.model.load_state_dict(raw["model_state_dict"], strict=False)
            elif "model" in raw:
                self.model.load_state_dict(raw["model"], strict=False)
            else:
                self.model.load_state_dict(raw, strict=False)
        else:
            self.model.load_state_dict(raw, strict=False)
        self.model.eval()

    # ------------------------------------------------------------------
    #  Build interception plan
    # ------------------------------------------------------------------

    def get_interception_plan(self) -> List[Tuple[str, str]]:
        """Return list of (module_type, layer_name) to intercept.

        module_type is "encoder" or "decoder".
        """
        plan: List[Tuple[str, str]] = []

        if self.model_type == "transformer":
            m = self.model
            num_layers = len(m.encoder_transformer.layers)

            # Encoder interception points
            plan.append(("encoder", "patch_embed"))
            for i in range(num_layers):
                plan.append(("encoder", f"enc_L{i}"))

            # Decoder interception points
            plan.append(("decoder", "latent_to_dec"))
            for i in range(num_layers):
                plan.append(("decoder", f"dec_L{i}"))
        else:
            # Conv model
            enc_names = ["enc_block1", "enc_block2", "enc_block3", "enc_block4"]
            dec_names = ["dec_expand", "dec_block1", "dec_block2", "dec_block3"]
            for name in enc_names:
                plan.append(("encoder", name))
            for name in dec_names:
                plan.append(("decoder", name))

        return plan

    # ------------------------------------------------------------------
    #  Transformer: Encoder probing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _transformer_encode_to_layer(
        self, audio_np: np.ndarray, layer_name: str,
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], int]:
        """Run encoder up to *layer_name*, return (features, continue_fn, orig_len).

        layer_name: "patch_embed" or "enc_L{i}".
        continue_fn(feat) → (B, 1, T) — full decoded audio.
        """
        model = self.model
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        T_orig = x.shape[-1]
        num_layers = len(model.encoder_transformer.layers)

        # ---- Patch embedding ----
        B, C, T = x.shape
        pad_len = (model.patch_size - T % model.patch_size) % model.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
        L_val = x.shape[-1] // model.patch_size
        x = x.reshape(B, L_val, model.patch_size)
        x = model.patch_to_embed(x)  # (1, L, d_model)

        if layer_name == "patch_embed":
            intermediate = x.clone()
            # Continue: remaining encoder → bottleneck → decoder
            def _continue(feat: torch.Tensor) -> torch.Tensor:
                h = feat
                for i in range(num_layers):
                    h = model.encoder_transformer.layers[i](h)
                if model.encoder_transformer.norm is not None:
                    h = model.encoder_transformer.norm(h)
                z = model.enc_to_latent(h)
                if model.soft_norm is not None:
                    z = model.soft_norm(z)
                return model.decode(z, original_length=T_orig)
            return intermediate, _continue, T_orig

        # Otherwise, run through encoder layers up to the specified index
        layer_idx = int(layer_name.split("_L")[-1])
        for i in range(layer_idx + 1):
            x = model.encoder_transformer.layers[i](x)

        intermediate = x.clone()

        def _continue_enc(feat: torch.Tensor) -> torch.Tensor:
            h = feat
            for i in range(layer_idx + 1, num_layers):
                h = model.encoder_transformer.layers[i](h)
            if model.encoder_transformer.norm is not None:
                h = model.encoder_transformer.norm(h)
            z = model.enc_to_latent(h)
            if model.soft_norm is not None:
                z = model.soft_norm(z)
            return model.decode(z, original_length=T_orig)

        return intermediate, _continue_enc, T_orig

    # ------------------------------------------------------------------
    #  Transformer: Decoder probing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _transformer_decode_to_layer(
        self, audio_np: np.ndarray, layer_name: str,
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], int]:
        """Full encode → run decoder up to *layer_name*, return (features, continue_fn, orig_len).

        layer_name: "latent_to_dec" or "dec_L{i}".
        """
        model = self.model
        T_orig = len(audio_np)
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # Full encode
        z = model.encode(x)  # (1, L, latent_dim)
        B, L_dec, _ = z.shape

        # Reverse soft-norm if needed
        if model.soft_norm is not None:
            z = model.soft_norm.reverse(z)

        # Expand to d_model
        x_dec = model.latent_to_dec(z)  # (1, L, d_model)

        if layer_name == "latent_to_dec":
            intermediate = x_dec.clone()

            def _continue_latent(feat: torch.Tensor) -> torch.Tensor:
                h = feat
                num_layers = len(model.decoder_transformer.layers)
                for i in range(num_layers):
                    h = model.decoder_transformer.layers[i](h)
                if model.decoder_transformer.norm is not None:
                    h = model.decoder_transformer.norm(h)
                h = model.embed_to_patch(h)
                out = h.reshape(B, 1, L_dec * model.patch_size)
                if out.shape[-1] > T_orig:
                    out = out[..., :T_orig]
                elif out.shape[-1] < T_orig:
                    out = torch.nn.functional.pad(out, (0, T_orig - out.shape[-1]))
                return out
            return intermediate, _continue_latent, T_orig

        # Run decoder layers up to specified index
        num_layers = len(model.decoder_transformer.layers)
        layer_idx = int(layer_name.split("_L")[-1])
        for i in range(layer_idx + 1):
            x_dec = model.decoder_transformer.layers[i](x_dec)

        intermediate = x_dec.clone()

        def _continue_dec(feat: torch.Tensor) -> torch.Tensor:
            h = feat
            for i in range(layer_idx + 1, num_layers):
                h = model.decoder_transformer.layers[i](h)
            if model.decoder_transformer.norm is not None:
                h = model.decoder_transformer.norm(h)
            h = model.embed_to_patch(h)
            out = h.reshape(B, 1, L_dec * model.patch_size)
            if out.shape[-1] > T_orig:
                out = out[..., :T_orig]
            elif out.shape[-1] < T_orig:
                out = torch.nn.functional.pad(out, (0, T_orig - out.shape[-1]))
            return out

        return intermediate, _continue_dec, T_orig

    # ------------------------------------------------------------------
    #  Conv: Encoder probing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _conv_encode_to_layer(
        self, audio_np: np.ndarray, layer_name: str,
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], int]:
        model = self.model
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        T_orig = x.shape[-1]

        encoder_blocks = {
            "enc_block1": model.enc_block1,
            "enc_block2": model.enc_block2,
            "enc_block3": model.enc_block3,
            "enc_block4": model.enc_block4,
        }
        block_order = ["enc_block1", "enc_block2", "enc_block3", "enc_block4"]
        target_idx = block_order.index(layer_name)

        h = x
        for i in range(target_idx + 1):
            h = encoder_blocks[block_order[i]](h)

        intermediate = h.clone()

        def _continue(feat: torch.Tensor) -> torch.Tensor:
            h2 = feat
            for i in range(target_idx + 1, len(block_order)):
                h2 = encoder_blocks[block_order[i]](h2)
            z = model.enc_bottleneck(h2)
            return model.decode(z, original_length=T_orig)

        return intermediate, _continue, T_orig

    # ------------------------------------------------------------------
    #  Conv: Decoder probing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _conv_decode_to_layer(
        self, audio_np: np.ndarray, layer_name: str,
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], int]:
        model = self.model
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        T_orig = x.shape[-1]

        z = model.encode(x)

        decoder_blocks = {
            "dec_expand": model.dec_expand,
            "dec_block1": model.dec_block1,
            "dec_block2": model.dec_block2,
            "dec_block3": model.dec_block3,
        }
        block_order = ["dec_expand", "dec_block1", "dec_block2", "dec_block3"]
        target_idx = block_order.index(layer_name)

        h = z
        for i in range(target_idx + 1):
            h = decoder_blocks[block_order[i]](h)

        intermediate = h.clone()

        def _continue(feat: torch.Tensor) -> torch.Tensor:
            h2 = feat
            for i in range(target_idx + 1, len(block_order)):
                h2 = decoder_blocks[block_order[i]](h2)
            out = model.dec_block4(h2)
            if out.shape[-1] > T_orig:
                out = out[..., :T_orig]
            elif out.shape[-1] < T_orig:
                out = torch.nn.functional.pad(out, (0, T_orig - out.shape[-1]))
            return out

        return intermediate, _continue, T_orig

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def probe_layer(
        self,
        audio_np: np.ndarray,
        module_type: str,
        layer_name: str,
    ) -> Tuple[np.ndarray, int]:
        """Extract features at *layer_name* and continue through pipeline.

        Returns
        -------
        audio_out : np.ndarray  shape (T,) — decoded audio.
        orig_len : int — original audio length.
        """
        if self.model_type == "transformer":
            if module_type == "encoder":
                feat, continue_fn, orig_len = self._transformer_encode_to_layer(
                    audio_np, layer_name,
                )
            else:
                feat, continue_fn, orig_len = self._transformer_decode_to_layer(
                    audio_np, layer_name,
                )
        else:
            if module_type == "encoder":
                feat, continue_fn, orig_len = self._conv_encode_to_layer(
                    audio_np, layer_name,
                )
            else:
                feat, continue_fn, orig_len = self._conv_decode_to_layer(
                    audio_np, layer_name,
                )

        out_tensor = continue_fn(feat)
        out_np = out_tensor.squeeze().cpu().numpy()
        if len(out_np) > orig_len:
            out_np = out_np[:orig_len]
        elif len(out_np) < orig_len:
            out_np = np.pad(out_np, (0, orig_len - len(out_np)))
        return out_np.astype(np.float32), orig_len

    @torch.no_grad()
    def full_reconstruct(self, audio_np: np.ndarray) -> np.ndarray:
        """Full encode-decode pass for baseline comparison."""
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)
        T_orig = x.shape[-1]
        out = self.model(x)
        out_np = out.squeeze().cpu().numpy()
        if len(out_np) > T_orig:
            out_np = out_np[:T_orig]
        elif len(out_np) < T_orig:
            out_np = np.pad(out_np, (0, T_orig - len(out_np)))
        return out_np.astype(np.float32)


# =============================================================================
#  4.  Plotting
# =============================================================================


def plot_spectrogram_grid(
    specs: Dict[str, np.ndarray],
    sample_rate: int,
    output_path: Path,
    max_freq: int = 8000,
    title_prefix: str = "",
) -> None:
    """Plot a grid of log-magnitude spectrograms.

    Parameters
    ----------
    specs : dict  {layer_label: spec_array (n_bins, n_frames)}
    sample_rate : int
    output_path : Path — where to save the PNG.
    max_freq : int — maximum frequency (Hz) to display.
    title_prefix : str — optional prefix for the suptitle.
    """
    n_plots = len(specs)
    if n_plots == 0:
        print("  ⚠ No spectrograms to plot.")
        return

    # Arrange in a grid: ~4 columns
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"{title_prefix}Layer-wise Spectrogram Probing",
        fontsize=14, fontweight="bold",
    )

    labels = list(specs.keys())
    for idx, label in enumerate(labels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        spec_db = specs[label]
        n_bins, n_frames = spec_db.shape
        freq_bins = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
        times = np.arange(n_frames) * HOP_LENGTH / sample_rate

        # Limit to max_freq
        freq_mask = freq_bins <= max_freq
        spec_cropped = spec_db[freq_mask, :]
        freq_cropped = freq_bins[freq_mask]

        # Determine vmin/vmax from the data (skip extreme outliers)
        vmin = np.percentile(spec_cropped, 5)
        vmax = np.percentile(spec_cropped, 95)

        im = ax.pcolormesh(
            times, freq_cropped, spec_cropped,
            shading="auto", cmap="magma",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(labels[idx], fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)" if row == n_rows - 1 else "")
        ax.set_ylabel("Freq (Hz)" if col == 0 else "")
        ax.set_ylim(0, max_freq)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved spectrogram grid → {output_path}")


# =============================================================================
#  5.  Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise Feature Probing via Spectrogram Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory of audio files (optional if --audio_path given).",
    )
    parser.add_argument(
        "--audio_path", type=str, default=None,
        help="Path to a single audio file. Overrides --data_dir random selection.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="toy_evaluation/eval_spectrograms",
        help="Directory for output plots and optional .wav files.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda, cpu, mps).",
    )
    parser.add_argument(
        "--save_wav", action="store_true",
        help="Also save per-layer reconstructed .wav files for listening.",
    )
    parser.add_argument(
        "--max_freq", type=int, default=8000,
        help="Maximum frequency (Hz) to display in spectrograms.",
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    if args.audio_path is None and args.data_dir is None:
        parser.error("Either --audio_path or --data_dir must be provided.")

    # ---- Setup ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Detect model type ----
    model_type = detect_model_type(args.checkpoint_path, device)
    if model_type == "conv":
        print("  ⚠ Conv autoencoder detected. Layer spectrogram probing works "
              "but has fewer interception points than the Transformer model.")
    print(f"[Model] {model_type} autoencoder")

    # ---- Load model ----
    print(f"[Loading] {args.checkpoint_path}")
    wrapper = LayerProbingWrapper(args.checkpoint_path, device, model_type)

    # ---- Load audio ----
    clip_samples = int(CLIP_DURATION * MODEL_SAMPLE_RATE)

    if args.audio_path:
        print(f"[Audio] Loading: {args.audio_path}")
        audio = load_mono_clip(args.audio_path, clip_samples, centre=True)
        stem_name = Path(args.audio_path).stem
    else:
        # Pick the first valid audio file from data_dir
        audio_path = None
        for root, _dirs, files in os.walk(args.data_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_STEMS:
                    audio_path = os.path.join(root, fname)
                    break
            if audio_path:
                break
        if audio_path is None:
            raise FileNotFoundError(f"No audio files found in {args.data_dir}")
        print(f"[Audio] Loading: {audio_path}")
        audio = load_mono_clip(audio_path, clip_samples, centre=True)
        stem_name = Path(audio_path).stem

    print(f"[Audio] Shape: {audio.shape}, SR: {MODEL_SAMPLE_RATE}, "
          f"Duration: {len(audio) / MODEL_SAMPLE_RATE:.2f}s")

    # ---- Full reconstruction baseline ----
    print("\nComputing full reconstruction baseline...")
    rec_baseline = wrapper.full_reconstruct(audio)

    # ---- Probe all layers ----
    plan = wrapper.get_interception_plan()
    print(f"\nProbing {len(plan)} interception points...")

    specs: Dict[str, np.ndarray] = {}
    wav_outputs: Dict[str, np.ndarray] = {}

    # Original audio spectrogram first
    specs["original"] = compute_log_mel_or_stft(audio)
    wav_outputs["original"] = audio

    # Full reconstruction
    specs["full_recon"] = compute_log_mel_or_stft(rec_baseline)
    wav_outputs["full_recon"] = rec_baseline

    for module_type, layer_name in plan:
        label = f"{'enc' if module_type == 'encoder' else 'dec'}:{layer_name}"
        print(f"  → {label} ...", end=" ", flush=True)
        try:
            out_audio, _ = wrapper.probe_layer(audio, module_type, layer_name)
            spec = compute_log_mel_or_stft(out_audio)
            specs[label] = spec
            wav_outputs[label] = out_audio
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # ---- Save spectrogram grid ----
    print(f"\nPlotting {len(specs)} spectrograms...")
    plot_spectrogram_grid(
        specs,
        MODEL_SAMPLE_RATE,
        out_dir / "layer_spectrograms.png",
        max_freq=args.max_freq,
        title_prefix=f"[{wrapper.name}] ",
    )

    # ---- Save WAV files ----
    if args.save_wav:
        wav_dir = out_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        for label, wav_audio in wav_outputs.items():
            safe_label = label.replace(":", "_")
            path = wav_dir / f"{stem_name}_{safe_label}.wav"
            wavfile.write(path, MODEL_SAMPLE_RATE, wav_audio.astype(np.float32))
        print(f"  ✓ Saved {len(wav_outputs)} .wav files → {wav_dir}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Layer Spectrogram Probing — Complete")
    print(f"  Model:       {wrapper.name}")
    print(f"  Audio:       {stem_name}")
    print(f"  Layers:      {len(specs)} spectrograms")
    print(f"  Output:      {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
