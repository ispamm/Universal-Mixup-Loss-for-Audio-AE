"""
Toy 1D Convolutional Autoencoder  (Scaled-Up, ~47.6M parameters)
=================================================================

A high-capacity continuous-latent convolutional autoencoder for raw audio
waveforms.  Designed for fair ablation comparison against a ~40M-parameter
Transformer model (``ToyTransformerAutoencoder``).

Architecture (block-based, 2×Conv per resolution)
--------------------------------------------------
- **Encoder**: 4 blocks of 2×Conv1d + LeakyReLU with progressive stride
  downsampling (total stride = 256), plus a bottleneck projection.
- **Bottleneck**: Continuous latent space of ``latent_dim`` channels
  (default 64).  **No** Vector Quantization (VQ) or discrete bottlenecks.
- **Decoder**: 1 expansion block + 4 upsample blocks, symmetrically
  upsampling back to the original waveform length.

Channel Progression
-------------------
    Encoder:  1 → 256 → 512 → 1024 → 2048 → latent_dim (64)
    Decoder:  latent_dim (64) → 2048 → 1024 → 512 → 256 → 1

Stride Breakdown
----------------
    Block 1:  ×2    →   total so far = 2
    Block 2:  ×4    →   total so far = 8
    Block 3:  ×4    →   total so far = 32
    Block 4:  ×8    →   total so far = 256

Input / Output
--------------
- Input:  ``(B, 1, T)``  — mono audio
- Output: ``(B, 1, T)``  — reconstructed mono audio
- The forward pass automatically crops/pads to keep the exact input length.

Usage
-----
    from toy_model.autoencoder import ToyConvAutoencoder

    model = ToyConvAutoencoder(latent_dim=64)
    audio = torch.randn(2, 1, 16000 * 3)   # (batch=2, mono, 3 s @ 16 kHz)
    reconstructed = model(audio)            # (2, 1, 48000)

    z = model.encode(audio)                # (2, 64, 375)  — latent
    rec = model.decode(z, original_length=48000)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_model.activations import SnakeActivation


# =============================================================================
#  ToyConvAutoencoder  (Scaled-Up)
# =============================================================================

class ToyConvAutoencoder(nn.Module):
    """1D convolutional autoencoder with a continuous latent bottleneck.

    Scaled to ~47.6M parameters —  roughly matching the ~40M Transformer
    baseline —  while preserving the same latent_dim (64) and total stride
    (256) as the Transformer autoencoder.

    Parameters
    ----------
    latent_dim : int
        Number of channels in the latent (bottleneck) representation.
    dropout : float
        Dropout probability applied after each intermediate activation.
        Set to 0.0 to disable (default 0.1 for regularisation).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()

        # Total downsampling factor: 2 × 4 × 4 × 8 = 256
        self.total_stride = 256
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        self.activation = activation

        _drop = [nn.Dropout1d(dropout)] if dropout > 0 else []

        def _act(ch: int) -> nn.Module:
            """Return the configured activation for a given channel count."""
            if activation == "snake":
                return SnakeActivation(ch)
            return nn.LeakyReLU(0.2, inplace=True)

        # ==================================================================
        #  Encoder  (B, 1, T)  →  (B, latent_dim, T // 256)
        # ==================================================================
        # Each block:  stride-conv → act → [Drop] → same-ch conv → act → [Drop]

        self.enc_block1 = nn.Sequential(
            # T → T/2,  1 ch → 256 ch
            nn.Conv1d(1, 256, kernel_size=7, stride=2, padding=3),
            _act(256),
            *_drop,
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            _act(256),
            *_drop,
        )

        self.enc_block2 = nn.Sequential(
            # T/2 → T/8,  256 ch → 512 ch
            nn.Conv1d(256, 512, kernel_size=7, stride=4, padding=3),
            _act(512),
            *_drop,
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            _act(512),
            *_drop,
        )

        self.enc_block3 = nn.Sequential(
            # T/8 → T/32,  512 ch → 1024 ch
            nn.Conv1d(512, 1024, kernel_size=7, stride=4, padding=3),
            _act(1024),
            *_drop,
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
            _act(1024),
            *_drop,
        )

        self.enc_block4 = nn.Sequential(
            # T/32 → T/256,  1024 ch → 2048 ch
            nn.Conv1d(1024, 2048, kernel_size=4, stride=8, padding=1),
            _act(2048),
            *_drop,
            nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding=1),
            _act(2048),
            *_drop,
        )

        # Bottleneck projection (stride 1, no resolution change)
        self.enc_bottleneck = nn.Conv1d(
            2048, latent_dim, kernel_size=3, stride=1, padding=1,
        )

        # ==================================================================
        #  Decoder  (B, latent_dim, T//256)  →  (B, 1, T_padded)
        # ==================================================================

        # Expansion (stride 1, deep processing, no upsampling)
        self.dec_expand = nn.Sequential(
            nn.Conv1d(latent_dim, 2048, kernel_size=3, stride=1, padding=1),
            _act(2048),
        )

        # Block 1:  ×8 upsample  (reverses enc_block4)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose1d(2048, 1024, kernel_size=4, stride=8,
                               padding=1, output_padding=6),
            _act(1024),
            *_drop,
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
            _act(1024),
            *_drop,
        )

        # Block 2:  ×4 upsample  (reverses enc_block3)
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=7, stride=4,
                               padding=3, output_padding=3),
            _act(512),
            *_drop,
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            _act(512),
            *_drop,
        )

        # Block 3:  ×4 upsample  (reverses enc_block2)
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=7, stride=4,
                               padding=3, output_padding=3),
            _act(256),
            *_drop,
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            _act(256),
            *_drop,
        )

        # Block 4:  ×2 upsample → back to 1 channel  (reverses enc_block1)
        self.dec_block4 = nn.ConvTranspose1d(
            256, 1, kernel_size=7, stride=2, padding=3, output_padding=1,
        )

    # ------------------------------------------------------------------
    #  Forward / encode / decode
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio through the autoencoder bottleneck.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, 1, T)``

        Returns
        -------
        torch.Tensor  shape ``(B, 1, T)`` — same length as input.
        """
        T_orig = x.shape[-1]
        z = self.encode(x)               # (B, latent_dim, T//256)
        out = self.decode(z)             # (B, 1, T_padded)
        # Crop / pad to exact original length
        if out.shape[-1] > T_orig:
            out = out[..., :T_orig]
        elif out.shape[-1] < T_orig:
            out = F.pad(out, (0, T_orig - out.shape[-1]))
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform to latent representation.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, 1, T)``

        Returns
        -------
        torch.Tensor  shape ``(B, latent_dim, T // 256)``
        """
        h = self.enc_block1(x)
        h = self.enc_block2(h)
        h = self.enc_block3(h)
        h = self.enc_block4(h)
        return self.enc_bottleneck(h)

    def decode(self, z: torch.Tensor,
               original_length: int | None = None) -> torch.Tensor:
        """Decode latent representation back to waveform.

        Parameters
        ----------
        z : torch.Tensor  shape ``(B, latent_dim, L)``
        original_length : int, optional
            If given, crop/pad the output to this exact length.

        Returns
        -------
        torch.Tensor  shape ``(B, 1, T)``
        """
        h = self.dec_expand(z)
        h = self.dec_block1(h)
        h = self.dec_block2(h)
        h = self.dec_block3(h)
        out = self.dec_block4(h)  # (B, 1, T')
        if original_length is not None:
            if out.shape[-1] > original_length:
                out = out[..., :original_length]
            elif out.shape[-1] < original_length:
                out = F.pad(out, (0, original_length - out.shape[-1]))
        return out


# =============================================================================
#  __main__  — parameter count & smoke test
# =============================================================================

if __name__ == "__main__":
    model = ToyConvAutoencoder(latent_dim=64, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"ToyConvAutoencoder (scaled-up)")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")

    # Smoke test with 2 s of mono audio @ 16 kHz
    x = torch.randn(1, 1, 32000)
    with torch.no_grad():
        z = model.encode(x)
        print(f"  Input shape:   {tuple(x.shape)}")
        print(f"  Latent shape:  {tuple(z.shape)}")
        y = model(x)
        print(f"  Output shape:  {tuple(y.shape)}")
        print(f"  Length match:  {x.shape[-1] == y.shape[-1]}")

    # Also test decode with explicit original_length
    with torch.no_grad():
        y2 = model.decode(z, original_length=32000)
        print(f"  Decode length: {y2.shape[-1]}")

    print("✓ Smoke test passed.")
