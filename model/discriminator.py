"""
STFT Discriminator  (PatchGAN-style, single-scale)
====================================================

A lightweight convolutional discriminator operating on the **magnitude STFT**
of the input waveform.  Designed for adversarial training of audio
autoencoders — penalises high-frequency artefacts (whistling / amplitude
modulation) that MRSTFT reconstruction losses alone cannot suppress.

Architecture
------------
1. Compute magnitude STFT  (``n_fft=1024``, ``hop_length=256``, Hann window)
   → spectrogram ``(B, 1, F, T_frames)``.
2. Stack of 4 ``Conv2d`` layers  (stride=2, 32 hidden channels, LeakyReLU)
   progressively reducing the spatial resolution.
3. Final ``Conv2d`` projection to a single-channel logit map (PatchGAN).
4. Returns **both** the list of intermediate feature maps (for optional
   feature matching) and the final logit map.

Usage
-----
    from model.discriminator import STFTDiscriminator

    D = STFTDiscriminator()
    audio = torch.randn(4, 1, 44100)          # (B, 1, T)
    feats, logits = D(audio)                  # feats: list[Tensor], logits: (B,1,H,W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  STFTDiscriminator
# =============================================================================

class STFTDiscriminator(nn.Module):
    """Lightweight STFT-domain PatchGAN discriminator.

    Parameters
    ----------
    n_fft : int
        FFT size for the STFT (default 1024).
    hop_length : int
        Hop size for the STFT (default 256).
    hidden_channels : int
        Number of channels in the hidden Conv2d layers (default 32).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        # Hann window (persistent buffer so it moves with the module)
        self.register_buffer("window", torch.hann_window(n_fft))

        # ------------------------------------------------------------------
        #  Convolutional stack  (4 layers, stride-2 downsampling)
        # ------------------------------------------------------------------
        self.convs = nn.ModuleList([
            nn.Conv2d(1,              hidden_channels, 3, stride=2, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
        ])

        # Final projection → single-channel logit map
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Normal-weight initialisation for all Conv2d layers."""
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Compute feature maps and logit map for a batch of waveforms.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, 1, T)`` — mono audio waveforms.

        Returns
        -------
        feats : list[torch.Tensor]
            Intermediate feature maps after each ``Conv2d + LeakyReLU``.
            Shapes: ``(B, 32, F_i, N_i)`` with decreasing spatial dims.
        logits : torch.Tensor  shape ``(B, 1, F_out, N_out)``
            PatchGAN logit map (real / fake per spatial location).
        """
        # ---- 1. Magnitude STFT ----
        x_wav = x.squeeze(1)  # (B, T)

        X = torch.stft(
            x_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # (B, n_fft//2+1, n_frames)

        X_mag = X.abs().unsqueeze(1)  # (B, 1, F, N_frames)

        # ---- 2. Convolutional feature extraction ----
        feats: list[torch.Tensor] = []
        h = X_mag
        for conv in self.convs:
            h = conv(h)
            h = self.activation(h)
            feats.append(h)

        # ---- 3. Final logit projection ----
        logits = self.final_conv(h)  # (B, 1, F_out, N_out)

        return feats, logits


# =============================================================================
#  Quick self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  STFTDiscriminator — Self-Test")
    print("=" * 60)

    B, T = 2, 44100  # 2 s @ 22050 Hz
    audio = torch.randn(B, 1, T)

    D = STFTDiscriminator()
    feats, logits = D(audio)

    print(f"\nInput shape:      {audio.shape}")
    print(f"Logits shape:     {logits.shape}")
    for i, f in enumerate(feats):
        print(f"  Feature [{i}]:    {f.shape}")

    n_params = sum(p.numel() for p in D.parameters())
    print(f"\nTotal parameters:  {n_params:,}")

    # Quick sanity: logits should have a spatial structure
    assert logits.ndim == 4, "Logits must be 4D (B, 1, H, W)"
    assert logits.shape[0] == B, "Batch dim mismatch"
    assert len(feats) == 4, "Expected 4 feature maps"

    print("\n✓  All assertions passed!")
