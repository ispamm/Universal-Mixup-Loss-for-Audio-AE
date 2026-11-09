"""
Transformer Autoencoder  (SAME-inspired, RoPE + Non-overlapping Patches)
=============================================================================

A Transformer-based continuous-latent autoencoder for raw audio waveforms,
aligned with the **SAME** paper architecture (Section 2).

Architecture (revised)
----------------------
- **Non-overlapping Patch Embedding**: Mono audio ``(B, 1, T)`` is padded to a
  multiple of ``patch_size``, reshaped to ``(B, L, patch_size)``, then linearly
  projected to ``d_model``.  **No overlap, no learned convolution** — this
  matches the SAME paper's parameter-free patching (Section 2.1) and eliminates
  the OLA-related fixed-frequency tone.
- **Rotary Position Embeddings (RoPE)**: Applied inside the self-attention
  mechanism on Q and K, replacing the additive sinusoidal encoding.  RoPE
  (used in the SAME paper, Section 2.2) encodes relative position without
  injecting a deterministic bias into the embeddings, preventing tonal leakage.
- **Encoder Transformer**: 6 custom pre-norm layers with RoPE-based multi-head
  self-attention and activation-gated GLU feed-forward networks.  Output
  projections are zero-initialised so each layer starts as identity (SAME
  paper, Section 2.2).
- **Continuous Latent Bottleneck**: Linear ``d_model → latent_dim → d_model``.
  Bias disabled on patch projections to prevent periodic artifacts; enabled on
  bottleneck projections for centering flexibility.
- **Decoder Transformer**: 6 layers, symmetric with the encoder.
- **Hybrid Activation (SAME paper, Section 2.2)**: periodic activations (Snake)
  are detrimental when applied across the whole model, but critical when placed
  selectively at the end of the decoder to provide a periodic inductive bias for
  fine waveform reconstruction.  The Encoder and the initial Decoder layers
  therefore use a standard base activation (``leaky_relu`` or ``silu``), while
  only the final ``decoder_num_snake_layers`` Decoder layers use
  ``SnakeActivation``.
- **Patch Unembedding**: Linear ``d_model → patch_size``, reshaped back to
  waveform ``(B, 1, T_padded)``, cropped to original length.

Input / Output
--------------
- Input:  ``(B, 1, T)``  — mono audio
- Output: ``(B, 1, T)``  — reconstructed mono audio

Usage
-----
    from model.transformer_autoencoder import TransformerAutoencoder

    model = TransformerAutoencoder(latent_dim=64)
    audio = torch.randn(2, 1, 44100 * 3)
    reconstructed = model(audio)

    z = model.encode(audio)                # (B, L, 64)
    rec = model.decode(z, original_length=132300)
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activations import SnakeActivation


# =============================================================================
#  Rotary Position Embedding  (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE applied per-head to queries and keys before attention.

    Uses the standard formulation: given head_dim, frequencies are computed
    as  θ_i = 10000^{-2i/head_dim}, and each position m receives
    cos(m·θ) and sin(m·θ) rotation.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head (must be even).
    max_len : int
        Maximum precomputed sequence length (default 8192).
    theta : float
        Base frequency (default 10000.0).
    """

    def __init__(
        self,
        head_dim: int,
        max_len: int = 8192,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.head_dim = head_dim

        # Inverse frequencies:  1 / (theta^{2i/head_dim})
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos / sin for all positions
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)                      # (max_len, head_dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)               # (max_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)  # (max_len, head_dim)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Apply the half-rotation:  [x1, x2] → [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to a per-head tensor.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, nhead, L, head_dim)``

        Returns
        -------
        torch.Tensor  same shape, rotated.
        """
        seq_len = x.shape[2]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)   # (1, 1, L, head_dim)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


# =============================================================================
#  Custom Transformer Encoder Layer  (RoPE + SiLU-GLU, pre-norm)
# =============================================================================

class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm transformer layer with RoPE in self-attention and an
    **activation-gated** GLU feed-forward network.

    The FFN gate uses the activation selected by *activation*:

    - ``"leaky_relu"`` — LeakyReLU-GLU (default),
    - ``"silu"``      — SiLU-GLU,
    - ``"snake"``     — periodic Snake-GLU (SAME §2.2; placed only at the end
      of the decoder pipeline to bias fine waveform reconstruction).

    ``SnakeActivation`` is **only instantiated when ``activation == "snake"``**,
    so non-snake layers expose no learnable Snake parameters.

    Output projections are zero-initialised so the layer starts as identity,
    matching the SAME paper's training stabilisation strategy.

    Parameters
    ----------
    d_model : int
        Transformer embedding dimension.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        Hidden dimension of the GLU FFN.
    dropout : float
        Dropout probability (applied after attention and FFN).
    max_len : int
        Maximum precomputed RoPE sequence length.
    activation : str
        FFN gate activation: ``"leaky_relu"``, ``"silu"`` or ``"snake"``.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        max_len: int = 8192,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.activation = activation

        # ---- Periodic Snake activation (FFN gate) — created only on request ----
        # Conditional creation is the core of the hybrid activation design
        # (SAME §2.2): non-snake layers must not carry learnable Snake params.
        self._snake_act = (
            SnakeActivation(dim_feedforward) if activation == "snake" else None
        )

        # ---- RoPE ----
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len)

        # ---- Self-Attention projections ----
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ---- SiLU-GLU Feed-Forward ----
        self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

        # ---- Dropout ----
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # ---- Layer Norms  (pre-norm) ----
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ---- Weight initialisation ----
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform for Q/K/V/gate/up; zero-init for output projections."""
        for proj in [self.q_proj, self.k_proj, self.v_proj,
                     self.gate_proj, self.up_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)

        # Zero-init branch outputs → layer starts as identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.down_proj.weight)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One transformer layer (pre-norm).

        Parameters
        ----------
        src : (B, L, d_model)
        src_mask : optional additive attention mask (L, L) or (B·nhead, L, L)
        src_key_padding_mask : optional (B, L) boolean mask

        Returns
        -------
        torch.Tensor  (B, L, d_model)
        """
        # ---- Self-Attention (pre-norm) ----
        residual = src
        src_normed = self.norm1(src)
        attn_out = self._sa_block(src_normed, src_mask, src_key_padding_mask)
        src = residual + self.dropout_attn(attn_out)

        # ---- FFN (pre-norm) ----
        residual = src
        src_normed = self.norm2(src)
        ffn_out = self._ff_block(src_normed)
        src = residual + self.dropout_ffn(ffn_out)

        return src

    # ------------------------------------------------------------------
    #  Self-Attention with RoPE
    # ------------------------------------------------------------------

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, L, E = x.shape
        H = self.nhead
        D = self.head_dim

        # Project and reshape to (B, H, L, D)
        Q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        K = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        # Apply RoPE to Q and K only
        Q = self.rope(Q)
        K = self.rope(K)

        # Scaled dot-product attention
        scale = D ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attn_mask is not None:
            # attn_mask can be (L, L) or (B*H, L, L) – broadcast as needed
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # (B, L) → (B, 1, 1, L) → broadcast over heads and queries
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)          # (B, H, L, D)

        # Merge heads back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(attn_out)

    # ------------------------------------------------------------------
    #  SiLU-GLU Feed-Forward
    # ------------------------------------------------------------------

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Gated linear unit:  activation(gate_proj) ⊙ up_proj → down_proj.

        The gate activation follows ``self.activation``: LeakyReLU (default),
        SiLU, or periodic Snake (present only in decoder-tail layers).
        """
        gate = self.gate_proj(x)
        if self.activation == "snake":
            gate = self._snake_act(gate)
        elif self.activation == "silu":
            gate = F.silu(gate)
        else:
            gate = F.leaky_relu(gate, negative_slope=0.2)
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# =============================================================================
#  Transformer Encoder Stack  (heterogeneous per-layer activations)
# =============================================================================

class RoPETransformerEncoder(nn.Module):
    """Stack of ``RoPETransformerEncoderLayer`` with an optional final
    LayerNorm, mirroring ``nn.TransformerEncoder`` but with RoPE layers.

    Supports **heterogeneous per-layer activations** (hybrid design, SAME
    §2.2).  Two construction modes are accepted:

    - ``layers=`` — a pre-built list of layer instances; or
    - ``encoder_layer=`` + ``num_layers=`` + ``activation_list=`` — the
      template is re-instantiated once per layer with the activation given at
      that position, so activation-specific sub-modules (e.g.
      ``SnakeActivation``) are created only where needed.
    """

    def __init__(
        self,
        encoder_layer: RoPETransformerEncoderLayer | None = None,
        num_layers: int | None = None,
        activation_list: Sequence[str] | None = None,
        layers: Sequence[nn.Module] | None = None,
        norm: nn.Module | None = None,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.norm = norm

        if layers is not None:
            self.layers = nn.ModuleList(list(layers))
            self.activation = (
                getattr(self.layers[0], "activation", activation)
                if len(self.layers) > 0 else activation
            )
            return

        if encoder_layer is None or num_layers is None:
            raise ValueError(
                "RoPETransformerEncoder needs either 'layers' or "
                "('encoder_layer', 'num_layers' [, 'activation_list'])."
            )
        if activation_list is None:
            activation_list = [activation] * num_layers
        if len(activation_list) != num_layers:
            raise ValueError(
                f"activation_list length ({len(activation_list)}) must match "
                f"num_layers ({num_layers})."
            )

        # Re-instantiate the template once per layer with the per-position
        # activation, so that activation-specific sub-modules are created
        # only where needed.
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.dim_feedforward,
                dropout=encoder_layer.dropout,
                max_len=encoder_layer.max_len,
                activation=act,
            )
            for act in activation_list
        ])
        self.activation = activation_list[0] if activation_list else activation

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# =============================================================================
#  SAME Soft-Normalisation Bottleneck  (Section 2.3)
# =============================================================================

class SoftNormBottleneck(nn.Module):
    """SAME Soft-Normalisation Bottleneck (SAME paper, Section 2.3).

    Replaces the VAE bottleneck with a lightweight normalisation scheme:

    1. **Per-channel affine transform**: ``z = scale * z + bias`` with learnable
       ``scale`` and ``bias`` per latent channel.
    2. **Standard-deviation normalisation**: divide by an EMA-tracked running
       standard deviation (or the current batch std during training).
    3. **Gaussian noise injection**: add noise scaled by the active standard
       deviation — stronger during training (0.05×) than inference (0.001×).

    The clean normalised latent (pre-noise) is stored internally for KL-loss
    computation in the training loop.

    Parameters
    ----------
    latent_dim : int
        Number of latent channels.
    ema_decay : float
        Exponential moving average decay for ``running_std`` (default 0.99).
    """

    def __init__(self, latent_dim: int, ema_decay: float = 0.99) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Learnable per-channel affine parameters
        self.scale = nn.Parameter(torch.ones(latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))

        # EMA-tracked running standard deviation (scalar)
        self.register_buffer("running_std", torch.ones(1))

        # Temporary storage for the clean normalised latent (pre-noise),
        # populated during forward() and consumed by the KL-loss computation.
        self.z_norm_clean: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply soft-norm: affine → normalise → noise.

        Parameters
        ----------
        z : torch.Tensor  shape ``(B, L, D)`` — raw latent from encoder.

        Returns
        -------
        torch.Tensor  shape ``(B, L, D)`` — normalised + noised latent.
        """
        B, L, D = z.shape

        # ---- Per-channel affine ----
        z = self.scale.view(1, 1, D) * z + self.bias.view(1, 1, D)

        # ---- Standard-deviation normalisation ----
        if self.training:
            batch_std = z.std()
            with torch.no_grad():
                self.running_std = (
                    self.ema_decay * self.running_std
                    + (1.0 - self.ema_decay) * batch_std.detach()
                )
            active_std = batch_std
        else:
            active_std = self.running_std

        z_norm = z / (active_std + 1e-8)

        # Store clean normalised latent for KL-loss computation
        self.z_norm_clean = z_norm

        # ---- Gaussian noise injection ----
        noise_scale = 0.05 if self.training else 0.001
        noise = noise_scale * active_std * torch.randn_like(z_norm)
        z_noised = z_norm + noise

        return z_noised

    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reverse the normalisation: multiply by the running standard
        deviation so the decoder receives latents in the original scale.

        Parameters
        ----------
        z : torch.Tensor  shape ``(B, L, D)`` — normalised latent.

        Returns
        -------
        torch.Tensor  shape ``(B, L, D)`` — denormalised latent.
        """
        return z * (self.running_std + 1e-8)


# =============================================================================
#  TransformerAutoencoder
# =============================================================================

class TransformerAutoencoder(nn.Module):
    """Transformer autoencoder with non-overlapping patch embedding, RoPE,
    and a continuous latent bottleneck.

    Parameters
    ----------
    latent_dim : int
        Bottleneck channels (default 64).
    d_model : int
        Transformer embedding dimension (default 512).
    nhead : int
        Attention heads (default 8).
    num_layers : int
        Encoder / decoder transformer layers (default 6).
    dim_feedforward : int
        GLU hidden dimension (default 2048).
    dropout : float
        Dropout probability (default 0.0).
    patch_size : int
        Non-overlapping waveform samples per patch (default 256).
        With sr=22050, this gives ≈ 86 frames per second.
    max_len : int
        Maximum precomputed positional encoding length (default 8192).
    activation : str
        Base activation for the Encoder and initial Decoder layers:
        ``"leaky_relu"`` (default) or ``"silu"``.
    decoder_num_snake_layers : int
        Number K of final Decoder layers equipped with the periodic
        ``SnakeActivation`` (SAME §2.2).  The first ``num_layers - K``
        Decoder layers use ``activation``; the final K use ``"snake"``.
        Set to 0 to disable Snake entirely.
    encoder_final_norm : bool
        If True (default), apply a final LayerNorm after the encoder
        transformer stack.  Set to False for a **scale-sensitive**
        bottleneck that preserves energy differences between latent
        paths (e.g. LERP vs TRIG mixup), removing the LayerNorm trap
        that crushes magnitude variations.
    decoder_final_norm : bool
        If True (default), apply a final LayerNorm after the decoder
        transformer stack.  Set to False for scale-sensitive decoding.
        Pre-norm inside each ``RoPETransformerEncoderLayer`` is always
        active for training stability regardless of this flag.
    use_soft_norm : bool
        If True, replace the plain linear bottleneck with the SAME
        Soft-Normalisation Bottleneck (EMA std tracking, per-channel
        affine, Gaussian noise injection).  Default False preserves
        backwards compatibility.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        patch_size: int = 256,
        max_len: int = 8192,
        activation: str = "leaky_relu",
        decoder_num_snake_layers: int = 2,
        encoder_final_norm: bool = True,
        decoder_final_norm: bool = True,
        use_soft_norm: bool = False,
    ) -> None:
        super().__init__()

        if activation not in ("leaky_relu", "silu"):
            raise ValueError(
                f"activation must be 'leaky_relu' or 'silu' for the base "
                f"(encoder / early-decoder) layers, got {activation!r}. "
                f"Periodic Snake is applied via decoder_num_snake_layers."
            )
        if not (0 <= decoder_num_snake_layers <= num_layers):
            raise ValueError(
                f"decoder_num_snake_layers ({decoder_num_snake_layers}) must be "
                f"in [0, num_layers ({num_layers})]."
            )

        self.patch_size = patch_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.activation = activation
        self.decoder_num_snake_layers = decoder_num_snake_layers
        self.encoder_final_norm = encoder_final_norm
        self.decoder_final_norm = decoder_final_norm
        self.use_soft_norm = use_soft_norm

        # ------------------------------------------------------------------
        #  Non-overlapping Patch Embedding
        # ------------------------------------------------------------------
        #  (B, 1, T) → pad → (B, L, patch_size) → Linear → (B, L, d_model)
        # ------------------------------------------------------------------
        self.patch_to_embed = nn.Linear(patch_size, d_model, bias=False)

        # ------------------------------------------------------------------
        #  Encoder Transformer  (every layer uses the base activation)
        # ------------------------------------------------------------------
        enc_layer = RoPETransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            activation=activation,
        )
        # Final LayerNorm removed when encoder_final_norm=False →
        # scale-sensitive bottleneck for energy-preserving latent mixup.
        # Pre-norm inside RoPETransformerEncoderLayer remains active.
        self.encoder_transformer = RoPETransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            activation_list=[activation] * num_layers,
            norm=nn.LayerNorm(d_model) if encoder_final_norm else None,
        )

        # ------------------------------------------------------------------
        #  Bottleneck:  d_model → latent_dim  /  latent_dim → d_model
        # ------------------------------------------------------------------
        self.enc_to_latent = nn.Linear(d_model, latent_dim, bias=True)
        self.latent_to_dec = nn.Linear(latent_dim, d_model, bias=True)

        # ------------------------------------------------------------------
        #  SAME Soft-Normalisation Bottleneck (optional)
        # ------------------------------------------------------------------
        # When enabled, replaces the plain linear bottleneck with EMA-tracked
        # std normalisation, per-channel affine, and Gaussian noise injection.
        self.soft_norm: SoftNormBottleneck | None = None
        if use_soft_norm:
            self.soft_norm = SoftNormBottleneck(latent_dim)

        # ------------------------------------------------------------------
        #  Decoder Transformer  (hybrid activation, SAME §2.2)
        # ------------------------------------------------------------------
        # The first (num_layers - K) layers use the base activation; the final
        # K layers use periodic Snake to bias fine waveform reconstruction.
        dec_activation_list = (
            [activation] * (num_layers - decoder_num_snake_layers)
            + ["snake"] * decoder_num_snake_layers
        )
        dec_layer = RoPETransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            activation=activation,
        )
        self.decoder_transformer = RoPETransformerEncoder(
            dec_layer,
            num_layers=num_layers,
            activation_list=dec_activation_list,
            norm=nn.LayerNorm(d_model) if decoder_final_norm else None,
        )

        # ------------------------------------------------------------------
        #  Patch Unembedding
        # ------------------------------------------------------------------
        #  (B, L, d_model) → Linear → (B, L, patch_size) → (B, 1, T)
        # ------------------------------------------------------------------
        self.embed_to_patch = nn.Linear(d_model, patch_size, bias=False)

        # ------------------------------------------------------------------
        #  Final initialisation  (bottleneck: Xavier; patch proj: Xavier)
        # ------------------------------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform for bottleneck and patch embedding layers."""
        for mod in [self.patch_to_embed, self.enc_to_latent,
                    self.latent_to_dec, self.embed_to_patch]:
            nn.init.xavier_uniform_(mod.weight, gain=1.0)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

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
        z = self.encode(x)                         # (B, L, latent_dim)
        out = self.decode(z, original_length=T_orig)  # (B, 1, T_orig)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform to continuous latent representation.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, 1, T)``

        Returns
        -------
        torch.Tensor  shape ``(B, L, latent_dim)`` where L = ceil(T / patch_size).
        """
        B, C, T = x.shape

        # Pad to multiple of patch_size
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # Non-overlapping reshape:  (B, 1, T_pad) → (B, L, patch_size)
        L = x.shape[-1] // self.patch_size
        x = x.reshape(B, L, self.patch_size)

        # Linear projection to d_model
        x = self.patch_to_embed(x)                  # (B, L, d_model)

        # Encoder transformer
        x = self.encoder_transformer(x)             # (B, L, d_model)

        # Bottleneck projection
        z = self.enc_to_latent(x)                   # (B, L, latent_dim)

        # Optional SAME Soft-Normalisation Bottleneck
        if self.soft_norm is not None:
            z = self.soft_norm(z)                   # (B, L, latent_dim) — normalised + noised

        return z

    def decode(
        self,
        z: torch.Tensor,
        original_length: int | None = None,
    ) -> torch.Tensor:
        """Decode latent representation back to waveform.

        Parameters
        ----------
        z : torch.Tensor  shape ``(B, L, latent_dim)``
        original_length : int, optional
            Crop the output to this exact length.

        Returns
        -------
        torch.Tensor  shape ``(B, 1, T)``
        """
        B, L, _ = z.shape

        # Reverse Soft-Normalisation before expanding to d_model
        if self.soft_norm is not None:
            z = self.soft_norm.reverse(z)           # (B, L, latent_dim) — denormalised

        # Expand from latent to d_model
        x = self.latent_to_dec(z)                   # (B, L, d_model)

        # Decoder transformer
        x = self.decoder_transformer(x)             # (B, L, d_model)

        # Linear projection to patch_size
        x = self.embed_to_patch(x)                  # (B, L, patch_size)

        # Reshape back to waveform:  (B, L, patch_size) → (B, 1, T_pad)
        out = x.reshape(B, 1, L * self.patch_size)

        # Crop to original length if requested
        if original_length is not None:
            if out.shape[-1] > original_length:
                out = out[..., :original_length]
            elif out.shape[-1] < original_length:
                out = F.pad(out, (0, original_length - out.shape[-1]))

        return out
