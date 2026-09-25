"""
SAME Paper Loss Functions  (Section 3.1)
===========================================

Exact implementation of the loss functions described in:

    **"SAME: A Semantically-Aligned Music Autoencoder"**

All equations are implemented from scratch in pure PyTorch — no external
audio-loss libraries are used.

Key design points
-----------------
- Complex STFT computed at **7 resolutions**: FFT sizes 32, 64, 128, 256,
  512, 1024, 2048, each with 75% overlap (hop = n_fft // 4) and Hann window.
- **Magnitude terms** (per resolution):
  1. Spectral Contrast  (Eq. 3) — Frobenius-normalised linear magnitude.
  2. Adaptive Log-Magnitude (Eq. 4) — L1 on log-compressed magnitudes with
     a per-resolution adaptive scaling factor σ.
- **Phase-derivative terms** (per resolution, only when ``use_phase=True``):
  3. Instantaneous Frequency (Eq. 5) — real part of unit‑phasor alignment
     along the time axis, magnitude-weighted.
  4. Group Delay (Eq. 6) — same along the frequency axis.
  5. Complex Distance (Eq. 7) — element-wise log‑compressed complex error.

Class
-----
- **SAMELoss** : single ``nn.Module`` that accepts ``(pred, target,
  use_phase=False)`` and returns the scalar sum across all resolutions.

Reference
---------
- Parker et al., "SAME: A Semantically-Aligned Music Autoencoder", 2026.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  Constants
# =============================================================================

# 7 FFT resolutions used in the SAME paper
FFT_SIZES: List[int] = [32, 64, 128, 256, 512, 1024, 2048]

# Each hop is 75% overlap:  hop = n_fft // 4
HOP_SIZES: List[int] = [n // 4 for n in FFT_SIZES]


# =============================================================================
#  SAMELoss
# =============================================================================

class SAMELoss(nn.Module):
    """Multi-resolution loss from the SAME paper (Section 3.1).

    When called with ``use_phase=False`` the loss includes only the two
    magnitude terms (spectral contrast + adaptive log-magnitude).  When
    ``use_phase=True`` the phase-derivative terms (IF, GD, complex distance)
    are added.

    Parameters
    ----------
    eps : float
        Small constant for numerical stability (default 1e-8).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

        # Pre-create Hann windows on CPU; moved to device on-the-fly in forward.
        self._windows: List[torch.Tensor] = [
            torch.hann_window(n) for n in FFT_SIZES
        ]

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        use_phase: bool = False,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the total SAME loss summed over all FFT resolutions.

        Parameters
        ----------
        pred : torch.Tensor  shape ``(B, 1, T)`` — reconstructed audio.
        target : torch.Tensor  shape ``(B, 1, T)`` — ground-truth audio.
        use_phase : bool
            If True, include the three phase-derivative terms (IF, GD, CD).
        return_components : bool
            If True, return ``(total_loss, mag_loss)`` as a tuple, where
            ``mag_loss`` is the magnitude-only component (Spectral Contrast +
            Adaptive Log-Magnitude) summed across resolutions.  The STFT is
            still computed **once** — the magnitude terms are reused.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Scalar loss when ``return_components=False`` (default), or
            ``(total_loss, mag_loss)`` when ``return_components=True``.
        """
        # Squeeze channel dim for torch.stft: (B, T)
        pred_wav = pred.squeeze(1)
        target_wav = target.squeeze(1)

        total_loss = 0.0
        mag_loss_sum = 0.0

        for n_fft, hop, hann in zip(FFT_SIZES, HOP_SIZES, self._windows):
            window = hann.to(pred.device)

            # --- Complex STFT  (B, n_fft//2+1, n_frames) ---
            X = torch.stft(
                pred_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True, center=True, pad_mode="reflect",
            )
            Y = torch.stft(
                target_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True, center=True, pad_mode="reflect",
            )

            # ---- Magnitudes ----
            X_mag = X.abs()  # (B, F, N)
            Y_mag = Y.abs()  # (B, F, N)

            # ==============================================================
            #  1. Spectral Contrast  — Eq. (3)
            # ==============================================================
            diff = Y_mag - X_mag
            numer = torch.linalg.norm(diff)
            denom = torch.linalg.norm(X_mag + Y_mag) + self.eps
            L_SC = numer / denom

            # ==============================================================
            #  2. Adaptive Log-Magnitude  — Eq. (4)
            # ==============================================================
            # sigma = sqrt( std(X_mag)^2 + std(Y_mag)^2 )   (computed per-sample
            # over frequency and time, then detached — Eq. (4) of the SAME paper)
            sigma = torch.sqrt(
                X_mag.std(dim=[1, 2], keepdim=True) ** 2
                + Y_mag.std(dim=[1, 2], keepdim=True) ** 2
            ).detach() + self.eps

            X_log = torch.log(X_mag / sigma + 1.0)
            Y_log = torch.log(Y_mag / sigma + 1.0)
            L_LM = F.l1_loss(X_log, Y_log)

            # ==============================================================
            #  3. Phase-Derivative Terms  (only when use_phase=True)
            # ==============================================================

            if use_phase:
                # ----------------------------------------------------------
                #  Cross-frame / cross-bin complex products (Eq. 5 / 6)
                # ----------------------------------------------------------
                R_t_X = X[:, :, 1:] * torch.conj(X[:, :, :-1])  # (B, F, N-1)
                R_t_Y = Y[:, :, 1:] * torch.conj(Y[:, :, :-1])

                R_f_X = X[:, 1:, :] * torch.conj(X[:, :-1, :])  # (B, F-1, N)
                R_f_Y = Y[:, 1:, :] * torch.conj(Y[:, :-1, :])

                # ----------------------------------------------------------
                #  Unit phasors — normalise by complex magnitude of R
                # ----------------------------------------------------------
                eps = self.eps

                U_t_X = R_t_X / (R_t_X.abs() + eps)
                U_t_Y = R_t_Y / (R_t_Y.abs() + eps)

                U_f_X = R_f_X / (R_f_X.abs() + eps)
                U_f_Y = R_f_Y / (R_f_Y.abs() + eps)

                # ----------------------------------------------------------
                #  Mean-normalised magnitude weights
                # ----------------------------------------------------------
                # NOTE: torch.sqrt(0) has infinite gradient → NaN in backprop.
                # Add eps *inside* sqrt and cast to float32 to prevent
                # underflow/overflow in mixed precision (AMP).
                sqrt_eps = 1e-7

                w_t = torch.sqrt(
                    (
                        X_mag[:, :, 1:].to(torch.float32)
                        * X_mag[:, :, :-1].to(torch.float32)
                        * Y_mag[:, :, 1:].to(torch.float32)
                        * Y_mag[:, :, :-1].to(torch.float32)
                    ) + sqrt_eps
                )
                w_t = w_t / (w_t.mean() + eps)

                w_f = torch.sqrt(
                    (
                        X_mag[:, 1:, :].to(torch.float32)
                        * X_mag[:, :-1, :].to(torch.float32)
                        * Y_mag[:, 1:, :].to(torch.float32)
                        * Y_mag[:, :-1, :].to(torch.float32)
                    ) + sqrt_eps
                )
                w_f = w_f / (w_f.mean() + eps)

                # ----------------------------------------------------------
                #  Instantaneous Frequency  — Eq. (5)
                # ----------------------------------------------------------
                #  1 - Re( U_t_X · conj(U_t_Y) )
                L_IF = (w_t * (1.0 - (U_t_X * torch.conj(U_t_Y)).real)).mean()

                # ----------------------------------------------------------
                #  Group Delay  — Eq. (6)
                # ----------------------------------------------------------
                L_GD = (w_f * (1.0 - (U_f_X * torch.conj(U_f_Y)).real)).mean()

                # ----------------------------------------------------------
                #  Complex Distance  — Eq. (7)
                # ----------------------------------------------------------
                diff_sq = (X - Y).abs() ** 2
                sigma_sg = diff_sq.std().detach() + eps
                L_cd = torch.log(diff_sq / sigma_sg + 1.0).mean()

                # ---- Sum phase terms ----
                L_phase = L_IF + L_GD + L_cd

            else:
                L_phase = torch.tensor(0.0, device=X.device)

            # ---- Accumulate across resolutions ----
            total_loss = total_loss + L_SC + L_LM + L_phase
            if return_components:
                mag_loss_sum = mag_loss_sum + L_SC + L_LM

        if return_components:
            return total_loss, mag_loss_sum
        return total_loss


# =============================================================================
#  Gain-range cosine anneal (shared by the implicit & universal losses)
# =============================================================================

def cosine_gain_range(
    gain_min: float,
    gain_max: float,
    progress: float | None,
) -> tuple[float, float]:
    """Effective gain range at training ``progress`` (single source of truth).

    The range is cosine-annealed from the full ``(gain_min, gain_max)`` at
    ``progress = 0`` toward ``(1, 1)`` (no gain / constant-power mixes) at
    ``progress = 1`` — the paper's curriculum: expose the decoder to the whole
    positive-weight cone early, then converge to the constant-power regime
    that the evaluation protocols use.  ``progress = None`` (or the caller
    disabling the anneal) returns the fixed ``(gain_min, gain_max)`` range.

    .. note::
       ``progress`` must be the fraction of the ACTUAL training horizon that
       will be run.  Passing ``(epoch - 1) / epochs`` with a huge safety-ceiling
       ``--epochs`` (e.g. 10000) while early stopping stops the run at ~200
       epochs pins the effective range at its ``progress ≈ 0`` endpoint for
       the whole run — historically this silently collapsed the gain to
       ``A ≈ 1`` and the cone was never exercised.
    """
    if progress is None:
        return float(gain_min), float(gain_max)
    f = 0.5 * (1.0 - torch.cos(torch.tensor(torch.pi * min(float(progress), 1.0))))
    f = float(f)
    g_min = float(gain_min) + (1.0 - float(gain_min)) * f
    g_max = float(gain_max) + (1.0 - float(gain_max)) * f
    return g_min, g_max


# =============================================================================
#  Torres-style implicit linearity (implicit homogeneity + additivity)
# =============================================================================

def torres_implicit_linearity_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    audio: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    *,
    gain_min: float = 0.0,
    gain_max: float = 3.0,
    gain_prob: float = 0.8,
    mix_fraction: float = 0.5,
    use_phase: bool = False,
    progress: float | None = None,
    anneal_gain: bool = True,
) -> torch.Tensor:
    """Implicit linearity regularizer (Torres, Moussallam & Meseguer-Brocal,
    "Learning Linearity in Audio Consistency Autoencoders via Implicit
    Regularization", ICASSP 2026, arXiv:2510.23530) — adapted from the
    consistency-model recipe to a plain autoencoder.

    The objective supervises the decoder on two implicit linearity properties,
    using only the SAME multi-resolution SAME loss (no new metric):

    * **Implicit homogeneity** (a fraction ``1 - mix_fraction`` of the batch):
      a random positive gain ``a`` is applied to the *latent* of the unscaled
      input, ``a·z``, and the decoder must output the gain-scaled audio ``a·x``.
      ``a`` is never given to the model explicitly, so the decoder must infer
      the output scale from the magnitude of the conditioning latent alone.
      With ``a = 1`` this reduces to plain reconstruction.

    * **Implicit additivity** (a fraction ``mix_fraction`` of the batch): the
      batch is circularly shifted by one position to form pairs ``(u, v)``;
      the artificial mixture ``m = u + v`` is the target, but the latent is the
      **sum of the individual latents** ``z_u + z_v`` (NOT ``Enc(m)``) —
      the paper's key trick.  A gain ``a`` is applied to both the summed
      latent and the mixture target.

    Gain sampling follows the paper: ``a ~ U(gain_min, gain_max)`` per sample
    with probability ``gain_prob`` (else ``a = 1``), ``|a| < 0.05`` is snapped
    to 0, and targets are clipped so the peak waveform amplitude does not
    exceed 1.  When ``anneal_gain`` and ``progress`` are given, the gain range
    is annealed from ``(gain_min, gain_max)`` to ``(1, 1)`` over training with
    a cosine ramp (the paper anneals 0→3 down to no gain).

    The regularizer requires a **scale-sensitive bottleneck** (no final
    LayerNorm on the latent), exactly like the trigonometric mixup loss —
    scaling ``z`` by ``a`` or summing two latents must survive the bottleneck
    for the decoder to learn the scale algebra.
    """
    B = audio.shape[0]
    T = audio.shape[-1]

    # ---- in-batch pairs (circular shift by one position, as in the paper) ----
    audio_rolled = torch.roll(audio, shifts=1, dims=0)
    z_rolled = torch.roll(z, shifts=1, dims=0)

    # ---- per-sample gain ----
    if anneal_gain:
        # cosine ramp of each range endpoint toward 1.0 as progress → 1:
        # full (gain_min, gain_max) cone at progress 0, (1, 1) at progress 1.
        g_min, g_max = cosine_gain_range(gain_min, gain_max, progress)
    else:
        g_min, g_max = float(gain_min), float(gain_max)

    a = torch.ones(B, 1, 1, device=device)
    if g_max > g_min:
        use_gain = torch.rand(B, 1, 1, device=device) < gain_prob
        a = torch.where(
            use_gain,
            g_min + (g_max - g_min) * torch.rand(B, 1, 1, device=device),
            a,
        )
    # paper: gains with |a| < 0.05 are set to 0 (silence the sample)
    a = torch.where(a.abs() < 0.05, torch.zeros_like(a), a)

    # ---- mixture assignment ----
    is_mix = torch.rand(B, 1, 1, device=device) < mix_fraction

    # latent the decoder is conditioned on (paper eq. 4, z' = z_x or z_u + z_v)
    latent = torch.where(is_mix, z + z_rolled, z)          # (B, L, D)
    # target waveform the decoder must output (x or u + v)
    target = torch.where(is_mix, audio + audio_rolled, audio)   # (B, 1, T)

    # apply the gain to BOTH the conditioning latent and the target
    latent_g = a * latent
    target_g = torch.clamp(a * target, -1.0, 1.0)          # gain clipping (max |waveform| ≤ 1)

    with torch.set_grad_enabled(model.training):
        recon = model.decode(latent_g, original_length=T)
        return loss_fn(recon, target_g, use_phase=use_phase)


# =============================================================================
#  Universal positive-weight interpolation (constant-power cone) loss
# =============================================================================

def universal_mix_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    audio: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    *,
    gain_min: float = 0.0,
    gain_max: float = 3.0,
    gain_prob: float = 0.8,
    mix_fraction: float = 0.5,
    use_phase: bool = False,
    progress: float | None = None,
    anneal_gain: bool = True,
    sampler: str = "polar",
) -> torch.Tensor:
    """Universal positive-weight interpolation ("interpolant cone") loss.

    Supervises the decoder on the **entire positive-weight mixing cone** via
    the polar parametrisation of every positive weight pair
    ``(w1, w2) = A·(cosθ, sinθ)`` with ``A = sqrt(w1²+w2²)`` the energy scale
    and ``θ ∈ [0, π/2]`` the mixing angle:

        D(A·cosθ·Z_A + A·sinθ·Z_B)  ≈  A·cosθ·X_A + A·sinθ·X_B

    where ``(X_A, X_B)`` is an in-batch pair (circular shift by one position,
    as in ``torres_implicit_linearity_loss``) and ``(Z_A, Z_B)`` its encoded
    latents.  Sampling ``θ ~ U(0, π/2)`` and ``A ~ U(gain_min, gain_max)``
    covers the whole cone:

      * ``A = 1``            → the trigonometric constant-power latent-mixup
        loss (the exact training manifold of the trig-mixup models);
      * ``θ = 0`` (or π/2)   → gain-scaled reconstruction ``A·X`` from ``A·Z``
        (the homogeneity objective of the implicit linearity loss);
      * any other ``(A, θ)`` → the exact operation of interest, so no
        interpolation law (lerp, log fader, parabolic, exponential, …) is
        out-of-distribution — the decoder is trained to interpolate *every*
        positive-weight path accurately instead of memorising one curve.

    A fraction ``1 - mix_fraction`` of the batch uses ``θ = 0`` (pure
    gain-scaled reconstruction), keeping the reconstruction objective explicit
    in the same batch.

    Gain sampling follows ``torres_implicit_linearity_loss``: ``A`` is drawn
    from ``U(gain_min, gain_max)`` with probability ``gain_prob`` (else
    ``A = 1``), ``|A| < 0.05`` is snapped to 0, targets are clipped to
    ``[-1, 1]``, and the range is optionally annealed to ``(1, 1)`` with a
    cosine ramp over ``progress``.

    Requires a scale-sensitive bottleneck (no final LayerNorm), exactly like
    the trigonometric mixup and implicit linearity losses.

    ``sampler`` selects the *distribution* over the positive-weight cone (the
    polar-vs-cartesian sampler ablation).  All three
    share the same target/decode/loss code path; only the joint law of the
    effective weights ``(w1, w2) = a·(cosθ, sinθ)`` differs:

      * ``"polar"`` (default) — ``a ~ U(gain_min, gain_max)``, w.p. ``gain_prob``
        else 1, and ``θ ~ U(0, π/2)`` (independent).  Density on the cone is
        *uniform in area* → isotropic in angle: no interpolation law is
        favoured.  This is the principled choice (curriculum on the radial
        scale is decoupled from the angular coverage, and both are natural).
      * ``"cart_square"`` — ``w_i ~ U(0, S)`` independently, with the *radius*
        ``S`` annealed via the same ``cosine_gain_range`` (so the
        energy→(1,1) curriculum is preserved).  This is the "natural default"
        a naive cartesian implementation would use; it is *anisotropic in θ*
        (over-samples θ → 0, π/2, under-samples θ ≈ π/4) and is expected to be
        worse than ``"polar"``.
      * ``"cart_iso"`` — ``w1, w2`` sampled so that the induced density on the
        cone matches ``"polar"`` exactly (uniform-in-area): draw
        ``a ~ U(gain_min, gain_max)``, ``θ ~ U(0, π/2)`` and set
        ``w_i = a·(cosθ, sinθ)``.  Numerically identical to ``"polar"``; it
        exists only as the honest control proving that *at equal distribution*
        the two parametrisations give the same backprop (§8.1).

    In every branch ``mix_fraction`` of the batch keeps the mixing branch and
    the rest uses ``θ = 0`` (pure gain-scaled reconstruction), exactly as the
    default, so the reconstruction objective stays explicit in the same batch.
    """
    B = audio.shape[0]
    T = audio.shape[-1]

    # ---- in-batch pairs (circular shift by one position, as in the paper) ----
    audio_rolled = torch.roll(audio, shifts=1, dims=0)
    z_rolled = torch.roll(z, shifts=1, dims=0)

    # ---- per-sample energy scale A (polar radius), same recipe as implicit ----
    if anneal_gain:
        # cosine ramp: full (gain_min, gain_max) cone at progress 0, (1,1) at 1.
        g_min, g_max = cosine_gain_range(gain_min, gain_max, progress)
    else:
        g_min, g_max = float(gain_min), float(gain_max)

    # Branch on the cone-density sampler (polar-vs-cartesian ablation).
    # Every branch produces
    # effective weights (w1, w2) = a·(cosθ, sinθ) and reuses the single
    # latent/target/decode/loss path below, so the only thing that differs
    # across branches is the joint law of (a, θ).
    if sampler == "polar":
        # Principled choice: A ~ U(gain_min, gain_max) (w.p. gain_prob else 1),
        # θ ~ U(0, π/2), independent.  Density uniform in area → isotropic in
        # angle; curriculum on the radius is decoupled from angular coverage.
        a = torch.ones(B, 1, 1, device=device)
        if g_max > g_min:
            use_gain = torch.rand(B, 1, 1, device=device) < gain_prob
            a = torch.where(
                use_gain,
                g_min + (g_max - g_min) * torch.rand(B, 1, 1, device=device),
                a,
            )
        a = torch.where(a.abs() < 0.05, torch.zeros_like(a), a)
        theta = (torch.pi / 2.0) * torch.rand(B, 1, 1, device=device)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
    elif sampler == "cart_square":
        # Naive cartesian default: w_i ~ U(0, S) independently, with the radius
        # S annealed via the same cosine_gain_range (preserves the energy→(1,1)
        # curriculum on the *support*, though the radius is max(|w|) here).
        # Anisotropic in θ: over-samples θ → 0, π/2 (square corners), under-
        # samples θ ≈ π/4.  Expected worse than "polar" (§8.2).
        S = g_max  # annealed upper bound; g_min is the floor (default 0).
        w1 = S * torch.rand(B, 1, 1, device=device)
        w2 = S * torch.rand(B, 1, 1, device=device)
        # reconstruct (a, cosθ, sinθ) from (w1, w2); guard the zero-pair.
        a = torch.sqrt(w1 * w1 + w2 * w2)
        a = torch.where(a.abs() < 0.05, torch.zeros_like(a), a)
        # cosθ = w1/a, sinθ = w2/a  (define 1/0 = 1 to keep θ=0 at the origin,
        # which is then silenced by a=0 anyway).
        inv_a = torch.where(a > 0, 1.0 / a, torch.ones_like(a))
        cos_t = w1 * inv_a
        sin_t = w2 * inv_a
    elif sampler == "cart_iso":
        # Honest control: sample (a, θ) exactly as "polar" and expose them as
        # cartesian weights w_i = a·(cosθ, sinθ).  Numerically identical to
        # "polar" — proves that at equal distribution the two parametrisations
        # give the same backprop (§8.1).  Kept as a separate branch so the run
        # is logged under its own name and the equality is visible in metrics.
        a = torch.ones(B, 1, 1, device=device)
        if g_max > g_min:
            use_gain = torch.rand(B, 1, 1, device=device) < gain_prob
            a = torch.where(
                use_gain,
                g_min + (g_max - g_min) * torch.rand(B, 1, 1, device=device),
                a,
            )
        a = torch.where(a.abs() < 0.05, torch.zeros_like(a), a)
        theta = (torch.pi / 2.0) * torch.rand(B, 1, 1, device=device)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
    else:
        raise ValueError(
            f"universal_mix_loss: unknown sampler '{sampler}' "
            f"(expected 'polar' | 'cart_square' | 'cart_iso')."
        )

    # ---- a fraction of the batch stays pure gain-scaled reconstruction (θ=0) ----
    is_mix = torch.rand(B, 1, 1, device=device) < mix_fraction
    # In the reconstruction branch we force θ=0 ⇒ (cosθ, sinθ)=(1, 0), i.e.
    # z' = a·z1 (gain-scaled single-source reconstruction), identical across
    # all three samplers — keeps the reconstruction objective explicit and
    # makes the branches differ ONLY on the mixing fraction of the batch.
    cos_t = torch.where(is_mix, cos_t, torch.ones_like(cos_t))
    sin_t = torch.where(is_mix, sin_t, torch.zeros_like(sin_t))

    # latent the decoder is conditioned on; target waveform (Eq. of the loss)
    latent = a * (cos_t * z + sin_t * z_rolled)
    target = torch.clamp(a * (cos_t * audio + sin_t * audio_rolled), -1.0, 1.0)

    with torch.set_grad_enabled(model.training):
        recon = model.decode(latent, original_length=T)
        return loss_fn(recon, target, use_phase=use_phase)


# =============================================================================
#  Encoder-side cone consistency (encoder homomorphism) loss
# =============================================================================

def encoder_cone_loss(
    model: nn.Module,
    audio: torch.Tensor,
    device: torch.device,
    *,
    gain_min: float = 0.0,
    gain_max: float = 3.0,
    gain_prob: float = 0.8,
    mix_fraction: float = 1.0,
    trig_only: bool = False,
    progress: float | None = None,
    anneal_gain: bool = True,
) -> torch.Tensor:
    """Encoder homomorphism (cone-consistency) loss — the encoder-side mirror
    of the decoder cone losses.

    Supervises

        E(w1·A + w2·B)  ≈  w1·E(A) + w2·E(B)

    on the **clean** (pre-noise) soft-norm codes, with the positive weights
    ``(w1, w2) = a·(cosθ, sinθ)`` sampled exactly as in
    ``universal_mix_loss`` (in-batch circular-shift pairs):

      * ``trig_only=True``  → constant-power trig arc, ``a ≡ 1``, whole batch
        (the training manifold of ``--use_latent_mixup``);
      * ``trig_only=False`` → the full cone ``a ~ U(gain_min, gain_max)``
        (w.p. ``gain_prob``, else 1) with θ ~ U(0, π/2) on a fraction
        ``mix_fraction`` of the batch and θ = 0 (gain-scaled homogeneity
        ``E(a·x) ≈ a·E(x)``) on the rest — mirroring ``universal_mix_loss``.

    The endpoint batch and the acoustic mix are encoded in ONE forward pass
    (``[audio; mix]``), so both code sides live in the same per-batch
    normalisation frame of the soft-norm bottleneck, and the loss is computed
    on the cached clean codes ``soft_norm.z_norm_clean`` (independent noise
    draws would otherwise floor the MSE).  Only the encoder is involved (no
    decode), complementing the decoder-side trig-mixup / universal losses so
    that the full positive-weight cone commutes with both E and D.

    Mean-reduction MSE matches the encoder-homomorphism metric of the
    evaluation battery (``latent_distance_toy.py``), so this loss directly
    optimises the reported quantity.
    """
    B = audio.shape[0]

    if model.soft_norm is None:
        raise ValueError(
            "encoder_cone_loss requires the soft-norm bottleneck (use_soft_norm) "
            "so that clean pre-noise codes are cached in soft_norm.z_norm_clean."
        )

    audio_rolled = torch.roll(audio, shifts=1, dims=0)          # (B, 1, T)

    # ---- per-sample weights (same samplers as the decoder-side losses) ----
    if trig_only:
        # constant-power trig arc over the whole batch, a ≡ 1
        a = torch.ones(B, 1, 1, device=device)
        theta = (torch.pi / 2.0) * torch.rand(B, 1, 1, device=device)
    else:
        if anneal_gain:
            g_min, g_max = cosine_gain_range(gain_min, gain_max, progress)
        else:
            g_min, g_max = float(gain_min), float(gain_max)
        a = torch.ones(B, 1, 1, device=device)
        if g_max > g_min:
            use_gain = torch.rand(B, 1, 1, device=device) < gain_prob
            a = torch.where(
                use_gain,
                g_min + (g_max - g_min) * torch.rand(B, 1, 1, device=device),
                a,
            )
        # paper convention: gains with |a| < 0.05 are set to 0 (silence)
        a = torch.where(a.abs() < 0.05, torch.zeros_like(a), a)
        theta = (torch.pi / 2.0) * torch.rand(B, 1, 1, device=device)
        # a fraction of the batch stays pure gain-scaled reconstruction (θ=0)
        is_mix = torch.rand(B, 1, 1, device=device) < mix_fraction
        theta = torch.where(is_mix, theta, torch.zeros_like(theta))

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # acoustic mix whose code must equal the weighted combo of the codes
    mix_audio = a * (cos_t * audio + sin_t * audio_rolled)      # (B, 1, T)

    # ---- one encode over [audio; mix] → identical normalisation frame ----
    with torch.set_grad_enabled(model.training):
        _ = model.encode(torch.cat([audio, mix_audio], dim=0))  # (2B, L, D)
    z_clean = model.soft_norm.z_norm_clean                      # (2B, L, D)
    z_a = z_clean[:B]                                            # codes of endpoints
    z_mix = z_clean[B:]                                          # codes of the mixes
    z_lin = a * (cos_t * z_a + sin_t * torch.roll(z_a, shifts=1, dims=0))
    return F.mse_loss(z_mix, z_lin)   # mean over elements, same as the eval metric


# =============================================================================
#  Quick self-test  (run this file directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SAME Loss — Self-Test")
    print("=" * 60)

    B, T = 2, 44100  # 1 second @ 44.1 kHz
    t = torch.linspace(0, 1.0, T).unsqueeze(0)

    # Target: 400 Hz sine
    freq = 400.0
    target = torch.sin(2 * torch.pi * freq * t)
    target = target.unsqueeze(1).repeat(B, 1, 1)  # (B, 1, T)

    # Prediction: same sine, slight amplitude and phase perturbation
    pred = 0.95 * torch.sin(2 * torch.pi * freq * t + 0.2)
    pred = pred.unsqueeze(1).repeat(B, 1, 1)

    loss_fn = SAMELoss()

    # ---- Magnitude only ----
    loss_mag = loss_fn(pred, target, use_phase=False)
    print(f"\nMagnitude-only loss (use_phase=False):  {loss_mag.item():.6f}")

    loss_mag_id = loss_fn(target, target, use_phase=False)
    print(f"Magnitude-only loss (identical):        {loss_mag_id.item():.10f}")

    # ---- Magnitude + Phase ----
    loss_phase = loss_fn(pred, target, use_phase=True)
    print(f"\nMag + Phase loss (use_phase=True):      {loss_phase.item():.6f}")

    loss_phase_id = loss_fn(target, target, use_phase=True)
    print(f"Mag + Phase loss (identical):            {loss_phase_id.item():.10f}")

    # ---- Verify phase term activates correctly ----
    assert loss_phase > loss_mag, \
        "Phase-aware loss should be >= magnitude-only loss"
    assert loss_mag_id < 0.01, \
        f"Magnitude-only loss for identical signals should be ~0, got {loss_mag_id}"
    assert loss_phase_id < 0.01, \
        f"Phase-aware loss for identical signals should be ~0, got {loss_phase_id}"

    print("\n✓  All assertions passed!")

