#!/usr/bin/env python3
"""
Shared metrics for the SAME-vs-LinCAE algebraic separation study.

Model-independent: both workers (venv for SAME, cae_env for Lin-CAE) import
this module so the exact same metric definitions are used on both sides.

  * ``si_sdr(ref, est)``      — scale-invariant SDR in dB (higher = better),
                                identical to toy_evaluation/mix_recon_ratio.py.
  * ``ms_stft_mse(pred, tgt)``— multi-scale STFT magnitude MSE (lower = better),
                                mean over the 7 SAME-paper FFT resolutions
                                [32,64,128,256,512,1024,2048], hop = n_fft//4,
                                Hann window, centre=True, reflect padding.

The two models operate at different sample rates (SAME: 16 kHz, Lin-CAE:
44.1 kHz); absolute metric values are therefore NOT directly comparable across
models.  Comparisons must be done on NORMALIZED quantities, i.e. separation
error relative to each model's own plain reconstruction error on the same
clips (see aggregate.py).
"""

from __future__ import annotations

import numpy as np
import torch

FFT_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
HOP_SIZES = [n // 4 for n in FFT_SIZES]


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019). Higher = better."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = np.dot(ref, ref)
    if denom < 1e-12:
        return float("nan")
    alpha = np.dot(est, ref) / denom
    e_target = alpha * ref
    e_noise = est - e_target
    snr = np.dot(e_target, e_target) / (np.dot(e_noise, e_noise) + 1e-12)
    return float(10.0 * np.log10(snr))


def ms_stft_mse(
    pred_np: np.ndarray,
    tgt_np: np.ndarray,
    device: torch.device,
) -> float:
    """Multi-scale STFT magnitude MSE between two mono waveforms (lower=better).

    Mean over the 7 FFT resolutions of the mean squared difference of the
    magnitude spectrograms (log-free, scale-free — identical for both models).
    """
    pred = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).to(device)
    tgt = torch.from_numpy(tgt_np.astype(np.float32)).unsqueeze(0).to(device)
    total = 0.0
    with torch.no_grad():
        for n_fft, hop in zip(FFT_SIZES, HOP_SIZES):
            win = torch.hann_window(n_fft, device=device)
            X = torch.stft(
                pred, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, return_complex=True, center=True, pad_mode="reflect",
            ).abs()
            Y = torch.stft(
                tgt, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, return_complex=True, center=True, pad_mode="reflect",
            ).abs()
            total += float(torch.mean((X - Y) ** 2))
    return total / len(FFT_SIZES)
