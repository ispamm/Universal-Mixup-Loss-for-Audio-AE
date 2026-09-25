"""
Custom Activation Functions
===========================

Provides non-standard activation functions for audio autoencoder experiments.

Classes
-------
- ``SnakeActivation`` : The periodic "snake" function from
  Liu et al. (2023), ``f(x) = x + (1/a) * sin²(a·x)``, with a learnable
  frequency parameter ``a``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SnakeActivation(nn.Module):
    """Snake activation function:  f(x) = x + (1/a) · sin²(a·x)

    Introduced by Liu et al. (2023) as a periodic alternative to ReLU / GELU.
    The learnable parameter ``a`` controls the frequency of the oscillation;
    larger ``a`` produces more zero-crossings per unit interval.  At a=1 the
    function is monotonic and behaves like a smooth, differentiable surrogate
    for LeakyReLU while preserving periodic inductive bias.

    Parameters
    ----------
    channels : int
        Number of input channels.  One learnable ``a`` per channel, broadcast
        over the time dimension.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(channels,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the snake activation.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, C, ...)`` (Conv) or ``(B, ..., C)``
                          (Transformer).  The channel dimension is inferred
                          by matching the size of ``self.a``.

        Returns
        -------
        torch.Tensor  same shape as ``x``.
        """
        # Find which dimension matches the number of channels
        channels = self.a.shape[0]
        # Build a shape for broadcasting: all 1s, except the channel dim
        shape = [1] * x.ndim
        # Locate the channel dimension by size
        channel_dim = None
        for d in range(x.ndim):
            if x.shape[d] == channels:
                channel_dim = d
                break
        if channel_dim is None:
            # Fallback: treat last dim as channels (Transformer: B, L, C)
            channel_dim = x.ndim - 1
        shape[channel_dim] = channels
        a_view = self.a.view(*shape)

        sin_sq = torch.sin(a_view * x) ** 2
        return x + (1.0 / (a_view + 1e-8)) * sin_sq
