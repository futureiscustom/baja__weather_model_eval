from __future__ import annotations

import torch
from torch import nn


class TinyWeatherNet(nn.Module):
    """Small residual CNN for next/future weather-state prediction.

    Input shape:  [batch, input_steps * channels, height, width]
    Output shape: [batch, channels, height, width]

    The network predicts a correction to the latest observed state. This gives
    persistence a natural starting point and makes the baseline easy to train.
    """

    def __init__(self, channels: int = 4, input_steps: int = 4, hidden: int = 48):
        super().__init__()
        self.channels = channels
        self.input_steps = input_steps

        in_channels = channels * input_steps
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got shape {tuple(x.shape)}")
        expected = self.channels * self.input_steps
        if x.shape[1] != expected:
            raise ValueError(f"Expected {expected} input channels, got {x.shape[1]}")

        latest = x[:, -self.channels :, :, :]
        return latest + self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
