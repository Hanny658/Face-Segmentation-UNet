from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention used inside MBConv blocks."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale
