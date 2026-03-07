from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.short = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.short = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.short(x)
        return self.relu(out)


def make_stage(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(max(0, int(blocks) - 1)):
        layers.append(BasicResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class PIDNetLite(nn.Module):
    """Parameter-efficient PIDNet-style architecture (P/I/D branches)."""

    def __init__(
        self,
        num_classes: int = 19,
        stem_channels: int = 32,
        p_channels: int = 64,
        d_channels: int = 32,
        i_channels: Sequence[int] = (48, 80, 128),
        i_blocks: Sequence[int] = (1, 1, 1),
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(i_channels) != 3:
            raise ValueError("i_channels must have length 3.")
        if len(i_blocks) != 3:
            raise ValueError("i_blocks must have length 3.")

        i1c, i2c, i3c = [int(v) for v in i_channels]
        b1, b2, b3 = [int(v) for v in i_blocks]
        stem_channels = int(stem_channels)
        p_channels = int(p_channels)
        d_channels = int(d_channels)
        head_channels = int(head_channels)

        # Shared stem to 1/4 resolution.
        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_channels, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1),
        )

        # I-branch (semantic, low-resolution).
        self.i_stage1 = make_stage(stem_channels, i1c, blocks=b1, stride=2)  # 1/8
        self.i_stage2 = make_stage(i1c, i2c, blocks=b2, stride=2)             # 1/16
        self.i_stage3 = make_stage(i2c, i3c, blocks=b3, stride=2)             # 1/32
        self.i_context = ConvBNReLU(i3c, i3c, kernel_size=3, stride=1, padding=1)

        # P-branch (detail-preserving at 1/8).
        self.p_proj = ConvBNReLU(stem_channels, p_channels, kernel_size=1, stride=1, padding=0)
        self.p_down = ConvBNReLU(p_channels, p_channels, kernel_size=3, stride=2, padding=1)
        self.i1_to_p = ConvBNReLU(i1c, p_channels, kernel_size=1, stride=1, padding=0)
        self.i2_to_p = ConvBNReLU(i2c, p_channels, kernel_size=1, stride=1, padding=0)
        self.p_refine = BasicResidualBlock(p_channels, p_channels, stride=1)

        # D-branch (boundary-sensitive, guided by P and I features).
        self.d_from_p = ConvBNReLU(p_channels, d_channels, kernel_size=3, stride=1, padding=1)
        self.i1_to_d = ConvBNReLU(i1c, d_channels, kernel_size=1, stride=1, padding=0)
        self.d_refine = BasicResidualBlock(d_channels, d_channels, stride=1)

        # Final fusion at 1/8 then upsample to full resolution.
        self.i_to_fuse = ConvBNReLU(i3c, p_channels, kernel_size=1, stride=1, padding=0)
        fuse_in = p_channels + p_channels + d_channels
        self.fuse = nn.Sequential(
            ConvBNReLU(fuse_in, head_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(head_channels, head_channels, kernel_size=3, stride=1, padding=1),
        )
        self.classifier = nn.Conv2d(head_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        stem = self.stem(x)                    # 1/4

        i1 = self.i_stage1(stem)               # 1/8
        i2 = self.i_stage2(i1)                 # 1/16
        i3 = self.i_context(self.i_stage3(i2)) # 1/32

        p = self.p_down(self.p_proj(stem))     # 1/8
        p = p + self._up_to(self.i1_to_p(i1), p)
        p = p + self._up_to(self.i2_to_p(i2), p)
        p = self.p_refine(p)

        d = self.d_from_p(p) + self._up_to(self.i1_to_d(i1), p)
        d = self.d_refine(d)

        i_fuse = self._up_to(self.i_to_fuse(i3), p)
        fused = torch.cat([p, i_fuse, d], dim=1)
        fused = self.fuse(fused)
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_pidnet_from_cfg(cfg: Dict) -> PIDNetLite:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    pid_cfg = model_cfg.get("pidnet", {})
    return PIDNetLite(
        num_classes=int(data_cfg["num_classes"]),
        stem_channels=int(pid_cfg.get("stem_channels", 32)),
        p_channels=int(pid_cfg.get("p_channels", 64)),
        d_channels=int(pid_cfg.get("d_channels", 32)),
        i_channels=tuple(int(x) for x in pid_cfg.get("i_channels", [48, 80, 128])),
        i_blocks=tuple(int(x) for x in pid_cfg.get("i_blocks", [1, 1, 1])),
        head_channels=int(pid_cfg.get("head_channels", 128)),
    )
