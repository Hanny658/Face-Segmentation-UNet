from __future__ import annotations

from typing import Dict, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


# Tests conducted on a tiny version of BiSeNetV2 with 1/4 channels
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


class StemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid = max(out_channels // 2, 8)
        self.conv_in = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, mid, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(mid, out_channels, kernel_size=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        left = self.left(x)
        right = self.right(x)
        return self.fuse(torch.cat([left, right], dim=1))


class GELayerS1(nn.Module):
    def __init__(self, channels: int, expand_ratio: int = 6) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.block = nn.Sequential(
            ConvBNReLU(channels, hidden, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class GELayerS2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 6) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.conv1 = ConvBNReLU(in_channels, hidden, kernel_size=3, stride=1, padding=1)
        self.dw = ConvBNReLU(hidden, hidden, kernel_size=3, stride=2, padding=1, groups=hidden)
        self.pw = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.short = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw(self.dw(self.conv1(x)))
        short = self.short(x)
        return self.relu(out + short)


class CEBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBNReLU(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.conv_gap(self.gap(x))
        return self.conv(x + context)


class DetailBranch(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c1, c2, c3 = [int(c) for c in channels]
        self.s1 = nn.Sequential(
            ConvBNReLU(3, c1, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c1, c1, kernel_size=3, stride=1, padding=1),
        )
        self.s2 = nn.Sequential(
            ConvBNReLU(c1, c2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c2, c2, kernel_size=3, stride=1, padding=1),
        )
        self.s3 = nn.Sequential(
            ConvBNReLU(c2, c3, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c3, c3, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(c3, c3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s1(x)  # 1/2
        x = self.s2(x)  # 1/4
        x = self.s3(x)  # 1/8
        return x


class SemanticBranch(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c2, c3, c4, c5 = [int(c) for c in channels]
        self.stem = StemBlock(3, c2)  # 1/4
        self.s3 = nn.Sequential(
            GELayerS2(c2, c3),
            GELayerS1(c3),
        )  # 1/8
        self.s4 = nn.Sequential(
            GELayerS2(c3, c4),
            GELayerS1(c4),
        )  # 1/16
        self.s5 = nn.Sequential(
            GELayerS2(c4, c5),
            GELayerS1(c5),
            GELayerS1(c5),
        )  # 1/32
        self.ce = CEBlock(c5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.ce(x)
        return x


class BGALayer(nn.Module):
    def __init__(self, detail_channels: int, semantic_channels: int) -> None:
        super().__init__()
        self.detail_dw = nn.Sequential(
            ConvBNReLU(
                detail_channels,
                detail_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=detail_channels,
            ),
            nn.Conv2d(detail_channels, detail_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(detail_channels),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, detail_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(detail_channels),
        )
        self.semantic_refine = ConvBNReLU(detail_channels, detail_channels, kernel_size=3, stride=1, padding=1)
        self.out = ConvBNReLU(detail_channels, detail_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        semantic_up = F.interpolate(semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        semantic_proj = self.semantic_proj(semantic_up)
        sem_gate = torch.sigmoid(semantic_proj)
        detail_feat = self.detail_dw(detail)
        fused = detail_feat * sem_gate + self.semantic_refine(semantic_proj)
        return self.out(fused)


class SegHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int) -> None:
        super().__init__()
        self.block = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.drop(x)
        return self.classifier(x)


class BiSeNetV2Lite(nn.Module):
    def __init__(
        self,
        num_classes: int = 19,
        detail_channels: Sequence[int] = (24, 48, 96),
        semantic_channels: Sequence[int] = (16, 32, 64, 128),
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        self.detail = DetailBranch(detail_channels)
        self.semantic = SemanticBranch(semantic_channels)
        self.bga = BGALayer(detail_channels=int(detail_channels[-1]), semantic_channels=int(semantic_channels[-1]))
        self.head = SegHead(in_channels=int(detail_channels[-1]), mid_channels=int(head_channels), num_classes=num_classes)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        detail = self.detail(x)
        semantic = self.semantic(x)
        fused = self.bga(detail, semantic)
        logits = self.head(fused)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_bisenet_from_cfg(cfg: Dict) -> BiSeNetV2Lite:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    bisenet_cfg = model_cfg.get("bisenet", {})
    return BiSeNetV2Lite(
        num_classes=int(data_cfg["num_classes"]),
        detail_channels=tuple(int(x) for x in bisenet_cfg.get("detail_channels", [24, 48, 96])),
        semantic_channels=tuple(int(x) for x in bisenet_cfg.get("semantic_channels", [16, 32, 64, 128])),
        head_channels=int(bisenet_cfg.get("head_channels", 128)),
    )
