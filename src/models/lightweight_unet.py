from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNAct, DecoderBlock, InvertedResidual


# The Unet architecture with MobileNetV2-style encoder
class MobileEncoder(nn.Module):
    """Compact MobileNetV2-style encoder built from scratch."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 56, 80, 112, 160),
        expand_ratio: int = 4,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("Expected exactly five encoder channel levels.")
        c1, c2, c3, c4, c5 = channels

        self.stem = ConvBNAct(3, c1, kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(c1, c2, blocks=2, stride=2, expand_ratio=expand_ratio, use_se=use_se)
        self.stage3 = self._make_stage(c2, c3, blocks=2, stride=2, expand_ratio=expand_ratio, use_se=use_se)
        self.stage4 = self._make_stage(c3, c4, blocks=3, stride=2, expand_ratio=expand_ratio, use_se=use_se)
        self.stage5 = self._make_stage(c4, c5, blocks=2, stride=2, expand_ratio=expand_ratio, use_se=use_se)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        expand_ratio: int,
        use_se: bool,
    ) -> nn.Sequential:
        layers = [
            InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_se=use_se,
            )
        ]
        for _ in range(blocks - 1):
            layers.append(
                InvertedResidual(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    expand_ratio=expand_ratio,
                    use_se=use_se,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        f1 = self.stem(x)    # 1/2
        f2 = self.stage2(f1)  # 1/4
        f3 = self.stage3(f2)  # 1/8
        f4 = self.stage4(f3)  # 1/16
        f5 = self.stage5(f4)  # 1/32
        return f1, f2, f3, f4, f5


# if using ResidualBlk
class ResidualBlock(nn.Module):
    """Basic residual block with optional downsample projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResidualEncoder(nn.Module):
    """Lightweight ResNet-style encoder producing 1/2..1/32 features."""

    def __init__(self, channels: Sequence[int], blocks_per_stage: Sequence[int] = (1, 1, 1, 1)) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("Expected exactly five encoder channel levels.")
        if len(blocks_per_stage) != 4:
            raise ValueError("Expected blocks_per_stage length of 4.")

        c1, c2, c3, c4, c5 = channels
        b2, b3, b4, b5 = [int(x) for x in blocks_per_stage]

        self.stem = ConvBNAct(3, c1, kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(c1, c2, blocks=b2, stride=2)
        self.stage3 = self._make_stage(c2, c3, blocks=b3, stride=2)
        self.stage4 = self._make_stage(c3, c4, blocks=b4, stride=2)
        self.stage5 = self._make_stage(c4, c5, blocks=b5, stride=2)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(max(0, blocks - 1)):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        f1 = self.stem(x)    # 1/2
        f2 = self.stage2(f1)  # 1/4
        f3 = self.stage3(f2)  # 1/8
        f4 = self.stage4(f3)  # 1/16
        f5 = self.stage5(f4)  # 1/32
        return f1, f2, f3, f4, f5


# if using FPNDecoder
class FPNDecoder(nn.Module):
    """Lightweight FPN-style decoder with top-down fusion."""

    def __init__(self, encoder_channels: Sequence[int], fpn_channels: int = 96) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = encoder_channels

        self.lateral2 = ConvBNAct(c2, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = ConvBNAct(c3, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral4 = ConvBNAct(c4, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral5 = ConvBNAct(c5, fpn_channels, kernel_size=1, stride=1, padding=0)

        # Smoothing heads after top-down addition.
        self.smooth2 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)
        self.smooth4 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)
        self.smooth5 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)

        # High-resolution refinement using the 1/2-scale stem feature.
        self.detail1 = ConvBNAct(c1, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.refine = nn.Sequential(
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1),
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1),
        )
        self.out_channels = fpn_channels

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        f1, f2, f3, f4, f5 = features

        p5 = self.lateral5(f5)
        p4 = self.lateral4(f4) + self._upsample_to(p5, f4)
        p3 = self.lateral3(f3) + self._upsample_to(p4, f3)
        p2 = self.lateral2(f2) + self._upsample_to(p3, f2)

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        # Aggregate all pyramid levels at 1/4 scale, then inject 1/2-scale detail.
        fused = p2 + self._upsample_to(p3, p2) + self._upsample_to(p4, p2) + self._upsample_to(p5, p2)
        fused = self._upsample_to(fused, f1) + self.detail1(f1)
        return self.refine(fused)


class LightweightUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 19,
        encoder_channels: Sequence[int] = (32, 56, 80, 112, 160),
        expand_ratio: int = 4,
        use_se: bool = False,
        encoder_type: str = "mobilenetv2",
        residual_blocks: Sequence[int] = (1, 1, 1, 1),
        decoder_type: str = "unet",
        fpn_channels: int = 96,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = encoder_channels

        encoder_type = encoder_type.lower().strip()
        self.encoder_type = encoder_type
        if encoder_type in {"mobilenetv2", "mobile", "mbconv"}:
            self.encoder = MobileEncoder(
                channels=encoder_channels,
                expand_ratio=expand_ratio,
                use_se=use_se,
            )
        elif encoder_type in {"resnet", "residual"}:
            self.encoder = ResidualEncoder(
                channels=encoder_channels,
                blocks_per_stage=tuple(int(x) for x in residual_blocks),
            )
        else:
            raise ValueError(
                f"Unsupported encoder_type '{encoder_type}'. Use 'mobilenetv2' or 'resnet'."
            )
        decoder_type = decoder_type.lower().strip()
        self.decoder_type = decoder_type

        if decoder_type == "unet":
            self.dec4 = DecoderBlock(in_channels=c5, skip_channels=c4, out_channels=c4)
            self.dec3 = DecoderBlock(in_channels=c4, skip_channels=c3, out_channels=c3)
            self.dec2 = DecoderBlock(in_channels=c3, skip_channels=c2, out_channels=c2)
            self.dec1 = DecoderBlock(in_channels=c2, skip_channels=c1, out_channels=c1)
            self.dec0 = DecoderBlock(in_channels=c1, skip_channels=0, out_channels=c1)
            decoder_out_channels = c1
        elif decoder_type == "fpn":
            self.fpn = FPNDecoder(encoder_channels=encoder_channels, fpn_channels=fpn_channels)
            decoder_out_channels = self.fpn.out_channels
        else:
            raise ValueError(f"Unsupported decoder_type '{decoder_type}'. Use 'unet' or 'fpn'.")

        self.classifier = nn.Conv2d(decoder_out_channels, num_classes, kernel_size=1)

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
        features = self.encoder(x)

        if self.decoder_type == "unet":
            f1, f2, f3, f4, f5 = features
            d = self.dec4(f5, f4)
            d = self.dec3(d, f3)
            d = self.dec2(d, f2)
            d = self.dec1(d, f1)
            d = self.dec0(d, None)
        else:
            d = self.fpn(features)

        logits = self.classifier(d)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_model(cfg: Dict) -> LightweightUNet:
    model_cfg = cfg["model"]
    model_type = str(model_cfg.get("model_type", "lightweight_unet")).lower()
    if model_type in {"bisenet", "bisenetv2", "bisenetv2_lite"}:
        from .bisenet import build_bisenet_from_cfg

        return build_bisenet_from_cfg(cfg)
    if model_type in {"pidnet", "pidnet_lite"}:
        from .pidnet import build_pidnet_from_cfg

        return build_pidnet_from_cfg(cfg)

    data_cfg = cfg["data"]
    channels: Iterable[int] = model_cfg["encoder_channels"]
    return LightweightUNet(
        num_classes=int(data_cfg["num_classes"]),
        encoder_channels=tuple(int(c) for c in channels),
        expand_ratio=int(model_cfg["expand_ratio"]),
        use_se=bool(model_cfg["use_se"]),
        encoder_type=str(model_cfg.get("encoder_type", "mobilenetv2")),
        residual_blocks=tuple(int(x) for x in model_cfg.get("residual_blocks", [1, 1, 1, 1])),
        decoder_type=str(model_cfg.get("decoder_type", "unet")),
        fpn_channels=int(model_cfg.get("fpn_channels", 96)),
    )
