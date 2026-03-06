from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DecoderBlock, InvertedResidual


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


class LightweightUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 19,
        encoder_channels: Sequence[int] = (32, 56, 80, 112, 160),
        expand_ratio: int = 4,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = encoder_channels

        self.encoder = MobileEncoder(
            channels=encoder_channels,
            expand_ratio=expand_ratio,
            use_se=use_se,
        )

        self.dec4 = DecoderBlock(in_channels=c5, skip_channels=c4, out_channels=c4)
        self.dec3 = DecoderBlock(in_channels=c4, skip_channels=c3, out_channels=c3)
        self.dec2 = DecoderBlock(in_channels=c3, skip_channels=c2, out_channels=c2)
        self.dec1 = DecoderBlock(in_channels=c2, skip_channels=c1, out_channels=c1)
        self.dec0 = DecoderBlock(in_channels=c1, skip_channels=0, out_channels=c1)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

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
        f1, f2, f3, f4, f5 = self.encoder(x)

        x = self.dec4(f5, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        x = self.dec0(x, None)
        logits = self.classifier(x)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_model(cfg: Dict) -> LightweightUNet:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    channels: Iterable[int] = model_cfg["encoder_channels"]
    return LightweightUNet(
        num_classes=int(data_cfg["num_classes"]),
        encoder_channels=tuple(int(c) for c in channels),
        expand_ratio=int(model_cfg["expand_ratio"]),
        use_se=bool(model_cfg["use_se"]),
    )
