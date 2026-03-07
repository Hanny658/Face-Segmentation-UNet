from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

bn_mom = 0.1
BatchNorm2d = nn.BatchNorm2d
algc = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=False,
    )


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        if self.no_relu:
            return out
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        outplanes = planes * self.expansion

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)

        self.conv3 = conv1x1(planes, outplanes)
        self.bn3 = BatchNorm2d(outplanes, momentum=bn_mom)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        if self.no_relu:
            return out
        return self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(inplanes, interplanes, stride=1),
            BatchNorm2d(interplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.f_x = nn.Sequential(
            conv1x1(in_channels, mid_channels),
            BatchNorm2d(mid_channels, momentum=bn_mom),
        )
        self.f_y = nn.Sequential(
            conv1x1(in_channels, mid_channels),
            BatchNorm2d(mid_channels, momentum=bn_mom),
        )
        self.proj = nn.Sequential(
            conv1x1(in_channels, in_channels),
            BatchNorm2d(in_channels, momentum=bn_mom),
        )

    def forward(self, x, y):
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=algc)

        fx = self.f_x(x)
        fy = self.f_y(y)
        sim = torch.sum(fx * fy, dim=1, keepdim=True)
        gate = torch.sigmoid(sim)

        out = x * (1 - gate) + y * gate
        out = self.proj(out)
        return out


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.refine = nn.Sequential(
            conv3x3(in_channels, out_channels),
            BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        mix = p * (1 - edge) + i * edge
        return self.refine(mix)


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.refine = nn.Sequential(
            conv1x1(in_channels, out_channels),
            BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        mix = p * (1 - edge) + i * edge
        return self.refine(mix)


class _PPMBranch(nn.Module):
    def __init__(self, in_ch, out_ch, pool: int):
        super().__init__()
        self.pool = pool
        self.conv = nn.Sequential(
            conv1x1(in_ch, out_ch),
            BatchNorm2d(out_ch, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        if self.pool == 1:
            y = x
        else:
            y = F.adaptive_avg_pool2d(x, output_size=(self.pool, self.pool))
        y = self.conv(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=algc)
        return y


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale0 = nn.Sequential(
            conv1x1(inplanes, branch_planes),
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.scale1 = _PPMBranch(inplanes, branch_planes, pool=1)
        self.scale2 = _PPMBranch(inplanes, branch_planes, pool=2)
        self.scale3 = _PPMBranch(inplanes, branch_planes, pool=4)
        self.scale4 = _PPMBranch(inplanes, branch_planes, pool=8)

        self.process = nn.Sequential(
            conv3x3(branch_planes * 5, outplanes),
            BatchNorm2d(outplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x0 = self.scale0(x)
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        x3 = self.scale3(x)
        x4 = self.scale4(x)
        out = torch.cat([x0, x1, x2, x3, x4], dim=1)
        return self.process(out)


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale0 = nn.Sequential(
            conv1x1(inplanes, branch_planes),
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.branches = nn.ModuleList(
            [
                _PPMBranch(inplanes, branch_planes, pool=1),
                _PPMBranch(inplanes, branch_planes, pool=2),
                _PPMBranch(inplanes, branch_planes, pool=4),
                _PPMBranch(inplanes, branch_planes, pool=8),
            ]
        )
        self.process = nn.Sequential(
            conv3x3(branch_planes * 5, outplanes),
            BatchNorm2d(outplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        xs = [self.scale0(x)]
        for b in self.branches:
            xs.append(b(x))
        out = torch.cat(xs, dim=1)
        return self.process(out)


class PIDNet(nn.Module):
    """Closer-to-paper PIDNet implementation adapted for this project pipeline."""

    def __init__(
        self,
        m=2,
        n=3,
        num_classes=19,
        planes=64,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    ):
        super().__init__()
        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.compression3 = nn.Sequential(
            conv1x1(planes * 4, planes * 2),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            conv1x1(planes * 8, planes * 2),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                conv3x3(planes * 4, planes),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                conv3x3(planes * 8, planes * 2),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                conv3x3(planes * 4, planes * 2),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                conv3x3(planes * 8, planes * 2),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        for mm in self.modules():
            if isinstance(mm, nn.Conv2d):
                nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mm, BatchNorm2d):
                nn.init.constant_(mm.weight, 1)
                nn.init.constant_(mm.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride=stride),
                BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = [block(inplanes, planes, stride=stride, downsample=downsample, no_relu=False)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, downsample=None, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride=stride),
                BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        return block(inplanes, planes, stride=stride, downsample=downsample, no_relu=True)

    def forward(self, x):
        input_size = x.shape[-2:]
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x), size=[height_output, width_output], mode="bilinear", align_corners=algc)
        if self.augment:
            temp_p = x_.clone()

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x), size=[height_output, width_output], mode="bilinear", align_corners=algc)
        if self.augment:
            temp_d = x_d.clone()

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)), size=[height_output, width_output], mode="bilinear", align_corners=algc)

        out = self.final_layer(self.dfm(x_, x, x_d))
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=algc)

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            x_extra_p = F.interpolate(x_extra_p, size=input_size, mode="bilinear", align_corners=algc)
            x_extra_d = F.interpolate(x_extra_d, size=input_size, mode="bilinear", align_corners=algc)
            return [x_extra_p, out, x_extra_d]
        return out


@dataclass
class SearchSpace:
    planes_list: Tuple[int, ...] = (12, 14, 16, 18, 20, 22, 24, 26, 28)
    mn_list: Tuple[Tuple[int, int], ...] = ((1, 2), (1, 3), (2, 2))
    ppm_list: Tuple[int, ...] = (24, 32, 40, 48, 56, 64, 72)
    head_list: Tuple[int, ...] = (24, 32, 40, 48, 56, 64, 80)
    augment: bool = False


def find_best_under_budget(
    target_params: int = 1_821_085,
    num_classes: int = 19,
    space: SearchSpace = SearchSpace(),
) -> Tuple[int, Tuple[int, int, int, int, int, bool]]:
    best_p = -1
    best_cfg: Optional[Tuple[int, int, int, int, int, bool]] = None
    for planes in space.planes_list:
        for (m, n) in space.mn_list:
            for ppm_planes in space.ppm_list:
                for head_planes in space.head_list:
                    model = PIDNet(
                        m=m,
                        n=n,
                        num_classes=num_classes,
                        planes=planes,
                        ppm_planes=ppm_planes,
                        head_planes=head_planes,
                        augment=space.augment,
                    )
                    p = count_params(model, trainable_only=True)
                    if p <= target_params and p > best_p:
                        best_p = p
                        best_cfg = (planes, m, n, ppm_planes, head_planes, space.augment)
    if best_cfg is None:
        raise RuntimeError("No config found under budget.")
    return best_p, best_cfg


def build_pidnet_from_cfg(cfg: Dict) -> PIDNet:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    pid_cfg = model_cfg.get("pidnet", {})
    return PIDNet(
        m=int(pid_cfg.get("m", 2)),
        n=int(pid_cfg.get("n", 3)),
        num_classes=int(data_cfg["num_classes"]),
        planes=int(pid_cfg.get("planes", 24)),
        ppm_planes=int(pid_cfg.get("ppm_planes", 48)),
        head_planes=int(pid_cfg.get("head_planes", 64)),
        augment=bool(pid_cfg.get("augment", True)),
    )
