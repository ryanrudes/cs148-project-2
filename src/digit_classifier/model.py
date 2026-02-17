"""ResNeXt model with stochastic depth (drop-path).

This is the only model architecture used by the training pipeline.  It is a
faithful port of the original ``resnext.py`` with no behavioural changes.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic Depth — drop an entire residual branch with probability *p*.

    During training each sample in the batch is independently dropped (replaced
    with identity).  At test time outputs are returned unchanged.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob
        return x * mask / keep_prob


def conv3x3(
    in_features: int,
    out_features: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3×3 convolution with padding."""
    return nn.Conv2d(
        in_features,
        out_features,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_features: int,
    out_features: int,
    stride: int = 1,
) -> nn.Conv2d:
    """1×1 convolution."""
    return nn.Conv2d(
        in_features,
        out_features,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


# ---------------------------------------------------------------------------
# Bottleneck block
# ---------------------------------------------------------------------------

class ResNeXtBlock(nn.Module):
    """Bottleneck block with grouped convolution and optional drop-path.

    The stride-2 down-sampling is placed at the 3×3 convolution (conv2),
    not at the first 1×1 convolution.
    """

    expansion: int = 4

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        width = int(out_features * (base_width / 64)) * groups

        self.conv1 = conv1x1(in_features, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, out_features * self.expansion)
        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(out_features * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        z = self.relu(self.bn1(self.conv1(x)))
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.bn3(self.conv3(z))

        if self.downsample is not None:
            identity = self.downsample(x)

        z = self.drop_path(z)
        z += identity
        return self.relu(z)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class ResNeXt(nn.Module):
    """ResNeXt with linearly increasing stochastic depth."""

    def __init__(
        self,
        layers: list[int] | tuple[int, ...],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        drop_path_rate: float = 0.0,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element list, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # Linearly increasing drop-path rates per block
        total_blocks = sum(layers)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self._block_idx = 0
        self._dpr = dpr

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtBlock.expansion, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNeXtBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    # ------------------------------------------------------------------

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * ResNeXtBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * ResNeXtBlock.expansion, stride),
                norm_layer(planes * ResNeXtBlock.expansion),
            )

        layer_blocks: list[nn.Module] = []
        layer_blocks.append(
            ResNeXtBlock(
                self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, previous_dilation,
                norm_layer, drop_path=self._dpr[self._block_idx],
            )
        )
        self._block_idx += 1
        self.inplanes = planes * ResNeXtBlock.expansion

        for _ in range(1, blocks):
            layer_blocks.append(
                ResNeXtBlock(
                    self.inplanes, planes,
                    groups=self.groups, base_width=self.base_width,
                    dilation=self.dilation, norm_layer=norm_layer,
                    drop_path=self._dpr[self._block_idx],
                )
            )
            self._block_idx += 1

        return nn.Sequential(*layer_blocks)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
