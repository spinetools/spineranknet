"""3-D asymmetric ResNet encoders for sagittal spinal MRI sub-volumes.

The encoders keep the slice (depth) dimension nearly intact while
aggressively downsampling the in-plane (height/width) resolution, which
suits thin sagittal slabs (e.g. ``12 × 128 × 256``). A single
:class:`ResNet3DEncoder` feeds the multi-task classification and ranking
heads in :class:`~spineranknet.networks.MultiHeadGradingRanker`.

The published default, ResNet-18, is exposed here as
:func:`resnet18_encoder` and registered in :data:`ENCODER_REGISTRY`.
The deeper backbones used only in the ablation study (ResNet-34/50/101/152)
live in :mod:`spineranknet.experiments.encoders` and register themselves
into the same shared registry on ``import spineranknet.experiments``.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Type, Union

import torch
from torch import Tensor
from torch import nn as nn


__all__ = [
    "conv3x3",
    "conv1x1",
    "BasicBlock",
    "Bottleneck",
    "ResNet3DEncoder",
    "resnet18_encoder",
    "ENCODER_REGISTRY",
    "register_encoder",
    "get_encoder_factory",
]


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv3d:
    """3×3×3 convolution with asymmetric (in-plane only) stride."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=(1, stride, stride),
        padding=(1, dilation, dilation),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1×1×1 convolution with asymmetric (in-plane only) stride."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 1, 1),
        stride=(1, stride, stride),
        padding=0,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Two-conv residual block (ResNet-18/34)."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm3d if norm_layer is None else norm_layer
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """Three-conv bottleneck residual block (ResNet-50+)."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm3d if norm_layer is None else norm_layer
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3DEncoder(nn.Module):
    """3-D asymmetric ResNet feature encoder.

    Parameters
    ----------
    block : type of BasicBlock or Bottleneck
        Residual block type.
    layers : list of int
        Number of blocks in each of the four stages.
    in_channels : int, optional
        Number of input channels (default ``1`` for single-modality MRI).

    Attributes
    ----------
    out_dim : int
        Dimensionality of the pooled feature vector
        (``512 * block.expansion``).
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self._norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d
        self.inplanes: int = 64
        self.dilation: int = 1
        self.groups: int = 1
        self.base_width: int = 64

        self.conv1 = nn.Conv3d(
            in_channels, self.inplanes, (3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_dim: int = 512 * block.expansion

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )
        layers: List[nn.Module] = [
            block(
                self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, 1, self._norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=self._norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ============================================================================
# FACTORY FUNCTIONS — ResNet-18 is the published default
# ============================================================================

def resnet18_encoder(**kwargs: int) -> ResNet3DEncoder:
    """ResNet-18 3D asymmetric encoder (~11M params) — the paper default."""
    return ResNet3DEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)


# ----------------------------------------------------------------------------
# Encoder registry
# ----------------------------------------------------------------------------
# Maps a backbone name to a zero-arg-friendly factory. The core package ships
# only the published ResNet-18 default; deeper backbones used in the paper's
# backbone-ablation study (ResNet-34/50/101/152) are registered by
# :mod:`spineranknet.experiments.encoders` when that subpackage is imported.

ENCODER_REGISTRY: dict[str, Callable[..., ResNet3DEncoder]] = {
    "resnet18": resnet18_encoder,
}


def register_encoder(
    name: str, factory: Callable[..., ResNet3DEncoder]
) -> None:
    """Register an encoder factory under ``name`` (used by experiments)."""
    ENCODER_REGISTRY[name] = factory


def get_encoder_factory(name: str) -> Callable[..., ResNet3DEncoder]:
    """Return the encoder factory registered under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is unknown. Deeper backbones require
        ``import spineranknet.experiments`` to register them first.
    """
    if name not in ENCODER_REGISTRY:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise KeyError(
            f"Unknown encoder {name!r}. Available: {available}. "
            f"(Deeper backbones are registered by spineranknet.experiments.)"
        )
    return ENCODER_REGISTRY[name]
