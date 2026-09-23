"""Deeper 3D ResNet encoders for the backbone-ablation study.

The published SpineRankNet model uses the ResNet-18 backbone that ships in
:mod:`spineranknet.networks.architectures.Resnet3dEncoders`. The deeper
backbones below (ResNet-34/50/101/152) are only used in the paper's
backbone-ablation experiments, so they live here in
:mod:`spineranknet.experiments` rather than the core package.

Importing this module (or the :mod:`spineranknet.experiments` package) registers
these factories into the shared
:data:`spineranknet.networks.architectures.Resnet3dEncoders.ENCODER_REGISTRY`,
so the ablation-capable trainer can build them via ``--arch resnet50`` etc.
"""
from __future__ import annotations

from spineranknet.networks.architectures.Resnet3dEncoders import (
    BasicBlock,
    Bottleneck,
    ResNet3DEncoder,
    register_encoder,
)

__all__ = [
    "resnet34_encoder",
    "resnet50_encoder",
    "resnet101_encoder",
    "resnet152_encoder",
    "EXTRA_ENCODERS",
]


def resnet34_encoder(**kwargs: int) -> ResNet3DEncoder:
    """ResNet-34 3D asymmetric encoder (~21M params)."""
    return ResNet3DEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_encoder(**kwargs: int) -> ResNet3DEncoder:
    """ResNet-50 3D asymmetric encoder (~23M params)."""
    return ResNet3DEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_encoder(**kwargs: int) -> ResNet3DEncoder:
    """ResNet-101 3D asymmetric encoder (~42M params)."""
    return ResNet3DEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_encoder(**kwargs: int) -> ResNet3DEncoder:
    """ResNet-152 3D asymmetric encoder (~58M params)."""
    return ResNet3DEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)


#: Deeper backbones registered into the shared encoder registry on import.
EXTRA_ENCODERS = {
    "resnet34": resnet34_encoder,
    "resnet50": resnet50_encoder,
    "resnet101": resnet101_encoder,
    "resnet152": resnet152_encoder,
}

for _name, _factory in EXTRA_ENCODERS.items():
    register_encoder(_name, _factory)
