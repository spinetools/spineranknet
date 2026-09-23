"""Multi-task classification model with per-task heads on a shared encoder.

:class:`MultiHeadClassifier` is the base classification model used across
SpineRankNet. It wraps a 3D ResNet encoder with one classification head
per ordinal task. Heads are sized according to the loss type declared
in :data:`spineranknet.config.TASK_DEFINITIONS` (``K`` outputs for
softmax-style losses; ``K − 1`` outputs for CORN, CORAL, and CLM).

The class is subclassed by :class:`MultiHeadGradingRanker` which adds
ranking heads on top.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn

from spineranknet.networks.architectures.Resnet3dEncoders import (
    ENCODER_REGISTRY,
    ResNet3DEncoder,
    resnet18_encoder,
)
from spineranknet.networks.heads.classification_heads import create_head


__all__ = ["MultiHeadClassifier", "build_multitask_resnet"]


#: Loss types that require ``K − 1`` head outputs rather than ``K``.
_K_MINUS_1_LOSSES: frozenset[str] = frozenset({"corn", "coral", "clm"})


class MultiHeadClassifier(nn.Module):
    """Encoder + per-task classification heads for multi-task ordinal grading.

    Parameters
    ----------
    encoder : ResNet3DEncoder
        3D ResNet feature extractor exposing an ``out_dim`` attribute
        (encoder feature dimensionality, e.g. 512 for ResNet-18).
    tasks : list of str
        Ordered list of task names (e.g.
        ``["Pfirrmann", "Narrowing", ...]``).
    task_defs : dict
        ``TASK_DEFINITIONS``-style mapping. Each entry must provide
        ``num_classes`` and optionally ``model_head`` (abbreviated head
        key, defaults to first 3 chars of the task name) and
        ``default_loss``.
    task_loss_override : dict, optional
        Maps task name → loss type, overriding ``default_loss``. The
        special key ``"_all_"`` applies to every task (default
        ``None``).
    head_type : str, optional
        Classification head architecture (``"linear"``, ``"mlp"``,
        ``"attention"``, ``"kan"``, ``"transformer"``); default
        ``"linear"``.
    hidden_dim : int, optional
        Hidden dimension for non-linear head types (default ``256``).

    Attributes
    ----------
    encoder : nn.Module
        The shared 3D ResNet backbone.
    tasks : list of str
        Task names in declaration order.
    head_names : list of str
        Abbreviated head keys, parallel to ``tasks``.
    heads : nn.ModuleDict
        Per-task classification heads keyed by abbreviated head name.
    task_num_classes : dict of {str: int}
        Original *K* (number of grades) per task.
    task_out_dims : dict of {str: int}
        Actual head output dimension (``K`` or ``K − 1`` depending on
        loss type).
    task_loss_types : dict of {str: str}
        Loss type used to size each head.
    """

    def __init__(
        self,
        encoder: ResNet3DEncoder,
        tasks: List[str],
        task_defs: Dict[str, Dict[str, Any]],
        task_loss_override: Optional[Dict[str, str]] = None,
        head_type: str = "linear",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder: ResNet3DEncoder = encoder
        self.tasks: List[str] = list(tasks)
        self.task_loss_override: Dict[str, str] = dict(task_loss_override or {})
        self.head_type: str = head_type

        self.head_names: List[str] = []
        self.heads: nn.ModuleDict = nn.ModuleDict()
        # Per-task metadata queryable after construction.
        self.task_num_classes: Dict[str, int] = {}   # original K from task_defs
        self.task_out_dims: Dict[str, int] = {}      # actual head output dim
        self.task_loss_types: Dict[str, str] = {}    # loss type used to size head

        print(f"\n{'=' * 80}")
        print(
            f"MultiHeadClassifier: Creating {len(self.tasks)} "
            f"{head_type.upper()} heads"
        )
        print(f"{'=' * 80}")

        for t in self.tasks:
            td = task_defs[t]
            head_name = td.get("model_head", t[:3])
            num_classes = int(td.get("num_classes", 2))
            loss_type = self.task_loss_override.get(
                t, td.get("default_loss", "ce")
            )

            if loss_type in _K_MINUS_1_LOSSES:
                out_dim = num_classes - 1
            else:
                out_dim = num_classes

            head = create_head(head_type, self.encoder.out_dim, out_dim, hidden_dim)
            self.heads[head_name] = head
            self.head_names.append(head_name)

            self.task_num_classes[t] = num_classes
            self.task_out_dims[t] = out_dim
            self.task_loss_types[t] = loss_type

            print(
                f"  {head_name:5s} ({t:25s}): "
                f"{self.encoder.out_dim:4d} → {out_dim:2d}, loss={loss_type}"
            )

        print(f"{'=' * 80}\n")

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass through encoder and per-task classification heads.

        Parameters
        ----------
        x : Tensor, shape (B, C, D, H, W)
            Input 3D volume batch.

        Returns
        -------
        tuple of Tensor
            Per-task logits in the order given by ``self.head_names``.
            Each entry has shape ``(B, out_dim)`` where ``out_dim`` is
            ``K`` or ``K − 1`` depending on the task loss type.
        """
        z = self.encoder(x)
        return tuple(self.heads[h](z) for h in self.head_names)

    def finetune(self, reset_weights: bool = True) -> None:
        """Freeze the encoder and prepare heads for fine-tuning.

        Parameters
        ----------
        reset_weights : bool, optional
            If ``True``, reinitialise head parameters via
            ``head.reset_parameters()`` when available (default
            ``True``).
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        for head in self.heads.values():
            for p in head.parameters():
                p.requires_grad = True
            if reset_weights and hasattr(head, "reset_parameters"):
                head.reset_parameters()


def build_multitask_resnet(
    arch: str,
    tasks: List[str],
    task_defs: Dict[str, Dict[str, Any]],
    task_loss_override: Optional[Dict[str, str]] = None,
    head_type: str = "linear",
    hidden_dim: int = 256,
    **enc_kwargs: Any,
) -> MultiHeadClassifier:
    """Build a :class:`MultiHeadClassifier` with the chosen ResNet backbone.

    Parameters
    ----------
    arch : str
        Backbone identifier. One of ``"resnet18"``, ``"resnet34"``,
        ``"resnet50"``, ``"resnet101"``, ``"resnet152"``.
    tasks : list of str
        Task names.
    task_defs : dict
        ``TASK_DEFINITIONS``-style mapping.
    task_loss_override : dict, optional
        Per-task loss overrides (default ``None``).
    head_type : str, optional
        Classification head type (``"linear"``, ``"mlp"``,
        ``"attention"``); default ``"linear"``.
    hidden_dim : int, optional
        Hidden dimension for non-linear heads (default ``256``).
    **enc_kwargs
        Additional encoder constructor arguments (e.g. ``in_channels``).

    Returns
    -------
    MultiHeadClassifier
        Assembled multi-task classifier.

    Raises
    ------
    ValueError
        If ``arch`` is not a known encoder factory.
    """
    # Resolved through the shared encoder registry: the core package registers
    # ResNet-18; importing ``spineranknet.experiments`` adds the deeper
    # backbones (ResNet-34/50/101/152) used in the backbone-ablation study.
    encoder_map = ENCODER_REGISTRY

    if arch not in encoder_map:
        available = ", ".join(encoder_map.keys())
        raise ValueError(
            f"Unknown architecture: {arch!r}. Available: {available}"
        )

    print(f"\n{'=' * 80}")
    print(f"Building 3D Asymmetric ResNet: {arch}")
    print(f"{'=' * 80}")

    enc = encoder_map[arch](**enc_kwargs)

    return MultiHeadClassifier(
        enc,
        tasks=tasks,
        task_defs=task_defs,
        task_loss_override=task_loss_override,
        head_type=head_type,
        hidden_dim=hidden_dim,
    )
