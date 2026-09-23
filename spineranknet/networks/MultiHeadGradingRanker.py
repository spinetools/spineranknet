"""Unified multi-task grading + ranking model.

:class:`MultiHeadGradingRanker` extends :class:`MultiHeadClassifier` with
per-task ranking heads that produce scalar severity scores. It is the
**single model class** used across all encoder-based training scripts
(classification-only, hybrid ranking, ranking finetune).

When ``RANKING_LOSS_WEIGHT=0`` the ranking heads are present but their
outputs are ignored during loss computation — this recovers pure
classification behaviour without changing the model class.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from spineranknet.networks.MultiHeadClassifier import MultiHeadClassifier
from spineranknet.networks.heads.ranking_heads import (
    ConcatLogitRankingHead,
    RankingHead,
    build_logit_ranking_head,
)
from spineranknet.networks.architectures.Resnet3dEncoders import (
    ENCODER_REGISTRY,
    get_encoder_factory,
    resnet18_encoder,
)


__all__ = ["MultiHeadGradingRanker", "build_grading_ranker"]


class MultiHeadGradingRanker(MultiHeadClassifier):
    """Encoder + classification heads + per-task ranking heads.

    Inherits all classification functionality from
    :class:`MultiHeadClassifier` (encoder, per-task classification heads,
    task metadata dicts) and adds per-task ranking heads that produce
    scalar severity scores.

    Naming convention
    -----------------
    Classification heads use ``clf_*`` keys (e.g. ``clf_pf``), ranking
    heads use ``rank_*`` keys (e.g. ``rank_pf``).  Both are derived from
    the ``model_head`` / ``ranking_head`` fields in
    :data:`spineranknet.config.TASK_DEFINITIONS`.

    Parameters
    ----------
    encoder : nn.Module
        3D ResNet encoder with ``out_dim`` attribute.
    tasks : list[str]
        Task names (ordinal names, e.g. ``"Pfirrmann"``, ``"Spondylolisthesis"``).
    task_defs : dict
        Task definitions from :data:`spineranknet.config.TASK_DEFINITIONS`.
    task_loss_override : dict, optional
        Per-task loss override (e.g. ``{"_all_": "corn"}``).
    head_type : str
        Classification head type (``"linear"``, ``"mlp"``, ``"transformer"``, ``"kan"``).
    hidden_dim : int
        Hidden dimension for classification heads.
    ranking_head_type : str
        Ranking head architecture (``"linear"``, ``"mlp"``, ``"transformer"``,
        ``"least_squares"``, ``"kan"``).
    rank_hidden_dim : int
        Hidden dimension for ranking heads.
    rank_dropout : float
        Dropout in ranking heads.
    bounded : bool
        If True, ranking scores are sigmoid-scaled to ``[0, K-1]``.
    rank_from_logits : bool
        If True, ranking scores are derived from classification logits.
        The ``ranking_head_type`` parameter controls the architecture:
        ``"linear"`` → :class:`LogitRankingHead`,
        ``"mlp"`` → :class:`MLPLogitRankingHead`,
        ``"least_squares"`` → :class:`GELULogitRankingHead`,
        ``"kan"`` → :class:`KANLogitRankingHead`,
        ``"expectation"`` → :class:`ExpectationRankingHead`.
    rank_from_concat_logits : bool
        If True, ALL task logits are concatenated into a single vector
        ``(B, sum_K)`` and processed by a shared :class:`ConcatLogitRankingHead`
        that models inter-task relationships.  Mutually exclusive with
        ``rank_from_logits`` (concat mode takes precedence).
    cross_task_mode : str
        Cross-task attention mode applied after per-task ranking heads.
        ``"none"`` disables (default).  Other options: ``"transformer"``,
        ``"mlp"``, ``"gated"``, ``"mean_pool"``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tasks: List[str],
        task_defs: Dict[str, Dict[str, Any]],
        task_loss_override: Optional[Dict[str, str]] = None,
        head_type: str = "linear",
        hidden_dim: int = 256,
        ranking_head_type: str = "linear",
        rank_hidden_dim: int = 128,
        rank_dropout: float = 0.3,
        bounded: bool = True,
        bounding_mode: str = "softplus",
        rank_from_logits: bool = False,
        rank_from_concat_logits: bool = False,
        cross_task_mode: str = "none",
        cross_task_d_model: int = 64,
        cross_task_nhead: int = 4,
        cross_task_num_layers: int = 2,
        cross_task_dim_ff: int = 128,
        cross_task_dropout: float = 0.1,
        cross_task_store_attn: bool = False,
    ) -> None:
        super().__init__(
            encoder, tasks, task_defs,
            task_loss_override=task_loss_override,
            head_type=head_type, hidden_dim=hidden_dim,
        )
        self.rank_from_logits = rank_from_logits
        self.rank_from_concat_logits = rank_from_concat_logits
        self.cross_task_mode = cross_task_mode
        enc_dim = encoder.out_dim

        # Build per-task ranking heads (skip classification-only tasks).
        # Keys use abbreviated rank_* names from TASK_DEFINITIONS["ranking_head"].
        self.ranking_heads = nn.ModuleDict()
        self.ranking_tasks = []       # task names that have ranking heads
        self.rank_head_names = []     # abbreviated rank_* head names (same order)
        self._task_to_rank_head = {}  # task_name → rank_* key (for forward)
        self._rank_head_to_task = {}  # rank_* key → task_name  (for inverse lookup)
        for task in self.tasks:
            td = task_defs.get(task, {})
            if not td.get("ranking", True):
                continue  # e.g. IVDlevel — classification only

            # Determine abbreviated head name
            rank_key = td.get("ranking_head", task)  # fallback to full name for compat
            self.ranking_tasks.append(task)
            self.rank_head_names.append(rank_key)
            self._task_to_rank_head[task] = rank_key
            self._rank_head_to_task[rank_key] = task

            nc = self.task_num_classes[task]
            if rank_from_concat_logits:
                # Concat mode — heads built below as a single ConcatLogitRankingHead
                pass
            elif rank_from_logits:
                out_dim = self.task_out_dims[task]
                self.ranking_heads[rank_key] = build_logit_ranking_head(
                    head_type=ranking_head_type,
                    num_classes=out_dim,
                    bounded=bounded,
                    bounding_mode=bounding_mode,
                    hidden_dim=rank_hidden_dim,
                    dropout=rank_dropout,
                )
            else:
                self.ranking_heads[rank_key] = RankingHead(
                    enc_dim, nc,
                    head_type=ranking_head_type,
                    hidden_dim=rank_hidden_dim,
                    dropout=rank_dropout,
                    bounded=bounded,
                    bounding_mode=bounding_mode,
                )

        # Concatenated-logit ranking head (inter-task modelling)
        self._concat_ranking_head = None
        if rank_from_concat_logits and self.ranking_tasks:
            task_dims = {}
            for task in self.ranking_tasks:
                task_dims[task] = self.task_out_dims[task]
            self._concat_ranking_head = ConcatLogitRankingHead(
                task_dims=task_dims,
                head_type=ranking_head_type,
                hidden_dim=rank_hidden_dim,
                dropout=rank_dropout,
                bounded=bounded,
                bounding_mode=bounding_mode,
            )

        # Optional cross-task attention refinement
        if cross_task_mode != "none" and len(self.ranking_tasks) > 1:
            from spineranknet.networks.heads.cross_task_attention import CrossTaskAttention
            self.cross_task_attn = CrossTaskAttention(
                num_tasks=len(self.ranking_tasks),
                d_model=cross_task_d_model,
                nhead=cross_task_nhead,
                num_layers=cross_task_num_layers,
                dim_ff=cross_task_dim_ff,
                dropout=cross_task_dropout,
                mode=cross_task_mode,
                store_attn=cross_task_store_attn,
            )

    def forward(
        self, x: Tensor
    ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor], Tensor]:
        """Forward pass returning classification, ranking, and features.

        Parameters
        ----------
        x : Tensor, shape (B, C, D, H, W)
            Input 3D volume batch.

        Returns
        -------
        clf_out : tuple of Tensor
            Per-task classification logits, ordered by
            ``self.head_names``. Each entry has shape ``(B, out_dim)``.
        rank_out : dict of {str: Tensor}
            Per-task scalar severity scores keyed by **task name**
            (e.g. ``{"Pfirrmann": (B,), ...}``). Task names (not
            abbreviated ``rank_*`` keys) are used so the training loop
            can match scores to labels without extra mapping.
        features : Tensor, shape (B, enc_dim)
            Encoder features.
        """
        features = self.encoder(x)
        clf_out = tuple(self.heads[h](features) for h in self.head_names)

        if self.rank_from_concat_logits and self._concat_ranking_head is not None:
            # Concatenate all ranking-task logits → shared head
            logit_parts = []
            for task in self._concat_ranking_head.task_names:
                idx = self.tasks.index(task)
                logit_parts.append(clf_out[idx])
            concat_logits = torch.cat(logit_parts, dim=1)  # (B, sum_K)
            rank_out = self._concat_ranking_head(concat_logits)
        elif self.rank_from_logits:
            rank_out = {}
            for i, task in enumerate(self.tasks):
                rank_key = self._task_to_rank_head.get(task)
                if rank_key is None:
                    continue  # classification-only task
                logits = clf_out[i]
                rank_out[task] = self.ranking_heads[rank_key](logits)
        else:
            rank_out = {
                task: self.ranking_heads[self._task_to_rank_head[task]](features)
                for task in self.ranking_tasks
            }

        # Cross-task attention refinement
        if self.cross_task_mode != "none" and hasattr(self, "cross_task_attn"):
            task_order = list(self.ranking_tasks)
            draft_scores = torch.stack(
                [rank_out[t] for t in task_order], dim=1,
            )  # (B, T)
            refined_scores = self.cross_task_attn(draft_scores)  # (B, T)
            rank_out = {
                t: refined_scores[:, i] for i, t in enumerate(task_order)
            }

        return clf_out, rank_out, features


# ════════════════════════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════════════════════════

# Backbone factories are resolved through the shared encoder registry. The core
# package registers ResNet-18; importing ``spineranknet.experiments`` adds the
# deeper backbones (ResNet-34/50/101/152) used in the backbone-ablation study.
_ENCODER_MAP = ENCODER_REGISTRY


def build_grading_ranker(
    arch: str,
    tasks: List[str],
    task_defs: Dict[str, Dict[str, Any]],
    task_loss_override: Optional[Dict[str, str]] = None,
    head_type: str = "linear",
    hidden_dim: int = 256,
    ranking_head_type: str = "linear",
    rank_hidden_dim: int = 128,
    rank_dropout: float = 0.3,
    bounded: bool = True,
    bounding_mode: str = "softplus",
    rank_from_logits: bool = False,
    rank_from_concat_logits: bool = False,
    cross_task_mode: str = "none",
    cross_task_d_model: int = 64,
    cross_task_nhead: int = 4,
    cross_task_num_layers: int = 2,
    cross_task_dim_ff: int = 128,
    cross_task_dropout: float = 0.1,
    cross_task_store_attn: bool = False,
    **enc_kwargs,
) -> MultiHeadGradingRanker:
    """Build a :class:`MultiHeadGradingRanker` with the specified backbone.

    Parameters
    ----------
    arch : str
        Backbone architecture (``resnet18``, ``resnet34``, ``resnet50``, etc.).
    tasks : list[str]
        Task names.
    task_defs : dict
        Task definitions from :data:`spineranknet.config.TASK_DEFINITIONS`.
    rank_from_concat_logits : bool
        If True, ALL task logits are concatenated and fed through a shared
        :class:`ConcatLogitRankingHead` that models inter-task relationships.
    cross_task_mode : str
        Cross-task attention mode (``"none"``, ``"transformer"``, ``"mlp"``,
        ``"gated"``, ``"mean_pool"``).
    **enc_kwargs
        Extra keyword arguments forwarded to the encoder factory
        (e.g. ``in_channels``).

    Returns
    -------
    MultiHeadGradingRanker
    """
    if arch not in _ENCODER_MAP:
        available = ", ".join(_ENCODER_MAP.keys())
        raise ValueError(f"Unknown architecture: {arch}. Available: {available}")

    print(f"\n{'=' * 80}")
    print(f"Building MultiHeadGradingRanker: {arch}")
    if rank_from_concat_logits:
        print(f"  Concat-logit ranking: head_type={ranking_head_type}")
    elif rank_from_logits:
        print(f"  Per-task logit ranking: head_type={ranking_head_type}")
    else:
        print(f"  Feature-based ranking: head_type={ranking_head_type}")
    if cross_task_mode != "none":
        print(f"  Cross-task attention: {cross_task_mode} "
              f"(d={cross_task_d_model}, heads={cross_task_nhead}, "
              f"layers={cross_task_num_layers})")
    print(f"{'=' * 80}")

    encoder = _ENCODER_MAP[arch](**enc_kwargs)

    return MultiHeadGradingRanker(
        encoder, tasks, task_defs,
        task_loss_override=task_loss_override,
        head_type=head_type,
        hidden_dim=hidden_dim,
        ranking_head_type=ranking_head_type,
        rank_hidden_dim=rank_hidden_dim,
        rank_dropout=rank_dropout,
        bounded=bounded,
        bounding_mode=bounding_mode,
        rank_from_logits=rank_from_logits,
        rank_from_concat_logits=rank_from_concat_logits,
        cross_task_mode=cross_task_mode,
        cross_task_d_model=cross_task_d_model,
        cross_task_nhead=cross_task_nhead,
        cross_task_num_layers=cross_task_num_layers,
        cross_task_dim_ff=cross_task_dim_ff,
        cross_task_dropout=cross_task_dropout,
        cross_task_store_attn=cross_task_store_attn,
    )
