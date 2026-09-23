"""Ranking losses for ordinal severity grading (paper-core).

This module implements the **paper** ranking objectives used by the published
SpineRankNet model:

* :class:`SpineRankLoss` — the SpineRank pairwise loss
  (squared hinge with adaptive margin + same-grade similarity regulariser).
* :class:`TripletMarginOrdinalLoss` — ordinal triplet term used by the hybrid
  loss.
* :func:`cross_task_ranking_loss` — cross-pathology concordance regulariser.

Comparison/ablation losses (RankNet, DeepRankSVM, Relative Attributes, JND,
MSE/MAE regression, …) live in :mod:`spineranknet.experiments.comparison_losses`
together with the full :data:`RANKING_LOSSES` registry used by the
ablation-capable trainer.

Pairwise losses share the signature::

    forward(si, sj, y_ij, pair_type=None, severity_weight=None)

where ``si``/``sj`` are scores on ``[0, UNIVERSAL_SCALE]``, ``y_ij`` ∈ {−1,0,+1}
is the label direction, ``pair_type`` ∈ {0,1} (0 = same grade, 1 = ordinal),
and ``severity_weight`` is the normalised ordinal distance.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

__all__ = [
    "UNIVERSAL_SCALE",
    "TripletMarginOrdinalLoss",
    "SpineRankLoss",
    "cross_task_ranking_loss",
    "RANKING_LOSSES",
]

# ============================================================================
# Module-level constants
# ============================================================================

#: Upper bound of the universal score range used by every ranking head.
#: All per-task ordinal labels are mapped to ``[0, UNIVERSAL_SCALE]`` via
#: ``label * UNIVERSAL_SCALE / (K - 1)`` so that losses with explicit
#: margins behave consistently across pathologies with different numbers
#: of grades *K*.
UNIVERSAL_SCALE: float = 10.0

# ============================================================================
# Pairwise ranking losses (paper-core)
# ============================================================================

class TripletMarginOrdinalLoss(nn.Module):
    """Triplet margin loss adapted for ordinal ranking.

    For each anchor sample, selects a **positive** (same grade) and a
    hard **negative** (different grade, closest in score space) from
    the batch. The margin is optionally scaled by the ordinal distance
    between the anchor and the negative so that larger grade differences
    are penalised more strongly.

    Operates on **scalar ranking scores** (not embedding vectors),
    using absolute score difference as the distance metric.

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\frac{1}{N}
          \\sum_{a=1}^{N}
          \\max\\bigl(0,\\;
            d(a, p) - d(a, n^{*}) + m \\cdot \\Delta_g
          \\bigr)

    where :math:`d(a, x) = |s_a - s_x|` is the score distance,
    :math:`n^{*}` is the hardest negative (smallest :math:`d(a,n)`),
    :math:`m` is the base margin, and :math:`\\Delta_g = |g_a - g_{n^*}|`
    scales the margin by ordinal distance (if ``scale_by_distance``).

    This is a **pointwise** loss (``pointwise = True``): it receives
    ``(scores, labels)`` directly and the training loop skips pair
    generation.

    Reference
    ---------
    Adapted from the standard triplet loss:

    F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified
    Embedding and Verification Approach to Face Recognition," in
    *Proc. IEEE CVPR*, 2015, pp. 815--823.
    doi:`10.1109/CVPR.2015.7298682
    <https://doi.org/10.1109/CVPR.2015.7298682>`_

    Extended with ordinal-distance margin scaling for severity grading.

    Parameters
    ----------
    margin : float, optional
        Base triplet margin (default ``1.0``).
    scale_by_distance : bool, optional
        If ``True``, multiply margin by the ordinal distance between
        anchor and hard negative (default ``True``).
    scale_factor : float, optional
        Multiplier for ordinal distance when scaling. With universal
        ``[0, 10]`` scores, set to ``UNIVERSAL_SCALE / (K − 1)`` to
        normalise ordinal distances to the score range
        (default ``1.0``).
    **kw
        Ignored. Accepted for API uniformity.
    """

    pointwise: bool = True  # flag checked by train_epoch — skip pair generation

    def __init__(
        self,
        margin: float = 1.0,
        scale_by_distance: bool = True,
        scale_factor: float = 1.0,
        **kw,
    ) -> None:
        super().__init__()
        self.margin: float = float(margin)
        self.scale_by_distance: bool = bool(scale_by_distance)
        self.scale_factor: float = float(scale_factor)

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        """Compute ordinal triplet loss over a batch.

        Parameters
        ----------
        scores : Tensor, shape (N,)
            Scalar ranking scores for *N* valid samples.
        labels : Tensor, shape (N,)
            Integer ordinal labels for *N* valid samples.

        Returns
        -------
        Tensor
            Scalar loss (0 if no valid triplets can be formed).
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        n = scores.size(0)
        if n < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        losses = []
        for i in range(n):
            anchor_score = scores[i]
            anchor_label = labels[i]

            pos_mask = labels == anchor_label
            neg_mask = labels != anchor_label
            pos_mask[i] = False  # exclude self

            if not pos_mask.any() or not neg_mask.any():
                continue

            # Positive: closest score among same-grade samples.
            pos_scores = scores[pos_mask]
            d_pos = (anchor_score - pos_scores).abs()

            # Negative: hardest (closest score) among different-grade.
            neg_scores = scores[neg_mask]
            neg_labels = labels[neg_mask]
            d_neg = (anchor_score - neg_scores).abs()

            # Hardest negative = minimum distance in score space.
            hard_neg_idx = d_neg.argmin()
            d_neg_val = d_neg[hard_neg_idx]
            d_pos_val = d_pos.mean()

            m = self.margin
            if self.scale_by_distance:
                ordinal_dist = (
                    anchor_label - neg_labels[hard_neg_idx]
                ).abs().float()
                m = self.margin * (
                    ordinal_dist.clamp(min=1.0) * self.scale_factor
                )

            triplet_loss = torch.relu(d_pos_val - d_neg_val + m)
            losses.append(triplet_loss)

        if not losses:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        return torch.stack(losses).mean()



class SpineRankLoss(nn.Module):
    """SpineRank — Adaptive Hinge + Similarity Loss.

    A hybrid ranking loss designed for multi-task ordinal severity
    grading in spinal pathology. Combines two mechanisms into a
    single differentiable objective:

    1. **Squared hinge with adaptive margin** — the margin between a
       pair (i, j) scales with the ordinal distance ``|g_i − g_j|``,
       so that larger grade gaps require proportionally larger score
       separations. Uses squared hinge ``max(0, m − y·d)²`` for
       smooth, margin-aware gradients (non-zero gradient at the
       boundary, unlike standard hinge).
    2. **Severity weighting** — misranking a severe-vs-normal pair
       incurs a higher penalty than misranking adjacent grades. The
       weight is the raw severity distance on the universal scale.
    3. **Similarity regularizer** — an L2 penalty on score differences
       for same-grade pairs (``pair_type = 0``), preventing score
       collapse by pulling same-class scores together.

    Formulation (Eq. 2 in the paper)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\mathcal{L}_{\\text{hinge}}
                      + C_2\\, \\mathcal{L}_{\\text{sim}}

    **Adaptive hinge term** (ordinal pairs, :math:`y_{ij} \\neq 0`):

    .. math::

        \\mathcal{L}_{\\text{hinge}} = \\frac{1}{|\\mathcal{O}|}
          \\sum_{(i,j)\\in\\mathcal{O}} w_{ij}\\,
          \\bigl[\\max(0,\\; m_{ij} - y_{ij}\\,(s_i - s_j))\\bigr]^2

    where the per-pair adaptive margin is:

    .. math::

        m_{ij} = m_{\\text{base}}
          + m_{\\text{scale}} \\cdot \\frac{w_{ij}}{S}

    with :math:`S` = ``UNIVERSAL_SCALE`` (10) and :math:`w_{ij}` the
    severity weight.

    **Similarity regularizer** (similar pairs, :math:`y_{ij} = 0`):

    .. math::

        \\mathcal{L}_{\\text{sim}} = \\frac{1}{|\\mathcal{S}|}
          \\sum_{(i,j)\\in\\mathcal{S}} (s_i - s_j)^2

    This pulls scores of same-grade samples together, preventing
    arbitrary score spread within a grade.

    **Optional cross-task ranking** — when used with
    :func:`cross_task_ranking_loss` in the training loop, SpineRank
    additionally encourages consistent severity ordering across
    pathologies (e.g. Pfirrmann ↔ CCS).

    Reference
    ---------
    SpineRankNet. Draws on:

    * Squared hinge from RankSVM — T. Joachims, "Optimizing Search
      Engines using Clickthrough Data," *Proc. KDD*, 2002.
    * Adaptive margin from ordinal-distance scaling — F. Schroff et al.,
      "FaceNet," *Proc. CVPR*, 2015 (triplet margin scaling concept).
    * Similarity regularization from Relative Attributes — D. Parikh
      and K. Grauman, "Relative Attributes," *Proc. ICCV*, 2011.

    Parameters
    ----------
    margin : float, optional
        Unused — kept for API compatibility with other losses
        (default ``1.0``).
    margin_base : float, optional
        Minimum margin applied to every ordinal pair (default ``0.5``).
    margin_scale : float, optional
        Controls how much the margin grows with grade distance.
        Effective margin: ``margin_base + margin_scale × (w / S)``
        (default ``1.5``).
    C1 : float, optional
        Unused — present for API compatibility with DeepRankSVM
        (default ``1.0``).
    C2 : float, optional
        Weight of the similarity (same-grade L2) term (default ``0.25``).
    use_adaptive_margin : bool, optional
        Enable adaptive margin (default ``True``). When ``False``,
        uses a fixed margin ``margin_base`` for all pairs — disabling
        the ``margin_scale × (w / S)`` term.
    use_severity_weight : bool, optional
        Enable severity weighting on the hinge term (default ``True``).
        When ``False``, all ordinal pairs have equal weight.
    use_similarity : bool, optional
        Enable the similarity regulariser (default ``True``). When
        ``False``, ``C2`` is forced to 0.
    **kw
        Ignored. Accepted for API uniformity.
    """

    def __init__(
        self,
        margin: float = 1.0,
        margin_base: float = 0.5,
        margin_scale: float = 1.5,
        C1: float = 1.0,
        C2: float = 0.25,
        use_adaptive_margin: bool = True,
        use_severity_weight: bool = True,
        use_similarity: bool = True,
        **kw,
    ) -> None:
        super().__init__()
        self.margin_base: float = float(margin_base)
        self.margin_scale: float = (
            float(margin_scale) if use_adaptive_margin else 0.0
        )
        self.C2: float = float(C2) if use_similarity else 0.0
        self.use_severity_weight: bool = bool(use_severity_weight)
        # Store flags for logging / repr.
        self._flags: dict[str, bool] = dict(
            adaptive_margin=bool(use_adaptive_margin),
            severity_weight=bool(use_severity_weight),
            similarity=bool(use_similarity),
        )

    def extra_repr(self) -> str:
        parts = [
            f"margin_base={self.margin_base}",
            f"margin_scale={self.margin_scale}",
            f"C2={self.C2}",
        ]
        disabled = [k for k, v in self._flags.items() if not v]
        if disabled:
            parts.append(f"disabled=[{', '.join(disabled)}]")
        return ", ".join(parts)

    def forward(
        self,
        si: Tensor,
        sj: Tensor,
        y_ij: Tensor,
        pair_type: Optional[Tensor] = None,
        severity_weight: Optional[Tensor] = None,
        **kw,
    ) -> Tensor:
        """Compute SpineRank loss: ``L_hinge + C2·L_sim``.

        Parameters
        ----------
        si, sj : Tensor, shape (P,)
            Predicted ranking scores for each side of *P* pairs,
            on ``[0, UNIVERSAL_SCALE]``.
        y_ij : Tensor, shape (P,)
            Label direction ∈ {−1, 0, +1}.
        pair_type : Tensor, shape (P,), optional
            1 = ordinal pair (different grades), 0 = similarity pair
            (same grade). If ``None``, inferred from ``y_ij``:
            ordinal where ``y_ij ≠ 0``, similar where ``y_ij = 0``.
        severity_weight : Tensor, shape (P,), optional
            Per-pair ordinal distance on ``[0, UNIVERSAL_SCALE]``.
            Controls both the adaptive margin and the loss weight.
        **kw
            Ignored. Accepted to absorb stray kwargs from the training
            loop (e.g. ``scores_full``, ``labels_full``).

        Returns
        -------
        Tensor
            Scalar loss: ``L_hinge + C2 × L_sim``.
        """
        si, sj = si.view(-1), sj.view(-1)
        y_ij = y_ij.view(-1).float()
        d = si - sj  # signed score difference

        # Severity weight → adaptive margin.
        if severity_weight is not None:
            w = severity_weight.view(-1)
            w_norm = w / UNIVERSAL_SCALE  # → [0, 1]
        else:
            w = torch.ones_like(si)
            w_norm = torch.zeros_like(si)

        # Per-pair margin: adaptive when margin_scale > 0, fixed otherwise.
        m = self.margin_base + self.margin_scale * w_norm

        # Severity weight for loss terms (disabled → uniform weight 1.0).
        w_loss = w if self.use_severity_weight else torch.ones_like(w)

        # Pair-type masks.
        ord_mask = (y_ij != 0).float()
        if pair_type is not None:
            sim_mask = (pair_type.view(-1) == 0).float()
        else:
            sim_mask = (y_ij == 0).float()

        n_ord = ord_mask.sum().clamp(min=1)
        n_sim = sim_mask.sum().clamp(min=1)

        # Term 1: Squared hinge with (optionally adaptive) margin.
        L_hinge = (
            torch.relu(m - y_ij * d).pow(2) * w_loss * ord_mask
        ).sum() / n_ord

        # Term 2: Similarity regulariser (disabled when C2 == 0).
        L_sim = (d.pow(2) * sim_mask).sum() / n_sim

        return L_hinge + self.C2 * L_sim


# ============================================================================
# Cross-task ranking loss (used alongside any per-task loss)
# ============================================================================



def cross_task_ranking_loss(
    rank_out: dict[str, Tensor],
    labels: dict[str, Tensor],
    tasks: list[str],
    task_defs: dict[str, dict],
    max_pairs: int = 200,
    margin: float = 0.5,
) -> Tensor:
    """Cross-task concordance loss for multi-pathology severity ranking.

    Enforces that severity rankings are **consistent across
    pathologies** for the same sample pairs. For every random pair of
    samples (a, b) in the batch, looks at all tasks where both have
    valid labels. If the label direction agrees across two tasks
    (e.g. sample *a* is more severe in both Pfirrmann and CCS), the
    corresponding score differences should also agree in direction.
    Discordant cross-task score pairs receive a squared hinge penalty
    weighted by the label distance and the number of co-agreeing tasks.

    This encourages the model to learn clinically coherent severity
    profiles: severe disc degeneration should co-rank with stenosis,
    herniation with foraminal narrowing, etc.

    Formulation
    ~~~~~~~~~~~
    For each random sample pair (a, b) and each task *t* with valid
    labels for both samples:

    .. math::

        \\mathcal{L}_{\\text{cross}} = \\frac{1}{|\\mathcal{V}|}
          \\sum_{(a,b)} \\sum_{t \\in \\mathcal{T}_{ab}}
          (w_t + 1)(c_t + 1)\\,
          \\bigl[\\max(0,\\; m - y_t \\cdot d_t)\\bigr]^2

    where:

    * :math:`y_t = \\mathrm{sign}(g_a^t - g_b^t)` is the label
      direction for task *t*,
    * :math:`d_t = s_a^t - s_b^t` is the score difference,
    * :math:`w_t = |g_a^t - g_b^t| \\cdot S / (K_t - 1)` is the
      severity gap scaled to the universal range,
    * :math:`c_t` is the number of *other* tasks that share the same
      label direction (cross-task agreement count), and
    * :math:`\\mathcal{V}` is the set of valid (pair, task) entries.

    The key insight is the **cross-task agreement weighting**:
    a task-pair discordance is penalised more heavily when many other
    tasks agree on the direction — indicating strong clinical evidence
    that the scores should be concordant.

    This function is called from the training loop **after** per-task
    ranking losses. It adds no learnable parameters.

    Reference
    ---------
    Novel cross-task consistency objective introduced in SpineRankNet.
    Inspired by multi-task ranking consistency in:

    * Y. Zhang and Q. Yang, "A Survey on Multi-Task Learning in Deep
      Neural Networks," *arXiv:1706.05098*, 2017.
    * Clinical co-occurrence patterns in spinal pathology grading
      (Pfirrmann, Modic, CCS correlation).

    Parameters
    ----------
    rank_out : dict of {str: Tensor of shape (B,)}
        Per-task severity scores on ``[0, UNIVERSAL_SCALE]``.
    labels : dict of {str: Tensor of shape (B,)}
        Per-task ordinal labels (``-1`` or ``INVALID_LABEL`` for
        missing entries).
    tasks : list of str
        Task names to consider.
    task_defs : dict
        ``TASK_DEFINITIONS``-style mapping with ``num_classes`` per task.
    max_pairs : int, optional
        Maximum number of random sample pairs per batch
        (default ``200``).
    margin : float, optional
        Hinge margin for cross-task concordance (default ``0.5``).

    Returns
    -------
    Tensor
        Scalar loss (0 if fewer than 2 tasks or 2 samples).
    """
    device = next(iter(rank_out.values())).device
    B = next(iter(rank_out.values())).shape[0]
    if B < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Build (B, T) score and label matrices.
    valid_tasks = [t for t in tasks if t in rank_out and t in labels]
    T = len(valid_tasks)
    if T < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    score_mat = torch.zeros(B, T, device=device)
    label_mat = torch.full((B, T), -1, dtype=torch.long, device=device)
    scale_vec = torch.ones(T, device=device)

    for t_idx, task in enumerate(valid_tasks):
        mask = labels[task] >= 0
        score_mat[mask, t_idx] = rank_out[task][mask]
        label_mat[mask, t_idx] = labels[task][mask]
        nc = task_defs.get(task, {}).get("num_classes", 4)
        scale_vec[t_idx] = UNIVERSAL_SCALE / max(nc - 1, 1)

    # Vectorised random pair selection ---------------------------------
    # Enumerate the full upper-triangular index set (a, b) with a < b,
    # then sample without replacement. This replaces an O(P) Python
    # loop with O(1) tensor indexing.
    all_a, all_b = torch.triu_indices(B, B, offset=1, device=device)
    n_possible = all_a.shape[0]
    n_pairs = min(max_pairs, n_possible)

    if n_pairs < n_possible:
        sel = torch.randperm(n_possible, device=device)[:n_pairs]
        idx_a = all_a[sel]
        idx_b = all_b[sel]
    else:
        idx_a, idx_b = all_a, all_b

    # Score and label differences for all pairs: (n_pairs, T)
    score_diff = score_mat[idx_a] - score_mat[idx_b]                    # (P, T)
    label_diff = label_mat[idx_a].float() - label_mat[idx_b].float()    # (P, T)

    # Valid mask: both samples have labels for this task.
    valid_mask = (label_mat[idx_a] >= 0) & (label_mat[idx_b] >= 0)      # (P, T)

    # Label direction: +1 (a more severe), -1 (b more severe), 0 (same).
    label_dir = label_diff.sign()                                       # (P, T)

    # Ordinal mask: only pairs with different grades and valid labels.
    ord_mask = (label_dir != 0) & valid_mask                            # (P, T)

    # Cross-task agreement count: for each (pair, task), how many
    # other ordinal tasks have the same label direction?
    # (P, T, 1) == (P, 1, T) → (P, T, T) → sum over last dim − 1.
    agree_count = (
        (label_dir.unsqueeze(2) == label_dir.unsqueeze(1))
        & ord_mask.unsqueeze(2)
        & ord_mask.unsqueeze(1)
    ).float().sum(dim=2) - 1.0                                          # (P, T)
    agree_count = agree_count.clamp(min=0)

    # Weight: severity gap × cross-task agreement.
    sev_weight = label_diff.abs() * scale_vec.unsqueeze(0)              # (P, T)
    weight = (sev_weight + 1.0) * (agree_count + 1.0)                   # (P, T)

    # Squared hinge loss: penalise discordant (pair, task) entries.
    hinge = torch.relu(margin - label_dir * score_diff).pow(2)          # (P, T)

    # Mask to ordinal pairs with valid labels.
    masked_loss = hinge * weight * ord_mask.float()                     # (P, T)

    n_valid = ord_mask.float().sum().clamp(min=1)
    return masked_loss.sum() / n_valid


# ============================================================================
# Registry
# ============================================================================


# ============================================================================
# Registry (paper-core). The full ablation registry, including all comparison
# losses, lives in :mod:`spineranknet.experiments.comparison_losses`.
# ============================================================================

RANKING_LOSSES: dict[str, type[nn.Module]] = {
    "SpineRank":     SpineRankLoss,
    "TripletOrdinal": TripletMarginOrdinalLoss,
}
