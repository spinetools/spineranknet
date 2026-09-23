"""Comparison / ablation ranking losses.

These losses are **not** part of the published SpineRankNet model; they are the
baselines and single-component ablations evaluated in the paper's ranking-loss
ablation tables. They are kept in :mod:`spineranknet.experiments` so the core
package advertises only the paper objective while remaining fully reproducible.

The module also owns the full :data:`RANKING_LOSSES` registry (CLI-name → class)
consumed by :mod:`spineranknet.baseline.train_ranking_mse` when an alternative
``--ranking_loss`` is requested. It re-exports the paper losses from
:mod:`spineranknet.losses.ranking_losses` so callers have a single source of
truth for the registry.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from spineranknet.losses.ranking_losses import (
    SpineRankLoss,
    TripletMarginOrdinalLoss,
)

__all__ = [
    "RelativeAttributesLoss",
    "DeepRankSVMLoss",
    "DeepRelativeAttributesLoss",
    "JustNoticeableDifferencesLoss",
    "RankNetLoss",
    "MSERegressionLoss",
    "MAERegressionLoss",
    "RANKING_LOSSES",
]

# ============================================================================
# Pairwise comparison losses
# ============================================================================

class RelativeAttributesLoss(nn.Module):
    """Relative Attributes — Ranking SVM loss with ordered + similar pairs.

    Learns a ranking function by enforcing two complementary constraints:

    * **Ordered pairs** (``pair_type = 1``, ``y_ij ∈ {-1, +1}``): the
      score difference must exceed a margin in the correct direction
      (squared hinge).
    * **Similar pairs** (``pair_type = 0``, ``y_ij = 0``): scores for
      same-grade samples should be close (squared L2 on the score
      difference).

    This faithfully implements the unconstrained Ranking SVM objective
    from Parikh & Grauman (2011), Eq. 2 (converted from constrained
    QP to a loss):

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\frac{1}{|\\mathcal{P}|}
          \\sum_{(i,j)\\in\\mathcal{P}}
          \\bigl[
            C_1 \\, p_{ij} \\, [\\max(0,\\; m - y_{ij} d_{ij})]^2
            + C_2 \\, (1 - p_{ij}) \\, d_{ij}^2
          \\bigr]

    where :math:`d_{ij} = s_i - s_j`, :math:`p_{ij}` is the pair type
    (1 = ordered, 0 = similar), :math:`m` is the hinge margin, and
    :math:`C_1, C_2` control the balance between ordering and similarity.

    Reference
    ---------
    D. Parikh and K. Grauman, "Relative Attributes," in *Proc. IEEE
    International Conference on Computer Vision (ICCV)*, 2011,
    pp. 503--510. doi:`10.1109/ICCV.2011.6126281
    <https://doi.org/10.1109/ICCV.2011.6126281>`_

    Parameters
    ----------
    margin : float, optional
        Hinge margin for ordered pairs (default ``1.0``).
    C1 : float, optional
        Weight for ordered-pair squared-hinge term (default ``1.0``).
    C2 : float, optional
        Weight for similar-pair L2 term (default ``0.25``).
    **kw
        Ignored. Accepted for API uniformity with other ranking losses.
    """

    def __init__(
        self,
        margin: float = 1.0,
        C1: float = 1.0,
        C2: float = 0.25,
        **kw,
    ) -> None:
        super().__init__()
        self.margin: float = float(margin)
        self.C1: float = float(C1)
        self.C2: float = float(C2)

    def forward(
        self,
        si: Tensor,
        sj: Tensor,
        y_ij: Tensor,
        pair_type: Optional[Tensor] = None,
        severity_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Ranking SVM loss over ordered + similar pairs.

        Parameters
        ----------
        si, sj : Tensor, shape (P,)
            Predicted ranking scores for each side of *P* pairs.
        y_ij : Tensor, shape (P,)
            Label direction ∈ {−1, 0, +1}.
        pair_type : Tensor, shape (P,), optional
            1 = ordered pair (different grades), 0 = similar pair
            (same grade). If ``None``, inferred from ``y_ij``:
            ordinal where ``y_ij ≠ 0``, similar where ``y_ij = 0``.
        severity_weight : Tensor, shape (P,), optional
            Ignored (present for API compatibility).

        Returns
        -------
        Tensor
            Scalar loss averaged over pairs.
        """
        si, sj = si.view(-1), sj.view(-1)
        y_ij = y_ij.view(-1).float()
        d = si - sj

        if pair_type is not None:
            pt = pair_type.view(-1).float()
        else:
            pt = (y_ij != 0).float()

        # Ordered pairs: squared hinge requiring correct direction
        loss_ord = torch.relu(self.margin - y_ij * d).pow(2)
        # Similar pairs: squared L2 pulling scores together
        loss_sim = d.pow(2)
        loss = self.C1 * pt * loss_ord + self.C2 * (1.0 - pt) * loss_sim
        return loss.mean()


class DeepRankSVMLoss(nn.Module):
    """Severity-Weighted Ranking SVM — RelativeAttributes + severity weighting.

    Extends :class:`RelativeAttributesLoss` with an optional **severity
    weight** that scales the per-pair loss by the ordinal distance
    between the two samples. This means misranking a severe-vs-normal
    pair (large ``|g_i − g_j|``) incurs a larger penalty than misranking
    adjacent grades.

    In the SpineRank ablation hierarchy this loss occupies the position::

        RelativeAttributes  →  DeepRankSVM (+ severity weight)  →  SpineRank (full)

    and serves as a single-component ablation of SpineRank that tests
    whether severity weighting alone improves over the vanilla Ranking
    SVM baseline.

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\frac{1}{|\\mathcal{P}|} \\sum_{(i,j)}
          w_{ij} \\cdot \\bigl[
            C_1 \\, p_{ij} \\, [\\max(0,\\; m - y_{ij} d_{ij})]^2
            + C_2 \\, (1 - p_{ij}) \\, d_{ij}^2
          \\bigr]

    where :math:`d_{ij} = s_i - s_j`, :math:`p_{ij}` is the pair type,
    and :math:`w_{ij}` is the severity weight (ordinal distance scaled
    to ``[0, UNIVERSAL_SCALE]``). When ``severity_weight`` is ``None``,
    this reduces to :class:`RelativeAttributesLoss`.

    .. note::

        This is **not** a faithful reproduction of the original Deep
        Ranking SVM paper (Lim & Learned-Miller, 2020,
        arXiv:2009.07717). The original paper uses ``C1 = C2 = 0.1``,
        ``margin = 1``, no bias in the ranking layer, and no severity
        weighting. Our :class:`RelativeAttributesLoss` more closely
        matches the original unconstrained Ranking SVM formulation
        (Parikh & Grauman, 2011) that both papers build upon.

    Reference
    ---------
    Base formulation from the Ranking SVM family:

    D. Parikh and K. Grauman, "Relative Attributes," in *Proc. IEEE
    ICCV*, 2011, pp. 503--510.

    T. Joachims, "Optimizing Search Engines using Clickthrough Data,"
    in *Proc. ACM KDD*, 2002, pp. 133--142.
    doi:`10.1145/775047.775067
    <https://doi.org/10.1145/775047.775067>`_

    Severity weighting concept from SpineRankNet (SpineRank loss).

    Parameters
    ----------
    margin : float, optional
        Hinge margin for ordinal pairs (default ``1.0``).
    C1 : float, optional
        Weight for ordinal (squared-hinge) term (default ``1.0``).
    C2 : float, optional
        Weight for similarity (L2) term (default ``0.25``).
    use_severity_weight : bool, optional
        Multiply the per-pair loss by ``severity_weight`` when provided
        (default ``True``).
    **kw
        Ignored. Accepted for API uniformity.
    """

    def __init__(
        self,
        margin: float = 1.0,
        C1: float = 1.0,
        C2: float = 0.25,
        use_severity_weight: bool = True,
        **kw,
    ) -> None:
        super().__init__()
        self.margin: float = float(margin)
        self.C1: float = float(C1)
        self.C2: float = float(C2)
        self.use_severity_weight: bool = bool(use_severity_weight)

    def forward(
        self,
        si: Tensor,
        sj: Tensor,
        y_ij: Tensor,
        pair_type: Tensor,
        severity_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute squared-hinge + L2 loss over sample pairs.

        Parameters
        ----------
        si, sj : Tensor, shape (P,)
            Predicted ranking scores for each side of *P* pairs.
        y_ij : Tensor, shape (P,)
            Label direction ∈ {−1, 0, +1}.
        pair_type : Tensor, shape (P,)
            1 = ordinal pair (different grades), 0 = similarity pair
            (same grade).
        severity_weight : Tensor, shape (P,), optional
            Per-pair weight proportional to the ordinal distance.

        Returns
        -------
        Tensor
            Scalar loss averaged over pairs.
        """
        si, sj = si.view(-1), sj.view(-1)
        y_ij = y_ij.view(-1).float()
        pair_type = pair_type.view(-1).float()
        d = si - sj
        loss_ord = torch.relu(self.margin - y_ij * d).pow(2)
        loss_sim = d.pow(2)
        loss = (
            self.C1 * pair_type * loss_ord
            + self.C2 * (1.0 - pair_type) * loss_sim
        )
        if self.use_severity_weight and severity_weight is not None:
            loss = loss * severity_weight.view_as(loss)
        return loss.mean()


class DeepRelativeAttributesLoss(nn.Module):
    """Deep Relative Attributes (DRA) — BCE ranking loss with soft targets.

    Reformulates pairwise ranking as a binary classification problem
    using a sigmoid on the score difference and binary cross-entropy.
    Ordered pairs use target 1.0 (or 0.0) and similar pairs use
    target 0.5, making the loss naturally handle both pair types in
    a unified probabilistic framework.

    This faithfully follows the Ghiaseddin implementation by Souri
    et al., which uses ``sigmoid(s_i − s_j)`` as the predicted
    probability that sample *i* ranks above *j*, and minimises the
    BCE against soft targets derived from the label direction.

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\hat{p}_{ij} = \\sigma(\\alpha \\cdot (s_i - s_j))

        t_{ij} = \\begin{cases}
          1.0 & \\text{if } y_{ij} = +1 \\\\
          0.0 & \\text{if } y_{ij} = -1 \\\\
          0.5 & \\text{if } y_{ij} = 0  \\quad\\text{(similar pair)}
        \\end{cases}

        \\mathcal{L} = -\\frac{1}{|\\mathcal{P}|}
          \\sum_{(i,j)\\in\\mathcal{P}} w_{ij}\\bigl[
            t_{ij} \\log \\hat{p}_{ij}
            + (1 - t_{ij}) \\log (1 - \\hat{p}_{ij})
          \\bigr]

    where :math:`\\alpha` is a scaling parameter and :math:`w_{ij}` is
    an optional severity weight.

    Paper fidelity
    ~~~~~~~~~~~~~~
    When ``severity_weight=None``, this is a **faithful** implementation
    of the original DRA loss (Souri et al., 2016, Eq. 2):

    * ``sigmoid(s_i − s_j)`` with no extra scaling (``alpha=1``) ✓
    * Soft targets ``{0, 0.5, 1}`` for ``{<, =, >}`` ✓
    * BCE loss averaged over pairs ✓

    The optional ``severity_weight`` post-multiplication is our
    extension (not in the original paper). When the training loop
    passes ``severity_weight``, it acts as a per-pair importance
    weight — equivalent to adding a SpineRank-style severity
    component.

    Reference
    ---------
    A. Souri, E. Noury, and E. Adeli, "Deep Relative Attributes,"
    in *Proc. Asian Conference on Computer Vision (ACCV)*, 2016,
    pp. 118--133. doi:`10.1007/978-3-319-54187-7_8
    <https://doi.org/10.1007/978-3-319-54187-7_8>`_

    Official implementation (Ghiaseddin):
    https://github.com/yassersouri/ghiaseddin

    Parameters
    ----------
    margin : float, optional
        Scaling factor :math:`\\alpha` on the score difference
        (default ``1.0``). Higher values make the sigmoid steeper.
    **kw
        Ignored. Accepted for API uniformity.
    """

    def __init__(self, margin: float = 1.0, **kw) -> None:
        super().__init__()
        self.alpha: float = float(margin)

    def forward(
        self,
        si: Tensor,
        sj: Tensor,
        y_ij: Tensor,
        pair_type: Optional[Tensor] = None,
        severity_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute BCE ranking loss with soft targets.

        Parameters
        ----------
        si, sj : Tensor, shape (P,)
            Predicted ranking scores for each side of *P* pairs.
        y_ij : Tensor, shape (P,)
            Label direction ∈ {−1, 0, +1}.
        pair_type : Tensor, shape (P,), optional
            Ignored (targets derived from ``y_ij`` directly).
        severity_weight : Tensor, shape (P,), optional
            Per-pair weight proportional to the ordinal distance.

        Returns
        -------
        Tensor
            Scalar loss averaged over pairs.
        """
        si, sj = si.view(-1), sj.view(-1)
        y_ij = y_ij.view(-1).float()

        # Predicted probability that i ranks above j
        logit = self.alpha * (si - sj)

        # Soft target: y_ij ∈ {-1, 0, +1} → target ∈ {0.0, 0.5, 1.0}
        target = (y_ij + 1.0) / 2.0

        loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        if severity_weight is not None:
            loss = loss * severity_weight.view_as(loss)
        return loss.mean()


class JustNoticeableDifferencesLoss(nn.Module):
    """Just Noticeable Differences — Ranking SVM with JND gating.

    Extends the Relative Attributes Ranking SVM formulation with a
    **Just-Noticeable-Difference threshold** that reclassifies pairs
    whose ordinal distance is below a perceptual threshold as
    "similar" (indistinguishable) regardless of their true label
    direction.

    This models the clinical intuition that adjacent severity grades
    are often unreliably distinguishable by human observers. Pairs
    with severity weight below ``jnd_threshold`` are treated as
    **similar pairs** (L2 pull-together), while pairs above the
    threshold are treated as **ordered pairs** (squared hinge).

    The approach faithfully adapts Yu & Grauman's concept of
    learning when attribute differences are "just noticeable" to the
    Ranking SVM framework. Whereas Yu & Grauman use a Bayesian
    local model for distinguishability, we use the ordinal grade
    distance as a proxy for perceptual distinguishability.

    Formulation
    ~~~~~~~~~~~
    For each pair (i, j), define the effective pair type:

    .. math::

        \\tilde{p}_{ij} = \\begin{cases}
          1 & \\text{if } w_{ij} \\ge \\tau \\text{ and } y_{ij} \\neq 0 \\\\
          0 & \\text{otherwise (indistinguishable)}
        \\end{cases}

    Then apply the Ranking SVM objective:

    .. math::

        \\mathcal{L} = \\frac{1}{|\\mathcal{P}|}
          \\sum_{(i,j)\\in\\mathcal{P}}
          \\bigl[
            C_1 \\, \\tilde{p}_{ij} \\, [\\max(0,\\; m - y_{ij} d_{ij})]^2
            + C_2 \\, (1 - \\tilde{p}_{ij}) \\, d_{ij}^2
          \\bigr]

    where :math:`d_{ij} = s_i - s_j`, :math:`\\tau` is the JND
    threshold, and :math:`w_{ij}` is the severity weight.

    Reference
    ---------
    A. Yu and K. Grauman, "Just Noticeable Differences in Visual
    Attributes," in *Proc. IEEE International Conference on Computer
    Vision (ICCV)*, 2015, pp. 2416--2424.
    doi:`10.1109/ICCV.2015.278
    <https://doi.org/10.1109/ICCV.2015.278>`_

    Builds on the Ranking SVM formulation from:

    D. Parikh and K. Grauman, "Relative Attributes," in *Proc. IEEE
    ICCV*, 2011, pp. 503--510.

    Parameters
    ----------
    margin : float, optional
        Hinge margin for ordered pairs (default ``1.0``).
    C1 : float, optional
        Weight for ordered-pair squared-hinge term (default ``1.0``).
    C2 : float, optional
        Weight for similar-pair L2 term (default ``0.25``).
    jnd_threshold : float, optional
        Minimum severity weight for a pair to be treated as "ordered"
        (distinguishable). Pairs below this threshold are treated as
        "similar" regardless of label direction (default ``1.0``).
    **kw
        Ignored. Accepted for API uniformity.
    """

    def __init__(
        self,
        margin: float = 1.0,
        C1: float = 1.0,
        C2: float = 0.25,
        jnd_threshold: float = 1.0,
        **kw,
    ) -> None:
        super().__init__()
        self.margin: float = float(margin)
        self.C1: float = float(C1)
        self.C2: float = float(C2)
        self.jnd_threshold: float = float(jnd_threshold)

    def forward(
        self,
        si: Tensor,
        sj: Tensor,
        y_ij: Tensor,
        pair_type: Optional[Tensor] = None,
        severity_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute JND-gated Ranking SVM loss.

        Parameters
        ----------
        si, sj : Tensor, shape (P,)
            Predicted ranking scores for each side of *P* pairs.
        y_ij : Tensor, shape (P,)
            Label direction ∈ {−1, 0, +1}.
        pair_type : Tensor, shape (P,), optional
            Original pair type (1 = ordinal, 0 = similar). Overridden
            by JND gating: ordinal pairs with severity below
            ``jnd_threshold`` are reclassified as similar.
        severity_weight : Tensor, shape (P,), optional
            Per-pair ordinal distance on ``[0, UNIVERSAL_SCALE]``.

        Returns
        -------
        Tensor
            Scalar loss averaged over pairs.
        """
        si, sj = si.view(-1), sj.view(-1)
        y_ij = y_ij.view(-1).float()
        d = si - sj

        sw = (
            severity_weight.view(-1)
            if severity_weight is not None
            else torch.ones_like(si)
        )

        # JND gating: pairs are "distinguishable" (ordered) only if
        # severity weight >= threshold AND labels differ.
        distinguishable = ((sw >= self.jnd_threshold) & (y_ij != 0)).float()

        # Ordered pairs: squared hinge requiring correct direction.
        loss_ord = torch.relu(self.margin - y_ij * d).pow(2)
        # Similar / indistinguishable pairs: squared L2 pulling together.
        loss_sim = d.pow(2)

        loss = (
            self.C1 * distinguishable * loss_ord
            + self.C2 * (1.0 - distinguishable) * loss_sim
        )
        return loss.mean()


class RankNetLoss(nn.Module):
    """RankNet probabilistic pairwise ranking loss.

    Models the probability that sample *i* should be ranked above *j*
    using a logistic (sigmoid) function on the scaled score difference.
    The loss is the binary cross-entropy between this probability and
    the ground-truth ordering, implemented via ``softplus`` for
    numerical stability.

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\frac{1}{|\\mathcal{P}|}
          \\sum_{(i,j)\\in\\mathcal{P}}
          w_{ij} \\, \\log\\bigl(1 + e^{-\\sigma \\, y_{ij}(s_i - s_j)}\\bigr)

    where :math:`\\sigma` is a scaling parameter (``margin``) and
    :math:`w_{ij}` is an optional severity weight.

    Paper fidelity
    ~~~~~~~~~~~~~~
    When ``severity_weight=None``, this is a **faithful** implementation
    of the original RankNet loss (Burges et al., 2005):

    * ``P_ij = sigmoid(σ · (s_i − s_j))`` with ``σ`` = ``margin`` ✓
    * Loss = ``log(1 + exp(-σ · y_ij · (s_i − s_j)))`` = softplus ✓
    * Averaged over pairs ✓

    The optional ``severity_weight`` post-multiplication is our
    extension (not in the original paper). When the training loop
    passes ``severity_weight``, it acts as a per-pair importance
    weight. The original RankNet does not distinguish pair types or
    apply severity-based weighting.

    Reference
    ---------
    C. Burges et al., "Learning to Rank using Gradient Descent," in
    *Proc. International Conference on Machine Learning (ICML)*, 2005,
    pp. 89--96. doi:`10.1145/1102351.1102363
    <https://doi.org/10.1145/1102351.1102363>`_

    Parameters
    ----------
    margin : float, optional
        Scaling factor :math:`\\sigma` on the score difference
        (default ``1.0``).
    **kw
        Ignored. Accepted for API uniformity.
    """

    def __init__(self, margin: float = 1.0, **kw) -> None:
        super().__init__()
        self.margin: float = float(margin)

    def forward(
        self,
        si: Tensor,
        sj: Tensor,
        y_ij: Tensor,
        pair_type: Optional[Tensor] = None,
        severity_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute RankNet cross-entropy loss over sample pairs.

        Parameters
        ----------
        si, sj : Tensor, shape (P,)
            Predicted ranking scores for each side of *P* pairs.
        y_ij : Tensor, shape (P,)
            Label direction ∈ {−1, 0, +1}.
        pair_type : Tensor, shape (P,), optional
            Ignored.
        severity_weight : Tensor, shape (P,), optional
            Per-pair weight proportional to the ordinal distance.

        Returns
        -------
        Tensor
            Scalar loss averaged over pairs.
        """
        si, sj = si.view(-1), sj.view(-1)
        y_ij = y_ij.view(-1).float()
        loss = F.softplus(-self.margin * y_ij * (si - sj))
        if severity_weight is not None:
            loss = loss * severity_weight.view_as(loss)
        return loss.mean()


# ============================================================================
# Pointwise / triplet losses
# ============================================================================



# ============================================================================
# Pointwise regression baselines
# ============================================================================

class MSERegressionLoss(nn.Module):
    """Mean Squared Error regression loss for ordinal ranking.

    Direct regression from predicted ranking score to scaled ordinal
    label. The simplest baseline: no pair generation, no margin —
    just minimise the squared distance between predicted scores and
    ground-truth grades mapped to the universal scale.

    This is a **pointwise** loss (``pointwise = True``): it receives
    ``(scores, labels)`` directly and the training loop skips pair
    generation.

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\frac{1}{N}
          \\sum_{i=1}^{N} (s_i - g_i)^2

    where :math:`g_i` is the label scaled to ``[0, UNIVERSAL_SCALE]``.

    Reference
    ---------
    Standard MSE / L2 loss. Used as a regression baseline in:

    C. Li and A. C. Bovik, "Content-Weighted Video Quality Assessment
    Using a Three-Component Image Model," *Journal of Electronic
    Imaging*, 2010 (among many others using MSE for quality / severity
    regression).

    Parameters
    ----------
    **kw
        Ignored. Accepted for API uniformity.
    """

    pointwise: bool = True  # flag checked by train_epoch

    def __init__(self, **kw) -> None:
        super().__init__()

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        """Compute MSE between predicted scores and target labels.

        Parameters
        ----------
        scores : Tensor, shape (N,)
            Predicted ranking scores.
        labels : Tensor, shape (N,)
            Target ordinal labels (already scaled to score range by
            the training loop).

        Returns
        -------
        Tensor
            Scalar MSE loss.
        """
        scores = scores.view(-1)
        labels = labels.view(-1).float()
        return F.mse_loss(scores, labels)


class MAERegressionLoss(nn.Module):
    """Mean Absolute Error (L1) regression loss for ordinal ranking.

    Direct regression using L1 distance from predicted ranking score
    to scaled ordinal label. More robust to outliers than MSE and
    produces sparser gradients (constant magnitude regardless of
    error size).

    This is a **pointwise** loss (``pointwise = True``): it receives
    ``(scores, labels)`` directly and the training loop skips pair
    generation.

    Formulation
    ~~~~~~~~~~~
    .. math::

        \\mathcal{L} = \\frac{1}{N}
          \\sum_{i=1}^{N} |s_i - g_i|

    where :math:`g_i` is the label scaled to ``[0, UNIVERSAL_SCALE]``.

    Reference
    ---------
    Standard L1 / MAE loss. Preferred over MSE in robust regression
    settings:

    P. J. Huber, "Robust Estimation of a Location Parameter," *Annals
    of Mathematical Statistics*, vol. 35, no. 1, pp. 73--101, 1964.

    Parameters
    ----------
    **kw
        Ignored. Accepted for API uniformity.
    """

    pointwise: bool = True  # flag checked by train_epoch

    def __init__(self, **kw) -> None:
        super().__init__()

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        """Compute MAE between predicted scores and target labels.

        Parameters
        ----------
        scores : Tensor, shape (N,)
            Predicted ranking scores.
        labels : Tensor, shape (N,)
            Target ordinal labels (already scaled to score range by
            the training loop).

        Returns
        -------
        Tensor
            Scalar MAE loss.
        """
        scores = scores.view(-1)
        labels = labels.view(-1).float()
        return F.l1_loss(scores, labels)


# ============================================================================
# SpineRank — custom loss for spinal pathology severity ranking
# ============================================================================



RANKING_LOSSES: dict[str, type[nn.Module]] = {
    "RelativeAttributes":        RelativeAttributesLoss,
    "DeepRankSVM":               DeepRankSVMLoss,
    "DeepRelativeAttributes":    DeepRelativeAttributesLoss,
    "JustNoticeableDifferences": JustNoticeableDifferencesLoss,
    "RankNet":                   RankNetLoss,
    "TripletOrdinal":            TripletMarginOrdinalLoss,
    "MSE":                       MSERegressionLoss,
    "MAE":                       MAERegressionLoss,
    "SpineRank":                 SpineRankLoss,
    # Short aliases for convenience.
    "RA":  RelativeAttributesLoss,
    "DRA": DeepRelativeAttributesLoss,
    "JND": JustNoticeableDifferencesLoss,
}
