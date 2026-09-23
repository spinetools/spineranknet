"""Rank–classification agreement and calibration utilities.

This module provides three complementary mechanisms that strengthen the
interaction between classification and ranking branches in SpineRankNet:

1. **Agreement regularization** (:func:`compute_agreement_loss`)
   A training-time MSE loss between the ranking score and the expected
   grade derived from classification logits. This enforces cross-branch
   consistency — unlike CORN (which enforces ordinal monotonicity *within*
   the classification head alone), the agreement loss links the two
   independent branches so that their outputs are mutually consistent.

2. **Expected grade from logits** (:func:`compute_expected_grade`)
   Converts classification logits (either standard K-logit softmax or
   K-1 cumulative logits from CORN/CORAL/CLM) into a continuous severity
   score on the universal ``[0, UNIVERSAL_SCALE]`` range.

3. **Rank-calibrated classification** (:func:`rank_calibrated_classify`)
   An inference-time method that fuses softmax classification probabilities
   with a ranking-derived Gaussian prior via geometric mean. This
   produces ordinal-consistent predictions that respect the severity
   ordering, unlike raw softmax which treats adjacent grades as
   independent classes.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from spineranknet.networks.heads.ranking_heads import UNIVERSAL_SCALE


__all__ = [
    "compute_expected_grade",
    "compute_agreement_loss",
    "rank_calibrated_classify",
]


def _cumulative_to_class_probs(logits: Tensor) -> Tensor:
    """Convert K-1 cumulative ordinal logits → K class probabilities.

    For CORN/CORAL/CLM losses that output K-1 logits representing
    P(Y > k | x) for k = 0, ..., K-2.

    Parameters
    ----------
    logits : Tensor
        Shape ``(B, K-1)`` cumulative logits.

    Returns
    -------
    Tensor
        Shape ``(B, K)`` class probabilities (non-negative, sums to 1).
    """
    cum = torch.sigmoid(logits)  # (B, K-1)  P(Y > k)
    ones = torch.ones(logits.shape[0], 1, device=logits.device)
    zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
    cum_ext = torch.cat([ones, cum, zeros], dim=1)  # (B, K+1)
    probs = cum_ext[:, :-1] - cum_ext[:, 1:]  # (B, K)
    return probs.clamp(min=0)


def compute_expected_grade(logits: Tensor, num_classes: int) -> Tensor:
    """Compute expected grade from classification logits.

    Maps logits to a continuous severity score on [0, UNIVERSAL_SCALE]
    using ``E[grade] = sum(probs * linspace(0, SCALE, K))``.

    Handles both standard K-logit (softmax) and K-1-logit
    (CORN/CORAL/CLM cumulative) cases automatically.

    Parameters
    ----------
    logits : Tensor
        Shape ``(B, K)`` or ``(B, K-1)``.
    num_classes : int
        True number of classes K.

    Returns
    -------
    Tensor
        Shape ``(B,)`` expected grade on [0, UNIVERSAL_SCALE].
    """
    if logits.shape[1] == num_classes - 1:
        probs = _cumulative_to_class_probs(logits)
    else:
        probs = F.softmax(logits, dim=1)

    grade_values = torch.linspace(
        0.0, UNIVERSAL_SCALE, num_classes, device=logits.device,
    )
    return (probs * grade_values).sum(dim=1)


def compute_agreement_loss(
    rank_scores: Tensor,
    clf_logits: Tensor,
    num_classes: int,
    detach_clf: bool = True,
) -> Tensor:
    """MSE between ranking score and expected grade from clf logits.

    This enforces cross-branch consistency: the ranking head's continuous
    score should agree with the severity implied by the classification
    probability distribution.

    **How this differs from CORN**: CORN enforces P(Y > k | x) monotonicity
    *within* the classification head.  Agreement regularization links the
    two *independent* branches (classification and ranking) so their
    outputs are mutually consistent.

    Parameters
    ----------
    rank_scores : Tensor
        Shape ``(B,)`` ranking scores on [0, UNIVERSAL_SCALE].
    clf_logits : Tensor
        Shape ``(B, K)`` or ``(B, K-1)`` classification logits.
    num_classes : int
        True number of classes K.
    detach_clf : bool
        If True (default), stop gradients from flowing into the
        classification head — the ranking branch learns to match clf
        (classification leads).  If False, both branches receive
        gradients (mutual agreement).

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    expected = compute_expected_grade(clf_logits, num_classes)
    if detach_clf:
        expected = expected.detach()
    return F.mse_loss(rank_scores, expected)


def rank_calibrated_classify(
    clf_logits: Tensor,
    rank_scores: Tensor,
    num_classes: int,
    sigma: float = 1.0,
    alpha: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Fuse classification probabilities with a ranking-derived Gaussian prior.

    The key insight: classification gives discrete decisions but ranking
    gives continuous severity ordering.  By fusing both via geometric
    mean, we get ordinal-consistent probabilities that respect the
    severity ordering — unlike raw softmax which treats adjacent grades
    as independent classes.

    Fusion formula:
        ``p_fused(k) ∝ p_clf(k)^α * p_rank(k)^(1-α)``
    where ``p_rank(k) ∝ exp(-(grade_k - s)² / (2σ²))`` is a Gaussian
    centered at the ranking score ``s``.

    Parameters
    ----------
    clf_logits : Tensor
        Shape ``(B, K)`` or ``(B, K-1)`` classification logits.
    rank_scores : Tensor
        Shape ``(B,)`` ranking scores on [0, UNIVERSAL_SCALE].
    num_classes : int
        True number of classes K.
    sigma : float
        Gaussian width on the [0, UNIVERSAL_SCALE] scale.
        Default 1.0 means ±1 unit covers ~68% of the prior mass.
    alpha : float
        Classification weight in fusion.  alpha=0.5 gives equal weight.
        Higher alpha trusts classification more; lower trusts ranking more.

    Returns
    -------
    fused_probs : Tensor
        Shape ``(B, K)`` calibrated probabilities.
    fused_preds : Tensor
        Shape ``(B,)`` argmax predictions (integer class indices).
    """
    # ── Classification probabilities ──────────────────────────────────
    if clf_logits.shape[1] == num_classes - 1:
        p_clf = _cumulative_to_class_probs(clf_logits)
    else:
        p_clf = F.softmax(clf_logits, dim=1)  # (B, K)

    # ── Gaussian prior from ranking score ─────────────────────────────
    grade_values = torch.linspace(
        0.0, UNIVERSAL_SCALE, num_classes, device=rank_scores.device,
    )  # (K,)
    diff = grade_values.unsqueeze(0) - rank_scores.unsqueeze(1)  # (B, K)
    p_rank = torch.exp(-diff.pow(2) / (2.0 * sigma ** 2))
    p_rank = p_rank / p_rank.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # ── Geometric mean fusion ─────────────────────────────────────────
    eps = 1e-8
    log_fused = (alpha * torch.log(p_clf.clamp(min=eps))
                 + (1.0 - alpha) * torch.log(p_rank.clamp(min=eps)))
    fused_probs = F.softmax(log_fused, dim=1)  # renormalize

    return fused_probs, fused_probs.argmax(dim=1)
