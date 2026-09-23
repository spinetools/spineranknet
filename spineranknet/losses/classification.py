#!/usr/bin/env python3
"""
SpineRankNet — Classification & ordinal losses + multi-task weighting
====================================================================

Provides:

1. **Ordinal loss functions** — EMD, MAE, Soft-CE, CLM, plus wrappers for
   CORN / CORAL from ``coral_pytorch``.
2. **``LOSS_REGISTRY``** — name → callable look-up.
3. **Multi-task loss weighting strategies**:

   * ``UniformWeighting``   — equal weight for all tasks (baseline).
   * ``UncertaintyWeighting`` — learns per-task homoscedastic uncertainty
     (Kendall et al. CVPR 2018).
   * ``DynamicWeightAveraging`` — weights proportional to the rate of loss
     decrease across epochs (Liu et al. CVPR 2019).
   * ``DifficultyWeighting`` — static weights proportional to √num_classes
     so that harder (more-class) tasks get more attention.

All weighting modules inherit from ``MultiTaskWeighting`` and expose a
``reweight(losses_dict, …) → weighted_total`` interface used by the
training loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# coral-pytorch is optional at import time — guard for environments that
# don't have it installed.
try:
    from coral_pytorch.losses import corn_loss, coral_loss
    from coral_pytorch.dataset import corn_label_from_logits
    _HAS_CORAL = True
except ImportError:
    _HAS_CORAL = False

    def corn_loss(*a: Any, **kw: Any) -> torch.Tensor:
        raise ImportError("coral_pytorch is required for CORN loss")

    def coral_loss(*a: Any, **kw: Any) -> torch.Tensor:
        raise ImportError("coral_pytorch is required for CORAL loss")

    def corn_label_from_logits(*a: Any, **kw: Any) -> torch.Tensor:
        raise ImportError("coral_pytorch is required for CORN predictions")


# ============================================================================
# ORDINAL LOSS FUNCTIONS
# ============================================================================

def earth_movers_distance_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wasserstein / EMD loss on ordinal distributions."""
    labels_oh = F.one_hot(labels, num_classes).float()
    pred_cdf = torch.cumsum(F.softmax(logits, dim=1), dim=1)
    true_cdf = torch.cumsum(labels_oh, dim=1)
    emd = torch.mean(torch.abs(pred_cdf - true_cdf), dim=1)
    if weight is not None:
        emd = emd * weight[labels]
    return emd.mean()


def mean_absolute_error_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expectation-based MAE: E[pred_class] vs true label."""
    probs = F.softmax(logits, dim=1)
    class_vals = torch.arange(num_classes, device=logits.device, dtype=torch.float32)
    pred_vals = (probs * class_vals).sum(dim=1)
    mae = torch.abs(pred_vals - labels.float())
    if weight is not None:
        mae = mae * weight[labels]
    return mae.mean()


def soft_ordinal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Soft label CE with Gaussian kernel centred on the true class."""
    class_idx = torch.arange(num_classes, device=logits.device).float()
    dist = (class_idx - labels.unsqueeze(1).float()) ** 2
    soft = torch.exp(-dist / (2 * sigma ** 2))
    soft = soft / soft.sum(dim=1, keepdim=True)
    log_p = F.log_softmax(logits, dim=1)
    loss = -(soft * log_p).sum(dim=1)
    if weight is not None:
        loss = loss * weight[labels]
    return loss.mean()


def cumulative_link_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cumulative link model (proportional odds) loss."""
    eps = 1e-7
    cum_probs = torch.sigmoid(logits)
    probs = torch.zeros(logits.size(0), num_classes, device=logits.device)
    probs[:, 0] = cum_probs[:, 0]
    probs[:, 1:-1] = cum_probs[:, 1:] - cum_probs[:, :-1]
    probs[:, -1] = 1.0 - cum_probs[:, -1]
    probs = probs.clamp(eps, 1.0 - eps)
    loss = -torch.log(probs[torch.arange(logits.size(0)), labels])
    if weight is not None:
        loss = loss * weight[labels]
    return loss.mean()


# ============================================================================
# LOSS REGISTRY
# ============================================================================

LOSS_REGISTRY = {
    "ce": lambda logits, labels, nc, w: F.cross_entropy(logits, labels, weight=w),
    "emd": earth_movers_distance_loss,
    "wasserstein": earth_movers_distance_loss,
    "mae": mean_absolute_error_loss,
    "soft_ce": soft_ordinal_cross_entropy,
    "soft_ordinal": soft_ordinal_cross_entropy,
    "clm": cumulative_link_loss,
    "corn": lambda logits, labels, nc, w: corn_loss(logits, labels, nc),
    "coral": lambda logits, labels, nc, w: coral_loss(logits, labels, nc),
}


def get_predictions(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Convert raw logits to predicted class indices, accounting for loss type."""
    if loss_type == "corn":
        return corn_label_from_logits(logits)
    if loss_type in ("coral", "clm"):
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return torch.argmax(logits, dim=1)


# ============================================================================
# MULTI-TASK LOSS WEIGHTING
# ============================================================================

class MultiTaskWeighting(nn.Module):
    """Base class for multi-task loss weighting strategies.

    Subclasses must implement ``reweight(losses_dict)`` which receives a
    dict ``{task_name: scalar_loss}`` and returns a weighted scalar total.
    """

    def reweight(
        self,
        losses: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError


class UniformWeighting(MultiTaskWeighting):
    """Equal weight = 1 for every task (baseline)."""

    def reweight(self, losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return sum(losses.values())


class UncertaintyWeighting(MultiTaskWeighting):
    """Homoscedastic uncertainty weighting (Kendall et al. CVPR 2018).

    Learns a per-task log-variance ``log_σ²``.  The effective loss for task *t*
    is ``L_t / (2·σ_t²) + ½·log(σ_t²)``, which automatically down-weights
    noisy / hard tasks while penalising excessive uncertainty.

    Reference: *Multi-Task Learning Using Uncertainty to Weigh Losses*,
    Kendall, Gal & Cipolla, CVPR 2018.
    """

    def __init__(self, tasks: List[str]) -> None:
        super().__init__()
        # Initialise log(σ²) = 0  ⟹  σ² = 1  (equal initial weighting)
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in tasks
        })

    def reweight(self, losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for task, loss in losses.items():
            if task in self.log_vars:
                log_var = self.log_vars[task]
                # L_t / (2·σ²) + ½·log(σ²)
                total = total + loss / (2.0 * torch.exp(log_var)) + 0.5 * log_var
            else:
                total = total + loss
        return total

    def get_weights(self) -> Dict[str, float]:
        """Return effective weights 1/(2·σ²) for logging."""
        return {t: float(1.0 / (2.0 * torch.exp(lv).item()))
                for t, lv in self.log_vars.items()}


class DynamicWeightAveraging(MultiTaskWeighting):
    """Dynamic Weight Averaging (Liu et al. CVPR 2019).

    Each task's weight at epoch *e* is proportional to
    ``exp(r_t(e) / T)`` where ``r_t(e) = L_t(e-1) / L_t(e-2)`` is the
    loss ratio capturing the rate of decrease, and ``T`` is a temperature.
    Tasks whose loss decreases more slowly get higher weights.

    Call ``update(losses_dict)`` at the end of each epoch.
    """

    def __init__(self, tasks: List[str], temperature: float = 2.0) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.temperature = temperature
        self._history: List[Dict[str, float]] = []
        self._weights: Dict[str, float] = {t: 1.0 for t in tasks}

    def update(self, epoch_losses: Dict[str, float]) -> None:
        """Call after each epoch with the mean losses."""
        self._history.append(dict(epoch_losses))
        if len(self._history) >= 2:
            ratios = {}
            for t in self.tasks:
                prev = max(self._history[-2].get(t, 1e-8), 1e-8)
                curr = max(self._history[-1].get(t, 1e-8), 1e-8)
                ratios[t] = curr / prev

            exp_r = {t: np.exp(r / self.temperature) for t, r in ratios.items()}
            total = sum(exp_r.values())
            n = len(self.tasks)
            self._weights = {t: n * exp_r[t] / total for t in self.tasks}

    def reweight(self, losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for task, loss in losses.items():
            w = self._weights.get(task, 1.0)
            total = total + w * loss
        return total

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)


class DifficultyWeighting(MultiTaskWeighting):
    """Static weights proportional to √num_classes.

    Harder tasks (more classes) receive proportionally higher weight.
    Weights are normalised so that their mean equals 1.
    """

    def __init__(self, tasks: List[str], task_definitions: Dict[str, Any]) -> None:
        super().__init__()
        raw = {t: np.sqrt(task_definitions[t]["num_classes"]) for t in tasks}
        mean_w = np.mean(list(raw.values()))
        self._weights = {t: float(w / mean_w) for t, w in raw.items()}

    def reweight(self, losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for task, loss in losses.items():
            total = total + self._weights.get(task, 1.0) * loss
        return total

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)


# ── factory ─────────────────────────────────────────────────────────────────

WEIGHTING_REGISTRY = {
    "uniform": UniformWeighting,
    "uncertainty": UncertaintyWeighting,
    "dwa": DynamicWeightAveraging,
    "difficulty": DifficultyWeighting,
}


def build_task_weighting(
    strategy: str,
    tasks: List[str],
    task_definitions: Optional[Dict] = None,
    temperature: float = 2.0,
) -> MultiTaskWeighting:
    """Construct a ``MultiTaskWeighting`` module from a strategy name.

    Parameters
    ----------
    strategy : str
        One of ``"uniform"``, ``"uncertainty"``, ``"dwa"``, ``"difficulty"``.
    tasks : list of str
        Task names.
    task_definitions : dict, optional
        Required for ``"difficulty"`` strategy.
    temperature : float
        Temperature for DWA.
    """
    strategy = strategy.lower()
    if strategy == "uniform":
        return UniformWeighting()
    if strategy == "uncertainty":
        return UncertaintyWeighting(tasks)
    if strategy == "dwa":
        return DynamicWeightAveraging(tasks, temperature=temperature)
    if strategy == "difficulty":
        if task_definitions is None:
            from spineranknet.config import TASK_DEFINITIONS as _td
            task_definitions = _td
        return DifficultyWeighting(tasks, task_definitions)
    raise ValueError(f"Unknown weighting strategy: {strategy!r}. "
                     f"Available: {list(WEIGHTING_REGISTRY.keys())}")
