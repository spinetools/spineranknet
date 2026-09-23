"""Ablation / experiment surface for SpineRankNet.

Everything required to reproduce the paper's ablation tables lives here, kept
separate from the core package so that ``import spineranknet`` advertises only
the published model and loss. Importing this subpackage:

* registers the deeper backbones (ResNet-34/50/101/152) into the shared
  encoder registry (see :mod:`spineranknet.experiments.encoders`), and
* exposes the full ranking-loss registry and the ablation presets.

The ablation-capable trainer
(:mod:`spineranknet.baseline.train_ranking_mse`) imports from here so that
``--arch``, ``--ranking_loss`` and ``--ablation`` flags all resolve.
"""
from __future__ import annotations

# Importing ``encoders`` registers the deeper backbones on package import.
from spineranknet.experiments import encoders  # noqa: F401
from spineranknet.experiments.comparison_losses import RANKING_LOSSES
from spineranknet.experiments.encoders import EXTRA_ENCODERS
from spineranknet.experiments.presets import ABLATION_PRESETS, apply_ablation_preset

__all__ = [
    "RANKING_LOSSES",
    "EXTRA_ENCODERS",
    "ABLATION_PRESETS",
    "apply_ablation_preset",
]
