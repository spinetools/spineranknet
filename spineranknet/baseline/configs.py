"""Configuration types for the baseline trainer.

This module owns :class:`HybridConfig`, the dataclass consumed by
:mod:`spineranknet.baseline.train_ranking_mse`. The ablation **preset
registry** (``ABLATION_PRESETS`` / ``apply_ablation_preset``) was moved to
:mod:`spineranknet.experiments.presets` so the core package advertises only
the paper configuration; the trainer imports the presets from there when an
``--ablation`` flag is requested.
"""
from __future__ import annotations

import logging
from dataclasses import fields as _dc_fields
from typing import Any

import yaml

from spineranknet.config import (
    GradingConfig,
    ORDINAL_TASKS_13,
    RankingConfig,
    STANDARD_TASKS_13,
    expand_env_paths,
)


__all__ = ["HybridConfig"]

logger = logging.getLogger(__name__)


class HybridConfig(RankingConfig):
    """Adds triplet + task weighting fields on top of RankingConfig."""

    def __init__(self, **kwargs: Any) -> None:
        # hybrid-specific defaults (extract before super().__init__)
        self.RANKING_LOSS_WEIGHT: float = float(kwargs.pop("RANKING_LOSS_WEIGHT", 1.0))
        self.USE_TRIPLET: bool = kwargs.pop("USE_TRIPLET", False)
        self.TRIPLET_LOSS_WEIGHT: float = float(kwargs.pop("TRIPLET_LOSS_WEIGHT", 0.5))
        self.TRIPLET_MARGIN: float = float(kwargs.pop("TRIPLET_MARGIN", 1.0))
        self.TASK_WEIGHT_STRATEGY: str = kwargs.pop("TASK_WEIGHT_STRATEGY", "uniform")
        self.DWA_TEMPERATURE: float = float(kwargs.pop("DWA_TEMPERATURE", 2.0))
        self.RETURN_RAW: bool = kwargs.pop("RETURN_RAW", False)
        self.FINETUNE: bool = kwargs.pop("FINETUNE", False)
        self.TEST_T2_S1_ONLY: bool = kwargs.pop("TEST_T2_S1_ONLY", True)
        self.ABLATION: str = kwargs.pop("ABLATION", "")
        self.ENCODER_ONLY: bool = kwargs.pop("ENCODER_ONLY", False)
        self.tags: list = kwargs.pop("tags", None)
        # Drop keys RankingConfig does not know (e.g. options of older runs'
        # train_config.yaml that are not part of this release) with a warning,
        # so such configs still load instead of raising TypeError.
        _known = {f.name for f in _dc_fields(RankingConfig)}
        _unknown = sorted(k for k in kwargs if k not in _known)
        if _unknown:
            logger.warning("HybridConfig: ignoring unknown config keys: %s",
                           ", ".join(_unknown))
            for _k in _unknown:
                kwargs.pop(_k)

        # let RankingConfig handle the rest
        super().__init__(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "HybridConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        # Keys RankingConfig does not know are dropped: do not expand them.
        _known = {fld.name for fld in _dc_fields(RankingConfig)}
        return cls(**expand_env_paths(raw, skip=raw.keys() - _known))

    @classmethod
    def from_ranking_config(cls, rc: RankingConfig, **overrides: Any) -> "HybridConfig":
        """Create a HybridConfig from an existing RankingConfig + overrides."""
        from dataclasses import fields as dc_fields
        d = {fld.name: getattr(rc, fld.name) for fld in dc_fields(rc)}
        d.update(overrides)
        return cls(**d)

    @classmethod
    def from_grading_config(cls, gc: GradingConfig, **overrides: Any) -> "HybridConfig":
        """Create a HybridConfig from a GradingConfig (classification-only).

        Sets ranking weights to zero by default so the training loop
        behaves as pure classification.
        """
        from dataclasses import fields as dc_fields
        d = {fld.name: getattr(gc, fld.name) for fld in dc_fields(gc)}
        d.setdefault("RANKING_LOSS_WEIGHT", 0.0)
        d.setdefault("USE_TRIPLET", False)
        d.setdefault("TRIPLET_LOSS_WEIGHT", 0.0)
        d.update(overrides)
        return cls(**d)


