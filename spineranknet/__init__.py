"""SpineRankNet — continuous spine degeneration severity scores.

Code for *Be Indiscrete: The Benefits of Learning Continuous Spine
Degeneration Severity Scores* (MICCAI 2026).

SpineRankNet is a multi-task 3D ResNet encoder with per-task classification
heads and logit-based ranking heads, trained in two phases (classification
pre-training, then fine-tuning with the SpineRank pairwise ranking loss).

Entry point
-----------
Training and evaluation::

    python -m spineranknet.baseline.train_ranking_mse --help

Modules
-------
``spineranknet.baseline``
    Unified training / evaluation script (``train_ranking_mse``) and the
    ``HybridConfig`` YAML schema (``configs``).

``spineranknet.config``
    ``TASK_DEFINITIONS`` and grading scales.

``spineranknet.losses``
    Classification / ordinal losses and ranking losses (SpineRank,
    RankNet, DeepRankSVM, ...).

``spineranknet.networks``
    Multi-task encoder, classification and ranking heads.

``spineranknet.dataloaders``
    GENODISC intervertebral-disc dataset.

See Also
--------
* Project page: https://spinetools.github.io/spineranknet/
* Code:         https://github.com/spinetools/spineranknet
"""
from __future__ import annotations

__version__: str = "1.0.0"

__all__ = ["__version__"]
