"""Ablation presets for the SpineRankNet experiments.

:data:`ABLATION_PRESETS` is the registry of named hyperparameter / loss
configurations used across the paper's ablation tables, and
:func:`apply_ablation_preset` applies one preset onto a
:class:`spineranknet.config.RankingConfig`-style dataclass.

These live in :mod:`spineranknet.experiments` (rather than the core trainer
config) so that the published package surfaces only the paper configuration,
while the ablation-capable trainer
(:mod:`spineranknet.baseline.train_ranking_mse`) imports them here to honour
``--ablation`` flags and remain fully reproducible.
"""
from __future__ import annotations

from typing import Any

from spineranknet.config import ORDINAL_TASKS_13, STANDARD_TASKS_13

__all__ = ["ABLATION_PRESETS", "apply_ablation_preset"]


# ════════════════════════════════════════════════════════════════════════════
# ABLATION PRESETS
# ════════════════════════════════════════════════════════════════════════════

ABLATION_PRESETS = {
    # ── loss component ablation ──
    "clf_only": {
        "desc": "Classification loss only (no ranking, no triplet)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 0.0,
        "USE_TRIPLET": False, "TRIPLET_LOSS_WEIGHT": 0.0,
    },
    "rank_only": {
        "desc": "Ranking loss only (no classification, no triplet)",
        "CLF_LOSS_WEIGHT": 0.0, "RANKING_LOSS_WEIGHT": 1.0,
        "USE_TRIPLET": False, "TRIPLET_LOSS_WEIGHT": 0.0,
    },
    "triplet_only": {
        "desc": "Triplet loss only (no classification, no pairwise ranking)",
        "CLF_LOSS_WEIGHT": 0.0, "RANKING_LOSS_WEIGHT": 0.0,
        "USE_TRIPLET": True, "TRIPLET_LOSS_WEIGHT": 1.0,
    },
    "clf_rank": {
        "desc": "Classification + Pairwise Ranking (no triplet)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 1.0,
        "USE_TRIPLET": False, "TRIPLET_LOSS_WEIGHT": 0.0,
    },
    "clf_triplet": {
        "desc": "Classification + Triplet (no pairwise ranking)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 0.0,
        "USE_TRIPLET": True, "TRIPLET_LOSS_WEIGHT": 0.5,
    },
    "rank_triplet": {
        "desc": "Pairwise Ranking + Triplet (no classification)",
        "CLF_LOSS_WEIGHT": 0.0, "RANKING_LOSS_WEIGHT": 1.0,
        "USE_TRIPLET": True, "TRIPLET_LOSS_WEIGHT": 0.5,
    },
    "all": {
        "desc": "Classification + Ranking + Triplet (full hybrid)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 1.0,
        "USE_TRIPLET": True, "TRIPLET_LOSS_WEIGHT": 0.5,
    },
    # ── ranking loss function ablation ──
    "rank_deepsvm": {
        "desc": "Ranking with DeepRankSVM loss",
        "RANKING_LOSS": "DeepRankSVM", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0, "USE_TRIPLET": False,
    },
    "rank_ranknet": {
        "desc": "Ranking with RankNet loss",
        "RANKING_LOSS": "RankNet", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0, "USE_TRIPLET": False,
    },
    "rank_deeprelativeattributes": {
        "desc": "Ranking with DeepRelativeAttributes loss",
        "RANKING_LOSS": "DeepRelativeAttributes", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0, "USE_TRIPLET": False,
    },
    "rank_justnoticeabledifferences": {
        "desc": "Ranking with JustNoticeableDifferences loss",
        "RANKING_LOSS": "JustNoticeableDifferences", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0, "USE_TRIPLET": False,
    },
    # ── classification loss ablation ──
    "clf_ce": {
        "desc": "Classification with CE loss",
        "TASK_LOSS_OVERRIDE": {}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    "clf_emd": {
        "desc": "Classification with EMD (Earth Mover's Distance) loss",
        "TASK_LOSS_OVERRIDE": {"_all_": "emd"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    "clf_corn": {
        "desc": "Classification with CORN ordinal loss",
        "TASK_LOSS_OVERRIDE": {"_all_": "corn"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    "clf_coral": {
        "desc": "Classification with CORAL ordinal loss",
        "TASK_LOSS_OVERRIDE": {"_all_": "coral"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    "clf_softce": {
        "desc": "Classification with Soft ordinal CE loss",
        "TASK_LOSS_OVERRIDE": {"_all_": "soft_ce"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    "clf_clm": {
        "desc": "Classification with Cumulative Link Model loss",
        "TASK_LOSS_OVERRIDE": {"_all_": "clm"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    "clf_mae": {
        "desc": "Classification with MAE ordinal loss",
        "TASK_LOSS_OVERRIDE": {"_all_": "mae"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
    },
    # ── ranking head type ablation ──
    "head_mlp": {
        "desc": "MLP ranking head",
        "RANKING_HEAD_TYPE": "mlp", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "head_transformer": {
        "desc": "Transformer ranking head",
        "RANKING_HEAD_TYPE": "transformer", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "head_least_squares": {
        "desc": "Least-squares ranking head",
        "RANKING_HEAD_TYPE": "least_squares", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "head_kan": {
        "desc": "KAN (Kolmogorov-Arnold) ranking head",
        "RANKING_HEAD_TYPE": "kan", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "head_linear": {
        "desc": "Linear ranking head",
        "RANKING_HEAD_TYPE": "linear", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    # ── label mode ablation ──
    "labels_standard": {
        "desc": "Standard labels (Pfi/Nar/CCS multi-class, rest binary)",
        "USE_RAW_LABELS": False, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
        "TASKS": list(STANDARD_TASKS_13),
    },
    "labels_ordinal": {
        "desc": "Ordinal labels (all tasks multi-class)",
        "USE_RAW_LABELS": True, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
        "TASKS": list(ORDINAL_TASKS_13),
    },
    # ── classification-only with standard (binary) labels ──
    # Combined presets: clf_only behavior + STANDARD_TASKS_13
    "clf_only_standard": {
        "desc": "Classification only + standard labels (Pfi/Nar/CCS multi-class, rest binary)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 0.0,
        "USE_TRIPLET": False, "TRIPLET_LOSS_WEIGHT": 0.0,
        "USE_RAW_LABELS": False,
        "TASKS": list(STANDARD_TASKS_13),
    },
    "clf_corn_standard": {
        "desc": "CORN loss + standard labels (Pfi/Nar/CCS multi-class, rest binary)",
        "TASK_LOSS_OVERRIDE": {"_all_": "corn"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
        "TRIPLET_LOSS_WEIGHT": 0.0, "USE_RAW_LABELS": False,
        "TASKS": list(STANDARD_TASKS_13),
    },
    "clf_coral_standard": {
        "desc": "CORAL loss + standard labels (Pfi/Nar/CCS multi-class, rest binary)",
        "TASK_LOSS_OVERRIDE": {"_all_": "coral"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
        "TRIPLET_LOSS_WEIGHT": 0.0, "USE_RAW_LABELS": False,
        "TASKS": list(STANDARD_TASKS_13),
    },
    "clf_mae_standard": {
        "desc": "MAE loss + standard labels (Pfi/Nar/CCS multi-class, rest binary)",
        "TASK_LOSS_OVERRIDE": {"_all_": "mae"}, "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 0.0, "USE_TRIPLET": False,
        "TRIPLET_LOSS_WEIGHT": 0.0, "USE_RAW_LABELS": False,
        "TASKS": list(STANDARD_TASKS_13),
    },
    # ── task weighting ablation ──
    "tw_uniform": {
        "desc": "Uniform task weighting",
        "TASK_WEIGHT_STRATEGY": "uniform",
    },
    "tw_difficulty": {
        "desc": "Difficulty-based task weighting (√K)",
        "TASK_WEIGHT_STRATEGY": "difficulty",
    },
    "tw_uncertainty": {
        "desc": "Uncertainty-based task weighting (Kendall 2018)",
        "TASK_WEIGHT_STRATEGY": "uncertainty",
    },
    "tw_dwa": {
        "desc": "Dynamic Weight Averaging (Liu 2019)",
        "TASK_WEIGHT_STRATEGY": "dwa",
    },
    # ── contribution validation ablation ──
    "no_hard_mining": {
        "desc": "Random pair selection (no hard-negative mining)",
        "HARD_NEGATIVE_RATIO": 0.0,
    },
    "no_severity_weight": {
        "desc": "No severity weighting multiplier on pairwise loss",
        "USE_SEVERITY_WEIGHT": False,
    },
    "no_oversampling": {
        "desc": "No difficulty-aware oversampling",
        "OVERSAMPLE_DIFFICULT": False,
    },
    "e2e_200ep": {
        "desc": "End-to-end joint training (200 epochs, no pre-training)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 1.0,
        "USE_TRIPLET": True, "TRIPLET_LOSS_WEIGHT": 0.5,
    },
    "e2e_400ep": {
        "desc": "End-to-end joint training (400 epochs, epoch-matched with two-phase)",
        "CLF_LOSS_WEIGHT": 1.0, "RANKING_LOSS_WEIGHT": 1.0,
        "USE_TRIPLET": True, "TRIPLET_LOSS_WEIGHT": 0.5,
        "EPOCHS": 400,
    },
    # ── cross-task attention ablation ──
    "cross_task_transformer": {
        "desc": "Cross-task Transformer attention on ranking scores",
        "CROSS_TASK_MODE": "transformer", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "cross_task_mlp": {
        "desc": "Cross-task MLP on ranking scores",
        "CROSS_TASK_MODE": "mlp", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "cross_task_gated": {
        "desc": "Cross-task gated attention on ranking scores",
        "CROSS_TASK_MODE": "gated", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    "cross_task_mean_pool": {
        "desc": "Cross-task mean-pool baseline (no learned attention)",
        "CROSS_TASK_MODE": "mean_pool", "CLF_LOSS_WEIGHT": 1.0,
        "RANKING_LOSS_WEIGHT": 1.0,
    },
    # ── rank–classification agreement regularization ──
    "agree_mse_01": {
        "desc": "Agreement regularization λ=0.1 (ranking follows clf)",
        "AGREEMENT_LOSS_WEIGHT": 0.1, "AGREEMENT_DETACH_CLF": True,
    },
    "agree_mse_05": {
        "desc": "Agreement regularization λ=0.5 (ranking follows clf)",
        "AGREEMENT_LOSS_WEIGHT": 0.5, "AGREEMENT_DETACH_CLF": True,
    },
    "agree_mutual_01": {
        "desc": "Mutual agreement λ=0.1 (both branches get gradients)",
        "AGREEMENT_LOSS_WEIGHT": 0.1, "AGREEMENT_DETACH_CLF": False,
    },
    # ── rank-calibrated inference ──
    "rank_cal_s05": {
        "desc": "Rank-calibrated classification (σ=0.5, α=0.5)",
        "USE_RANK_CALIBRATION": True, "CALIBRATION_SIGMA": 0.5,
        "CALIBRATION_ALPHA": 0.5,
    },
    "rank_cal_s10": {
        "desc": "Rank-calibrated classification (σ=1.0, α=0.5)",
        "USE_RANK_CALIBRATION": True, "CALIBRATION_SIGMA": 1.0,
        "CALIBRATION_ALPHA": 0.5,
    },
    "rank_cal_a07": {
        "desc": "Rank-calibrated classification (σ=1.0, α=0.7 — clf-heavy)",
        "USE_RANK_CALIBRATION": True, "CALIBRATION_SIGMA": 1.0,
        "CALIBRATION_ALPHA": 0.7,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# apply_ablation_preset
# ════════════════════════════════════════════════════════════════════════════


def apply_ablation_preset(config: Any, preset_name: str) -> str:
    """Apply an ablation preset to the config, returns description string."""
    if preset_name not in ABLATION_PRESETS:
        available = ", ".join(sorted(ABLATION_PRESETS.keys()))
        raise ValueError(f"Unknown ablation preset: {preset_name!r}. Available: {available}")

    preset = ABLATION_PRESETS[preset_name]
    desc = preset.get("desc", preset_name)

    for key, value in preset.items():
        if key == "desc":
            continue
        # Handle special "_all_" key in TASK_LOSS_OVERRIDE
        if key == "TASK_LOSS_OVERRIDE" and isinstance(value, dict) and "_all_" in value:
            loss_name = value["_all_"]
            tasks = getattr(config, "TASKS", None) or []
            config.TASK_LOSS_OVERRIDE = {t: loss_name for t in tasks}
        else:
            setattr(config, key, value)

    return desc


# ════════════════════════════════════════════════════════════════════════════
# TEST SET FILTERING: T2w-only + one S1 per patient
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# HybridConfig
# ════════════════════════════════════════════════════════════════════════════

