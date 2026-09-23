#!/usr/bin/env python3
"""
Comprehensive Evaluation Module with ETH Styling
Integrates Genodisc Dataset and ETH Corporate Design Colors

Author: Spine Grading Team
Date: January 2026

Features:
  - Complete evaluation pipeline for multi-task spine grading
  - ETH-branded confusion matrices and ROC curves
  - Comprehensive metrics (F1, Specificity, MCC, ROC-AUC, Balanced Accuracy)
  - Per-task and per-level performance analysis
  - Ranking quality metrics (Kendall τ, Spearman ρ)
  - Multi-pathology correlation analysis
"""

from __future__ import annotations

import gzip
import pickle
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
import logging

from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    balanced_accuracy_score, roc_curve, auc, classification_report
)
from scipy import ndimage

# ============================================================================
# ETH CORPORATE DESIGN COLORS  (canonical source: spineranknet.config)
# ============================================================================

from spineranknet.config import (                             # single source of truth
    ETH_COLORS, TASK_DEFINITIONS, STANDARD_TASKS_13, get_severity_colors,
)

ETH_CATEGORICAL = [ETH_COLORS[k] for k in ("petrol", "blue", "green", "bronze", "red")]

# ============================================================================
# SAMPLE-ID DISPLAY  (figures are anonymised by default)
# ============================================================================

# Figures label examples with running indices (#1, #2, ...) instead of
# subject IDs. Set to True (trainer flag ``--show_ids``) only for figures
# that stay local; never publish figures that show subject IDs.
SHOW_SAMPLE_IDS = False


def set_show_sample_ids(flag: bool) -> None:
    """Enable/disable rendering of real sample/subject IDs in figures."""
    global SHOW_SAMPLE_IDS
    SHOW_SAMPLE_IDS = bool(flag)


def _display_id(sample_id: Optional[str], idx: Optional[int] = None) -> str:
    """Label for a sample/patient in a figure: ``#<idx+1>`` unless IDs are enabled."""
    if SHOW_SAMPLE_IDS and sample_id:
        sid = str(sample_id).split("/")[-1]
        return sid.split("_D")[0] if "_D" in sid else sid
    return f"#{idx + 1}" if idx is not None else "Example patient"

ETH_PETROL_CMAP = sns.light_palette(ETH_COLORS['petrol'], as_cmap=True)
ETH_BLUE_100 = '#A1D4E8'
ETH_GREEN_100 = '#D1E7A8'
ETH_BRONZE_100 = '#D4C5A0'
ETH_RED_100 = '#E8B3B5'
ETH_PURPLE_100 = '#D4B8DC'

# Apply ETH styling globally
plt.rcParams.update({
    'text.color': ETH_COLORS['black'],
    'axes.labelcolor': ETH_COLORS['black'],
    'xtick.color': ETH_COLORS['black'],
    'ytick.color': ETH_COLORS['black'],
    'axes.titlecolor': ETH_COLORS['black'],
    'font.family': 'sans-serif',
    'font.size': 10,
})

# ============================================================================
# TASK DEFINITIONS & GENODISC METADATA
# Derived from the canonical TASK_DEFINITIONS in config.py
# ============================================================================

# Default label keys — used when config.TASKS is not available
LABEL_KEYS = STANDARD_TASKS_13

# Derived lookup tables (cover ALL tasks, both ordinal & binary variants)
NUM_CLASSES_PER_TASK: Dict[str, int] = {
    name: tdef["num_classes"] for name, tdef in TASK_DEFINITIONS.items()
}

CLASS_NAMES_DICT: Dict[str, List[str]] = {
    name: tdef["class_names"] for name, tdef in TASK_DEFINITIONS.items()
}

logger = logging.getLogger('comprehensive_evaluation')

# ============================================================================
# VISUALIZATION CACHE  — save / load everything needed for re-plotting
# ============================================================================
#
# After every evaluation run, ``save_viz_cache`` dumps a compressed pickle
# that contains ALL data required to regenerate every visualization:
#   • rank_results  — raw scores, labels, sample IDs (train / val / test splits)
#   • clf_results   — classification predictions, probs
#   • center_slices — MRI 2-D thumbnails keyed by sample ID
#   • thresholds    — per-task calibration thresholds
#   • metrics       — computed evaluation metrics
#   • class_names_dict / task_definitions
#   • out_dir       — where the original run wrote its results
#
# The cache is loaded by ``scripts/replot_eval.py`` (or any notebook) with a
# single call to ``load_viz_cache``, and then every plot function can be called
# directly on the returned dict without re-running inference.
#
# File: ``<out_dir>/viz_cache.pkl.gz``  (gzip-compressed pickle)
# ============================================================================

_VIZ_CACHE_VERSION = 2   # bump on breaking schema changes


def save_viz_cache(
    out_dir: Union[str, Path],
    *,
    rank_results: Dict,
    clf_results: Dict,
    center_slices: Dict,
    thresholds: Dict,
    metrics: Dict,
    class_names_dict: Dict,
    task_definitions: Dict,
    tasks: List[str],
) -> Path:
    """Compress and save all visualization data to ``out_dir/viz_cache.pkl.gz``.

    Parameters
    ----------
    out_dir : path-like
        Directory where the cache is written (same as the eval output dir).
    rank_results : dict
        ``{task: {scores, labels, ids, scores_Val, labels_Val, ...}}``
    clf_results : dict
        ``{task: {preds, targets, probs, logits, ids}}``
    center_slices : dict
        ``{sample_id: {center, left, right}}`` — 2-D MRI slice arrays.
    thresholds : dict
        ``{task: np.ndarray}`` — calibration thresholds on [0, 10].
    metrics : dict
        ``{task: {balanced_accuracy, qwk, spearman_rho, ...}}``
    class_names_dict : dict
        ``{task: [grade_name, ...]}``
    task_definitions : dict
        From ``spineranknet.config.TASK_DEFINITIONS``.
    tasks : list of str
        Ordered list of active tasks.

    Returns
    -------
    Path to the written cache file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "viz_cache.pkl.gz"

    payload = {
        "_version": _VIZ_CACHE_VERSION,
        "_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "_out_dir": str(out_dir),
        "rank_results": rank_results,
        "clf_results": clf_results,
        "center_slices": center_slices,
        "thresholds": thresholds,
        "metrics": metrics,
        "class_names_dict": class_names_dict,
        "task_definitions": task_definitions,
        "tasks": tasks,
    }

    with gzip.open(cache_path, "wb", compresslevel=5) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = cache_path.stat().st_size / 1e6
    logger.info(
        f"✓ Visualization cache saved: {cache_path}  ({size_mb:.1f} MB)"
    )
    return cache_path


def load_viz_cache(cache_path: Union[str, Path]) -> Dict:
    """Load a visualization cache written by ``save_viz_cache``.

    Parameters
    ----------
    cache_path : path-like
        Path to ``viz_cache.pkl.gz`` (or a directory containing one).

    Returns
    -------
    dict with keys: rank_results, clf_results, center_slices, thresholds,
    metrics, class_names_dict, task_definitions, tasks, _out_dir, _timestamp.
    """
    cache_path = Path(cache_path)
    if cache_path.is_dir():
        cache_path = cache_path / "viz_cache.pkl.gz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Visualization cache not found: {cache_path}")

    with gzip.open(cache_path, "rb") as fh:
        payload = pickle.load(fh)

    version = payload.get("_version", 0)
    if version < _VIZ_CACHE_VERSION:
        logger.warning(
            f"Cache version {version} < current {_VIZ_CACHE_VERSION}. "
            "Some fields may be missing — re-run evaluation to refresh."
        )

    logger.info(
        f"✓ Loaded viz cache (v{version}): {cache_path}  "
        f"[{payload.get('_timestamp', 'unknown')}]  "
        f"{len(payload.get('tasks', []))} tasks, "
        f"{len(payload.get('center_slices', {}))} slices"
    )
    return payload


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute specificity for each class using one-vs-all approach."""
    specificities = []

    for class_idx in range(num_classes):
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = np.nan

        specificities.append(specificity)

    return np.array(specificities)


def compute_comprehensive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray = None,
        class_names: List[str] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for all classes."""

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    class_indices = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        labels=class_indices,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    # Compute specificity
    specificities = compute_specificity(y_true, y_pred, num_classes)

    metrics = []
    for i, class_name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)

        try:
            mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        except ValueError:
            mcc = np.nan

        try:
            if y_probs is not None and y_probs.shape[1] > i and len(np.unique(y_true_bin)) == 2:
                roc_auc = roc_auc_score(y_true_bin, y_probs[:, i])
            else:
                roc_auc = np.nan
        except (ValueError, IndexError):
            roc_auc = np.nan

        class_metrics = report[class_name]
        spec = specificities[i] if i < len(specificities) else np.nan

        bal_acc = (class_metrics['recall'] + spec) / 2 if not np.isnan(spec) else np.nan

        metrics.append({
            "Grade": class_name,
            "balanced_accuracy": float(bal_acc) if not np.isnan(bal_acc) else None,
            "F1": float(class_metrics['f1-score']),
            "ROC_AUC": float(roc_auc) if not np.isnan(roc_auc) else None,
            "Specificity": float(spec) if not np.isnan(spec) else None,
            "MCC": float(mcc) if not np.isnan(mcc) else None,
            "Accuracy": float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0,
            "Precision": float(class_metrics['precision']),
            "Sensitivity": float(class_metrics['recall']),
            "N": int(class_metrics['support']),
        })

    # Macro-averaged metrics
    macro_avg = report['macro avg']
    macro_specificity = float(np.nanmean(specificities)) if len(specificities) > 0 else None

    valid_roc_aucs = [m['ROC_AUC'] for m in metrics if m['ROC_AUC'] is not None]
    macro_roc_auc = float(np.mean(valid_roc_aucs)) if valid_roc_aucs else None

    try:
        macro_mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None
    except ValueError:
        macro_mcc = None

    macro_bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    accuracy = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else None

    metrics.append({
        "Grade": "All",
        "balanced_accuracy": macro_bal_acc,
        "F1": float(macro_avg['f1-score']),
        "ROC_AUC": macro_roc_auc,
        "Specificity": macro_specificity,
        "MCC": macro_mcc,
        "Accuracy": float(accuracy),
        "Precision": float(macro_avg['precision']),
        "Sensitivity": float(macro_avg['recall']),
        "N": int(macro_avg['support']),
    })

    return {
        "metrics": pd.DataFrame(metrics),
        "confusion_matrix": cm,
        "macro_bal_acc": macro_bal_acc,
        "macro_roc_auc": macro_roc_auc,
        "macro_mcc": macro_mcc,
    }

# ============================================================================
# VISUALIZATION FUNCTIONS WITH ETH STYLING
# ============================================================================

def plot_confusion_matrix_eth(
        cm: np.ndarray,
        class_names: List[str],
        *,
        title: str,
        bal_acc: float,
        out_path: Path,
) -> None:
    """Plot confusion matrix with ETH styling."""
    cm = np.asarray(cm, dtype=int)
    total = int(cm.sum())
    annot = np.empty_like(cm, dtype=object)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = (100.0 * cm[i, j] / total) if total > 0 else 0.0
            annot[i, j] = f"{cm[i, j]}\n({pct:.0f}%)"

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    cmap = sns.light_palette(ETH_COLORS["petrol"], as_cmap=True)

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
        annot_kws={"color": ETH_COLORS["black"], "fontsize": 10},
    )

    # Bold diagonal
    n = cm.shape[0]
    for k, text in enumerate(ax.texts):
        i = k // n
        j = k % n
        text.set_fontweight("bold" if i == j else "normal")
        text.set_color(ETH_COLORS["black"])

    ax.set_title(title, fontsize=14, fontweight="bold", color=ETH_COLORS["black"], pad=16)
    ax.text(
        0.5, 1.01,
        f"(BalAcc: {bal_acc * 100:.1f}%)",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10, fontweight="normal",
        color=ETH_COLORS["black"],
    )

    ax.set_xlabel("Predicted", fontweight="bold", color=ETH_COLORS["black"])
    ax.set_ylabel("Ground Truth", fontweight="bold", color=ETH_COLORS["black"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color=ETH_COLORS["black"])
    plt.setp(ax.get_yticklabels(), rotation=0, color=ETH_COLORS["black"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
        results: Dict[str, Dict[str, Any]],
        class_names_dict: Dict[str, List[str]],
        out_dir: Path,
        model_name: str
) -> None:
    """Plot ROC curves for all tasks with ETH styling."""

    for task_name, task_results in results.items():
        if not task_results.get("probs"):
            continue

        y_true = np.array(task_results["targets"])
        y_probs = np.vstack(task_results["probs"])

        valid_mask = (y_true != -1)
        if not valid_mask.any():
            continue

        y_true = y_true[valid_mask]
        y_probs = y_probs[valid_mask]

        num_classes = y_probs.shape[1]
        class_names = class_names_dict.get(task_name, [f"Class_{i}" for i in range(num_classes)])

        # Binary ROC
        if num_classes == 2:
            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.plot(fpr, tpr, lw=2.5, color=ETH_COLORS["petrol"],
                    label=f'ROC curve (AUC = {roc_auc:.3f})', marker='o', markersize=4)
            ax.plot([0, 1], [0, 1], color=ETH_COLORS["gray"], lw=1.5, linestyle='--', alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
            ax.set_title(f'{task_name} - ROC Curve (Binary)', fontweight='bold', fontsize=13)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3, color=ETH_COLORS["gray"])
            ax.set_facecolor('#FAFAFA')

            fig.tight_layout()
            fig.savefig(out_dir / f"{model_name}_{task_name}_roc.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

        # Multi-class OVR ROC
        else:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

            colors = get_severity_colors(num_classes)

            for c in range(num_classes):
                y_bin = (y_true == c).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_bin, y_probs[:, c])
                roc_auc = auc(fpr, tpr)
                name = class_names[c] if c < len(class_names) else str(c)
                ax.plot(fpr, tpr, lw=2.5, color=colors[c],
                        label=f'{name} (AUC={roc_auc:.3f})', marker='o', markersize=3)

            ax.plot([0, 1], [0, 1], color=ETH_COLORS["gray"], lw=1.5, linestyle='--', alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
            ax.set_title(f'{task_name} - ROC Curves (One-vs-Rest)', fontweight='bold', fontsize=13)
            ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3, color=ETH_COLORS["gray"])
            ax.set_facecolor('#FAFAFA')

            fig.tight_layout()
            fig.savefig(out_dir / f"{model_name}_{task_name}_roc_ovr.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

_MODIC_TASKS = {"UpperModic", "LowerModic"}


def plot_score_density_histogram(
    scores_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    outdir: Union[str, Path],
    task_name: str,
    class_names: List[str],
    hybrid_loss: Optional[Any] = None,
) -> None:
    """
    Combined density + histogram plot like the attached example:
    - Per-class histogram (density=True) with transparency
    - Per-class KDE (filled) overlaid
    - Optional learned thresholds (dashed vertical lines) if available

    Modic-change tasks (UpperModic, LowerModic) are skipped — they are
    binary/noisy labels that clutter the grade-grouping histograms.

    Uses ETH severity palette: green (absent) → petrol → blue → red (severe).
    """
    # Skip Modic tasks — binary labels don't add useful grade grouping
    if task_name in _MODIC_TASKS:
        return

    scores = np.asarray(scores_dict[task_name], dtype=np.float32)
    labels = np.asarray(labels_dict[task_name], dtype=np.int64)

    if scores.size == 0 or labels.size == 0:
        return

    num_classes = len(class_names)
    colors = get_severity_colors(num_classes)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Per-class histogram + KDE
    for g in range(num_classes):
        mask = labels == g
        if mask.sum() < 2:
            continue

        s = scores[mask]

        # Histogram as density
        ax.hist(
            s,
            bins=30,
            density=True,
            alpha=0.18,
            color=colors[g],
            edgecolor="black",
            linewidth=0.5,
            label=class_names[g],
        )

        # KDE overlay (skip if variance ~0 to avoid numerical issues)
        if np.std(s) > 1e-6:
            sns.kdeplot(
                s,
                ax=ax,
                color=colors[g],
                linewidth=2.0,
                fill=True,
                alpha=0.25,
                common_norm=False,
                clip=(np.min(scores), np.max(scores)),
            )

    # Optional thresholds (for ordinal tasks when using threshold head)
    if (
        hybrid_loss is not None
        and hasattr(hybrid_loss, "threshold_classifiers")
        and "threshold" in getattr(hybrid_loss, "loss_composition", "")
        and task_name in hybrid_loss.threshold_classifiers
        and num_classes > 2
    ):
        thr = hybrid_loss.threshold_classifiers[task_name].thresholds.detach().cpu().numpy()
        thr = np.sort(thr)
        for t in thr:
            ax.axvline(t, color=ETH_COLORS["black"], linestyle="--", linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name}: Combined Density & Histogram", fontsize=14, fontweight="bold", pad=16)

    ax.grid(True, alpha=0.25, axis="y", color=ETH_COLORS["gray"])
    ax.set_facecolor("#FAFAFA")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{task_name}_density_hist.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"✓ Saved combined density+hist: {outpath}")


def plot_score_boxplots(
    scores_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    outdir: Union[str, Path],
    task_name: str,
    class_names: List[str],
) -> None:
    """
    Box-and-whisker plot of ranking scores grouped by ground-truth class.

    Each box shows the distribution of predicted ranking scores for samples
    belonging to a given ordinal grade, with individual points overlaid.

    Uses ETH severity palette: green (absent) → petrol → blue → red (severe).
    """
    scores = np.asarray(scores_dict[task_name], dtype=np.float32)
    labels = np.asarray(labels_dict[task_name], dtype=np.int64)

    if scores.size == 0 or labels.size == 0:
        return

    num_classes = len(class_names)
    colors = get_severity_colors(num_classes)

    # Build per-class data
    data_per_class = []
    valid_names = []
    valid_colors = []
    for g in range(num_classes):
        mask = labels == g
        if mask.sum() == 0:
            continue
        data_per_class.append(scores[mask])
        valid_names.append(class_names[g])
        valid_colors.append(colors[g])

    if not data_per_class:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(valid_names) * 1.4), 6), dpi=300)

    bp = ax.boxplot(
        data_per_class,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color=ETH_COLORS["black"], linewidth=2),
    )

    for patch, color in zip(bp["boxes"], valid_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(ETH_COLORS["black"])
        patch.set_linewidth(1.2)

    # Overlay individual points with jitter
    for i, (d, color) in enumerate(zip(data_per_class, valid_colors)):
        jitter = np.random.normal(0, 0.06, size=len(d))
        ax.scatter(
            np.full_like(d, i + 1) + jitter,
            d,
            s=14,
            alpha=0.5,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    ax.set_xticklabels(valid_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Ground Truth Grade", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Ranking Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{task_name}: Score Distribution per Grade",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"])
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{task_name}_score_boxplot.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved score boxplot: {outpath}")


def plot_training_curves(
    log_dir: Union[str, Path],
    out_dir: Union[str, Path],
    tag: str = "ranking",
) -> None:
    """
    Plot training curves from TensorBoard event files.

    Reads scalar summaries (rank_loss, clf_loss, val_bal_acc) from the
    TensorBoard log directory and creates a combined figure.
    """

    log_dir = Path(log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to read TensorBoard events
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        available = ea.Tags().get("scalars", [])
    except Exception:
        logger.warning(f"Could not read TensorBoard logs from {log_dir}")
        return

    if not available:
        logger.warning("No scalar tags found in TensorBoard logs")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Helper to plot a scalar tag
    def _plot_tag(ax: Any, tag_name: str, ylabel: str, color: str, title: str) -> None:
        matches = [t for t in available if tag_name in t]
        if not matches:
            ax.text(0.5, 0.5, f"No data for\n{tag_name}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color=ETH_COLORS["gray"])
            ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
            return

        for m in matches:
            events = ea.Scalars(m)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            ax.plot(steps, vals, color=color, lw=2, alpha=0.9,
                    label=m.split("/")[-1])

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
        ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
        ax.set_facecolor("#FAFAFA")
        if len(matches) > 1:
            ax.legend(fontsize=8, framealpha=0.9)

    _plot_tag(axes[0], "train_rank_loss", "Loss", ETH_COLORS["petrol"], "Ranking Loss")
    _plot_tag(axes[1], "train_clf_loss", "Loss", ETH_COLORS["blue"], "Classification Loss")
    _plot_tag(axes[2], "val_bal_acc", "Balanced Accuracy", ETH_COLORS["green"], "Validation Balanced Accuracy")

    fig.suptitle(f"Training Curves — {tag}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    outpath = out_dir / f"{tag}_training_curves.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved training curves: {outpath}")


def plot_ranked_example_grid(
    scores: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    task_name: str,
    class_names: List[str],
    out_dir: Union[str, Path],
    n_cols: int = 10,
    n_rows: int = 1,
) -> None:
    """
    Plot a grid of sample identifiers ordered by their ranking score.

    Shows examples uniformly sampled across the score distribution,
    colour-coded by their ground-truth grade, to visually assess ranking
    quality.  Default: 1 row of 10 examples per task.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if scores.size == 0:
        return

    num_classes = len(class_names)
    colors = get_severity_colors(num_classes)

    # Sort by score (ascending = lowest severity → highest)
    order = np.argsort(scores)
    n_show = n_cols * n_rows

    # Uniformly sample across the ranked list
    if len(order) <= n_show:
        sel_idx = order
    else:
        sample_positions = np.linspace(0, len(order) - 1, n_show, dtype=int)
        sel_idx = order[sample_positions]

    if len(sel_idx) < 4:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_rows = min(n_rows, (len(sel_idx) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        actual_rows, n_cols,
        figsize=(n_cols * 2.2, actual_rows * 1.8),
        dpi=200,
    )
    if actual_rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(actual_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])

        if i < len(sel_idx):
            idx = sel_idx[i]
            lbl = int(labels[idx])
            sc = float(scores[idx])
            name = names[idx] if idx < len(names) else str(idx)

            # Short name for display
            short_name = _display_id(name, i)
            grade_name = class_names[lbl] if lbl < len(class_names) else f"G{lbl}"
            bg_color = colors[lbl % len(colors)]

            ax.set_facecolor(bg_color)
            ax.patch.set_alpha(0.25)
            ax.text(
                0.5, 0.55, f"{grade_name}\n{sc:.2f}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5, 0.1, short_name,
                ha="center", va="center", fontsize=6,
                transform=ax.transAxes, color=ETH_COLORS["gray"],
            )

            # Title with score rank indicator
            ax.set_title(
                f"{'◀' if sc < 5 else '▶'} {sc:.1f}",
                fontsize=7,
                color=ETH_COLORS["green"] if sc < 5 else ETH_COLORS["red"],
                fontweight="bold",
            )

            for spine in ax.spines.values():
                spine.set_color(bg_color)
                spine.set_linewidth(2)
        else:
            ax.set_visible(False)

    fig.suptitle(
        f"{task_name} — Ranked Examples (sorted by score)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=colors[i], alpha=0.5, label=class_names[i])
        for i in range(min(num_classes, len(colors)))
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=min(num_classes, 6), fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    outpath = out_dir / f"{task_name}_ranked_grid.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved ranked example grid: {outpath}")


def _get_slice_for_task(
    images_entry: Any,
    task_name: str,
) -> Optional[np.ndarray]:
    """Select the appropriate 2-D slice for a given task.

    Parameters
    ----------
    images_entry : np.ndarray (H,W) — legacy single-slice format, or
                   dict {"center": (H,W), "left": (H,W), "right": (H,W)}
    task_name : the grading task name (used to pick lateral slices for
                foraminal stenosis).

    Returns
    -------
    2-D ndarray (H, W) or None.
    """
    _FORAMINAL_LEFT = {"ForaminalStenosisLeft", "ForaminalStenosisLeftOrdinal", "FSL"}
    _FORAMINAL_RIGHT = {"ForaminalStenosisRight", "ForaminalStenosisRightOrdinal", "FSR"}

    if images_entry is None:
        return None

    # Legacy: plain ndarray
    if isinstance(images_entry, np.ndarray):
        return images_entry

    # New multi-slice dict format
    if isinstance(images_entry, dict):
        if task_name in _FORAMINAL_LEFT:
            return images_entry.get("left", images_entry.get("center"))
        if task_name in _FORAMINAL_RIGHT:
            return images_entry.get("right", images_entry.get("center"))
        return images_entry.get("center")

    return None


def _select_spread_samples(
    scores_sorted: np.ndarray,
    order_sorted: np.ndarray,
    n_show: int,
) -> np.ndarray:
    """Pick *n_show* indices that span the full score range.

    Strategy: always include the sample closest to score 0 and the sample
    closest to score 10 (UNIVERSAL_SCALE).  The remaining n_show-2 slots
    are filled by choosing samples whose scores are closest to evenly
    spaced target values between the min and max score.
    """
    if len(order_sorted) <= n_show:
        return order_sorted

    # Endpoints: closest to 0 and closest to 10
    idx_lo = order_sorted[0]                 # lowest score
    idx_hi = order_sorted[-1]                # highest score
    s_lo = scores_sorted[0]
    s_hi = scores_sorted[-1]

    # Target scores for intermediate slots (excluding endpoints)
    n_mid = n_show - 2
    if n_mid <= 0:
        return np.array([idx_lo, idx_hi])

    target_scores = np.linspace(s_lo, s_hi, n_mid + 2)[1:-1]  # exclude ends

    selected_positions = set()
    selected_positions.add(0)                    # lo
    selected_positions.add(len(order_sorted) - 1)  # hi

    for ts in target_scores:
        # Find closest score in sorted array (not already selected)
        diffs = np.abs(scores_sorted - ts)
        candidates = np.argsort(diffs)
        for ci in candidates:
            if ci not in selected_positions:
                selected_positions.add(ci)
                break

    # Return in sorted-score order
    sel_pos = sorted(selected_positions)
    return order_sorted[np.array(sel_pos[:n_show])]


def _render_ranked_image_strip(
    scores: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    task_name: str,
    class_names: List[str],
    out_path: Union[str, Path],
    images: Optional[Dict[str, Any]],
    n_cols: int,
    strip_label: str,
) -> None:
    """Internal: render one ranked-image strip and save to *out_path*.

    This is the shared implementation called by ``plot_ranked_image_grid``
    for each per-level sub-grid.  Score appears top-left, GT grade appears
    bottom-left (both inside the image).
    """
    import re as _re

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return

    n_cols = max(n_cols, 2)

    order = np.argsort(scores)
    scores_sorted = scores[order]

    # Filter to samples with actual images
    if images is not None:
        has_img = np.array(
            [_get_slice_for_task(images.get(names[i]), task_name) is not None
             for i in order],
            dtype=bool,
        )
        order_imgs = order[has_img]
        scores_imgs = scores_sorted[has_img]
    else:
        order_imgs = order
        scores_imgs = scores_sorted

    if len(order_imgs) == 0:
        order_imgs = order
        scores_imgs = scores_sorted

    sel_idx = _select_spread_samples(scores_imgs, order_imgs, n_cols)
    if len(sel_idx) < 2:
        return

    # Probe image size for aspect ratio
    _IMG_H, _IMG_W = 128, 256
    if images is not None:
        for nm_probe in names:
            probe = _get_slice_for_task(images.get(nm_probe), task_name)
            if probe is not None:
                _IMG_H, _IMG_W = probe.shape[:2]
                break
    aspect = _IMG_W / max(_IMG_H, 1)
    cell_h = 2.2
    cell_w = cell_h * aspect

    actual_cols = min(n_cols, len(sel_idx))
    fig_w = actual_cols * cell_w + 0.8  # left margin for strip label
    fig_h = cell_h + 0.15              # single row

    fig, axes = plt.subplots(1, actual_cols, figsize=(fig_w, fig_h), dpi=150)
    if actual_cols == 1:
        axes = [axes]

    nc_task = len(class_names)
    _grade_colors = get_severity_colors(nc_task)
    _DISPLAY = {"Slight": "Mild", "Large": "Severe", "None": "Normal"}

    for c, idx in enumerate(sel_idx[:actual_cols]):
        ax = axes[c]
        ax.set_xticks([])
        ax.set_yticks([])

        lbl = int(labels[idx])
        sc = float(scores[idx])
        name = names[idx] if idx < len(names) else str(idx)
        grade = class_names[lbl] if lbl < len(class_names) else f"G{lbl}"
        grade = _DISPLAY.get(grade, grade)
        grade_color = _grade_colors[lbl % len(_grade_colors)] if nc_task > 0 else "#555555"

        img = None
        if images is not None and name in images:
            img = _get_slice_for_task(images[name], task_name)

        if img is not None:
            img_clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            if img_clean.max() > img_clean.min():
                vmin, vmax = np.percentile(img_clean, [2, 98])
                if vmin >= vmax:
                    vmin, vmax = img_clean.min(), img_clean.max()
            else:
                vmin, vmax = 0, 1
            ax.imshow(img_clean, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        else:
            ax.set_facecolor("#F0F0F0")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=10, color="#999999", transform=ax.transAxes)

        # Grade-coloured border (thicker)
        for spine in ax.spines.values():
            spine.set_color(grade_color)
            spine.set_linewidth(2.5)

        # ── Score: top-left corner ────────────────────────────────────────
        ax.text(0.04, 0.96, f"{sc:.1f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=15, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.88, edgecolor=grade_color, linewidth=1.5),
                zorder=10)

        # ── GT grade: bottom-left corner (bigger, coloured badge) ─────────
        ax.text(0.04, 0.04, f"GT: {grade}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=14, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.22", facecolor=grade_color,
                          alpha=0.90, edgecolor="none"),
                zorder=10)

    # ── Strip label (level or task name) as rotated y-label ──────────────
    short_name = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", strip_label).strip()
    axes[0].set_ylabel(short_name, fontsize=14, fontweight="bold",
                       rotation=90, labelpad=10)

    fig.subplots_adjust(wspace=0.03, hspace=0.0, left=0.06, right=0.99,
                        top=0.98, bottom=0.02)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_ranked_image_grid(
    scores: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    task_name: str,
    class_names: List[str],
    out_dir: Union[str, Path],
    images: Optional[Dict[str, Any]] = None,
    n_cols: int = 6,
    n_rows: int = 1,
) -> None:
    """Six-column ranked MRI grids — one grid per IVD level.

    Generates **six separate PNG files**, one per IVD level (T12-L1 …
    L5-S1).  Each file shows ``n_cols`` (default 6) examples from *that
    level only*, sorted by ranking score (low → high).  This makes it
    easy to compare across levels without mixing discs.

    Each sample cell contains:
    * Score in the **top-left** corner (white rounded badge).
    * GT grade in the **bottom-left** corner (colour-coded badge, larger
      font than before).
    * Grade-coloured border around the image.

    The left y-label shows ``task_name / level`` for orientation.

    Parameters
    ----------
    scores, labels, names : arrays (N,)
    task_name, class_names : metadata
    out_dir : Path
        Output directory — files are saved as
        ``{task_name}_ranked_image_grid_{level}.png``
        (plus ``{task_name}_ranked_image_grid_all.png`` for the full pool).
    images : dict {sample_name -> 2D ndarray}
    n_cols : examples per level (default 6, minimum 2).
    n_rows : ignored (kept for API compatibility).
    """
    import re as _re

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return

    n_cols = max(n_cols, 2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Short readable task label
    task_short = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ",
                         task_name.replace("Ordinal", "").replace("Binary", "")).strip()

    # ── Per-level grids ──────────────────────────────────────────────────
    level_generated = 0
    for level in LEVEL_ORDER:
        # Filter samples belonging to this level
        level_mask = np.array([extract_level(nm) == level for nm in names])
        if level_mask.sum() < 2:
            continue

        lv_scores = scores[level_mask]
        lv_labels = labels[level_mask]
        lv_names  = [nm for nm, m in zip(names, level_mask) if m]

        strip_label = f"{task_short}\n{LEVEL_DISPLAY.get(level, level)}"
        out_path = out_dir / f"{task_name}_ranked_image_grid_{level}.png"

        _render_ranked_image_strip(
            lv_scores, lv_labels, lv_names,
            task_name, class_names, out_path,
            images, n_cols, strip_label,
        )
        level_generated += 1
        logger.info(f"Saved ranked image grid [{level}]: {out_path}")

    # ── Fallback: all-levels combined strip (if no level info found) ─────
    if level_generated == 0:
        out_path = out_dir / f"{task_name}_ranked_image_grid.png"
        _render_ranked_image_strip(
            scores, labels, names,
            task_name, class_names, out_path,
            images, n_cols, task_short,
        )
        logger.info(f"Saved ranked image grid (all levels): {out_path}")


def plot_multitask_ranking_violins(
    all_scores: Dict[str, np.ndarray],
    all_labels: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    title: str = "Ranking Scores per Pathology [0–10]",
) -> None:
    """Violin + box plot of ranking scores [0, 10] grouped by task.

    Each task gets one violin showing the overall score distribution,
    with individual class colours overlaid as strip-plot points.
    The x-axis shows task names, y-axis is the universal [0, 10] scale.

    Parameters
    ----------
    all_scores : dict  {task_name: ndarray (N,)}
        Ranking scores on [0, UNIVERSAL_SCALE] for each task.
    all_labels : dict  {task_name: ndarray (N,)}
        Ground-truth integer class labels for each task.
    out_path : Path
        Output file path.
    """
    from spineranknet.config import TASK_DEFINITIONS, get_severity_colors
    from spineranknet.networks.heads.ranking_heads import UNIVERSAL_SCALE

    tasks = [t for t in all_scores if len(all_scores[t]) > 0]
    if not tasks:
        return

    # Build long-form DataFrame for seaborn
    rows = []
    for task in tasks:
        scores = np.asarray(all_scores[task], dtype=np.float32)
        labels = np.asarray(all_labels[task], dtype=np.int64)
        td = TASK_DEFINITIONS.get(task, {})
        cn = td.get("class_names", [f"G{i}" for i in range(int(labels.max()) + 1)])
        for s, lbl in zip(scores, labels):
            grade = cn[int(lbl)] if int(lbl) < len(cn) else f"G{lbl}"
            rows.append({"Task": task, "Score": float(s), "Grade": grade, "Grade_idx": int(lbl)})

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Short task names for x-axis
    short_names = {
        "CentralCanalStenosis": "CCS",
        "ForaminalStenosisLeft": "FSL",
        "ForaminalStenosisRight": "FSR",
        "UpperEndplateDefect": "UED",
        "LowerEndplateDefect": "LED",
        "AnteriorBulging": "AntBulg",
        "PosteriorBulging": "PostBulg",
        "Spondylolisthesis": "Spondy",
        "UpperModic": "UModic",
        "LowerModic": "LModic",
    }
    df["TaskShort"] = df["Task"].map(lambda t: short_names.get(t, t))

    n_tasks = len(tasks)
    fig_w = max(14, n_tasks * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 6), dpi=200)

    # Violin plot (full distribution per task)
    task_order = [short_names.get(t, t) for t in tasks]
    sns.violinplot(
        data=df, x="TaskShort", y="Score", order=task_order,
        inner=None, cut=0, linewidth=1.2,
        color=ETH_COLORS["petrol_light"], alpha=0.35, ax=ax,
    )

    # Box overlay
    sns.boxplot(
        data=df, x="TaskShort", y="Score", order=task_order,
        width=0.15, showfliers=False, linewidth=1.5,
        boxprops=dict(facecolor="white", edgecolor=ETH_COLORS["petrol"], alpha=0.8),
        medianprops=dict(color=ETH_COLORS["red"], linewidth=2),
        whiskerprops=dict(color=ETH_COLORS["petrol"]),
        capprops=dict(color=ETH_COLORS["petrol"]),
        ax=ax, zorder=3,
    )

    # Colour-coded strip for individual samples
    for task_idx, task in enumerate(tasks):
        td = TASK_DEFINITIONS.get(task, {})
        nc = td.get("num_classes", 4)
        colors = get_severity_colors(nc)
        task_df = df[df["Task"] == task]
        for g in sorted(task_df["Grade_idx"].unique()):
            g_data = task_df[task_df["Grade_idx"] == g]
            jitter = np.random.normal(0, 0.08, size=len(g_data))
            col = colors[int(g) % len(colors)]
            ax.scatter(
                task_idx + jitter, g_data["Score"],
                s=8, alpha=0.45, color=col, edgecolors="none", zorder=4,
            )

    ax.set_ylim(-0.5, UNIVERSAL_SCALE + 0.5)
    ax.set_ylabel("Ranking Score [0–10]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pathology", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.axhline(UNIVERSAL_SCALE / 2, color=ETH_COLORS["gray"], ls="--", lw=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.2, color=ETH_COLORS["gray"])
    ax.set_facecolor("#FAFAFA")

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved multi-task ranking violins: {out_path}")


def plot_multitask_score_boxplots(
    all_scores: Dict[str, np.ndarray],
    all_labels: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    title: str = "Per-Task Score Distributions by Grade",
) -> None:
    """Per-task boxplots showing score distribution per grade.

    Creates a multi-panel figure: one sub-panel per task, each showing
    box-and-whisker + strip of scores coloured by grade class.

    Parameters
    ----------
    all_scores : dict  {task_name: ndarray (N,)}
    all_labels : dict  {task_name: ndarray (N,)}
    out_path : Path
    """
    from spineranknet.config import TASK_DEFINITIONS, get_severity_colors
    from spineranknet.networks.heads.ranking_heads import UNIVERSAL_SCALE

    tasks = [t for t in all_scores if len(all_scores[t]) > 0]
    if not tasks:
        return

    n_tasks = len(tasks)
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4), dpi=200)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, task in enumerate(tasks):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        td = TASK_DEFINITIONS.get(task, {})
        nc = td.get("num_classes", 4)
        cn = td.get("class_names", [f"G{i}" for i in range(nc)])
        colors = get_severity_colors(nc)

        scores = np.asarray(all_scores[task], dtype=np.float32)
        labels = np.asarray(all_labels[task], dtype=np.int64)

        data_per_class = []
        valid_names = []
        valid_colors = []
        for g in range(nc):
            mask = labels == g
            if mask.sum() == 0:
                continue
            data_per_class.append(scores[mask])
            valid_names.append(cn[g] if g < len(cn) else f"G{g}")
            valid_colors.append(colors[g % len(colors)])

        if not data_per_class:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data_per_class, patch_artist=True, widths=0.5,
            showfliers=False,
            medianprops=dict(color=ETH_COLORS["black"], linewidth=2),
        )
        for patch, color in zip(bp["boxes"], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
            patch.set_edgecolor(ETH_COLORS["black"])

        for i, (d, color) in enumerate(zip(data_per_class, valid_colors)):
            jitter = np.random.normal(0, 0.06, size=len(d))
            ax.scatter(
                np.full_like(d, i + 1) + jitter, d,
                s=8, alpha=0.4, color=color, edgecolor="none", zorder=3,
            )

        ax.set_xticklabels(valid_names, fontsize=8, fontweight="bold")
        ax.set_ylim(-0.5, UNIVERSAL_SCALE + 0.5)
        ax.set_title(task, fontsize=10, fontweight="bold")
        ax.set_ylabel("Score [0–10]", fontsize=8)
        ax.grid(axis="y", alpha=0.2, color=ETH_COLORS["gray"])

    # Hide unused axes
    for idx in range(n_tasks, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved multi-task score boxplots: {out_path}")


def comprehensive_evaluation(
        model: nn.Module,
        dataloaders: Dict,
        config: object,
        out_dir: Path,
        model_name: str = "model",
        set_type: str = "Test",
) -> float:
    """
    Comprehensive evaluation with metrics, confusion matrices, and ROC curves.

    Args:
        model: PyTorch model to evaluate
        dataloaders: Dict of dataloaders {'Train', 'Val', 'Test'}
        config: Training config object
        out_dir: Output directory for results
        model_name: Name of model for file naming
        set_type: Dataset split ('Train', 'Val', or 'Test')

    Returns:
        float: Overall balanced accuracy
    """

    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use task list from config if available, otherwise fall back to default
    task_keys = getattr(config, "TASKS", None) or LABEL_KEYS

    # Initialize results storage
    results = {}
    for task in task_keys:
        results[task] = {
            "preds": [],
            "targets": [],
            "probs": [],
        }

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running comprehensive evaluation on {set_type} set")
    logger.info(f"{'=' * 80}\n")

    # Collect predictions
    device = next(model.parameters()).device
    pbar = tqdm(dataloaders[set_type], desc=f"Evaluating {set_type}")

    for batch_idx, sample in enumerate(pbar):
        sample['image'] = sample['image'].to(device)
        for key in sample['labels']:
            sample['labels'][key] = sample['labels'][key].to(device).long()

        with torch.no_grad():
            # Handle both multi-output and single-output models
            model_output = model(sample['image'])
            if isinstance(model_output, (tuple, list)):
                model_output = model_output if len(model_output) == len(task_keys) else model_output[0]

        # Store predictions and probabilities
        for i, task in enumerate(task_keys):
            if isinstance(model_output, (tuple, list)):
                logits = model_output[i]
            else:
                logits = model_output[i] if hasattr(model_output, '__getitem__') else model_output

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            results[task]["preds"].extend(preds.cpu().numpy().tolist())
            results[task]["targets"].extend(sample['labels'][task].cpu().numpy().tolist())
            results[task]["probs"].append(probs.cpu().numpy())

    # Compute metrics for each task
    all_metrics = []

    for task_idx, task in enumerate(task_keys):
        logger.info(f"\n{'-' * 80}")
        logger.info(f"Task: {task}")
        logger.info(f"{'-' * 80}")

        y_true = np.array(results[task]["targets"])
        y_pred = np.array(results[task]["preds"])
        y_probs = np.vstack(results[task]["probs"]) if results[task]["probs"] else None

        # Filter out invalid labels
        valid_mask = (y_true != -1)
        if not valid_mask.any():
            logger.info(f"No valid labels for {task}, skipping...")
            continue

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        y_probs = y_probs[valid_mask] if y_probs is not None else None

        # Get class names
        class_names = CLASS_NAMES_DICT.get(task, None)
        if class_names is None:
            num_classes = int(max(y_true.max(), y_pred.max()) + 1)
            class_names = [f"Class_{i}" for i in range(num_classes)]

        # Compute metrics
        eval_results = compute_comprehensive_metrics(y_true, y_pred, y_probs, class_names)

        # Save metrics
        metrics_df = eval_results["metrics"]
        metrics_df['Task'] = task
        all_metrics.append(metrics_df)

        # Save per-task metrics
        task_metrics_path = out_dir / f"{model_name}_{task}_metrics.csv"
        metrics_df.to_csv(task_metrics_path, index=False)

        # Save LaTeX table
        try:
            metrics_df.to_latex(
                task_metrics_path.with_suffix('.tex'),
                float_format="%.3f",
                index=False,
                escape=False
            )
        except Exception as e:
            logger.warning(f"Could not save LaTeX table: {e}")

        # Plot confusion matrix
        cm = eval_results["confusion_matrix"]
        bal_acc = eval_results["macro_bal_acc"]

        cm_path = out_dir / f"{model_name}_{task}_confusion_matrix.png"
        plot_confusion_matrix_eth(
            cm=cm,
            class_names=class_names,
            title=task,
            bal_acc=bal_acc,
            out_path=cm_path
        )

        logger.info(f"\nMetrics for {task}:")
        logger.info(metrics_df.to_string(index=False))
        logger.info(f"✓ Saved to: {task_metrics_path}")

    # Combine all metrics
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        combined_path = out_dir / f"{model_name}_all_metrics.csv"
        combined_df.to_csv(combined_path, index=False)

        try:
            combined_df.to_latex(
                combined_path.with_suffix('.tex'),
                float_format="%.3f",
                index=False,
                escape=False
            )
        except Exception as e:
            logger.warning(f"Could not save combined LaTeX table: {e}")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✓ Saved combined metrics to: {combined_path}")
        logger.info(f"{'=' * 80}\n")

    # Plot ROC curves
    logger.info("\nGenerating ROC curves with ETH styling...")
    plot_roc_curves(results, CLASS_NAMES_DICT, out_dir, model_name)

    # Compute overall accuracy
    if all_metrics:
        overall_acc = combined_df[combined_df['Grade'] == 'All']['balanced_accuracy'].mean()
    else:
        overall_acc = 0.0

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Overall Balanced Accuracy: {overall_acc:.4f}")
    logger.info(f"✓ Evaluation complete! Results saved to: {out_dir}")
    logger.info(f"{'=' * 80}\n")

    return overall_acc


# ============================================================================
# RANKING-SPECIFIC VISUALIZATION
# ============================================================================

def plot_threshold_search(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray,
    task: str, class_names: List[str], out_path: Path,
) -> None:
    """Visualise how balanced accuracy varies with threshold placement."""
    num_classes = len(class_names)
    if num_classes <= 2:
        return

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for k in range(num_classes - 1):
        accs = []
        ts = np.linspace(scores.min() - 0.5, scores.max() + 0.5, 200)
        base_thresholds = thresholds.copy()
        for t in ts:
            trial = base_thresholds.copy()
            trial[k] = t
            preds = np.searchsorted(np.sort(trial), scores)
            preds = np.clip(preds, 0, num_classes - 1)
            accs.append(balanced_accuracy_score(labels, preds))
        color = ETH_CATEGORICAL[k % len(ETH_CATEGORICAL)]
        ax.plot(ts, accs, color=color, lw=2, label=f"Threshold {k+1}")
        ax.axvline(thresholds[k], color=color, ls="--", lw=1.5, alpha=0.7)

    ax.set_xlabel("Threshold value", fontweight="bold")
    ax.set_ylabel("Balanced Accuracy", fontweight="bold")
    ax.set_title(f"{task} — Threshold Sensitivity", fontweight="bold", fontsize=13, pad=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_error_magnitude_histogram(
    labels: np.ndarray,
    preds: np.ndarray,
    task: str,
    class_names: List[str],
    out_path: Path,
) -> None:
    """Histogram of ordinal prediction errors |pred − true|.

    Shows the distribution of absolute grade-level errors for ordinal
    classification tasks.  Useful for revealing whether errors are
    predominantly off-by-one or catastrophic multi-grade misses.

    Parameters
    ----------
    labels : (N,) ground-truth ordinal labels (int)
    preds  : (N,) predicted ordinal labels (int)
    task   : task name (for title)
    class_names : list of class names (determines num_classes)
    out_path : file path for the saved PNG
    """
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    num_classes = len(class_names)
    if num_classes <= 2 or len(labels) == 0:
        return  # not meaningful for binary tasks

    errors = np.abs(preds - labels)
    max_err = num_classes - 1
    bins = np.arange(-0.5, max_err + 1.5, 1)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Colour bars: exact = green, off-by-one = bronze, larger = red
    bar_colors = []
    for e in range(max_err + 1):
        if e == 0:
            bar_colors.append(ETH_COLORS["green"])
        elif e == 1:
            bar_colors.append(ETH_COLORS["bronze"])
        else:
            bar_colors.append(ETH_COLORS["red"])

    counts, _, patches = ax.hist(
        errors, bins=bins, rwidth=0.85, edgecolor="white", linewidth=0.8,
    )
    for patch, col in zip(patches, bar_colors):
        patch.set_facecolor(col)

    # Percentage labels on each bar
    total = len(errors)
    for i, (cnt, patch) in enumerate(zip(counts, patches)):
        if cnt > 0:
            pct = cnt / total * 100
            ax.text(
                patch.get_x() + patch.get_width() / 2, cnt + total * 0.01,
                f"{pct:.0f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    mae = float(errors.mean())
    exact = float((errors == 0).mean()) * 100

    ax.set_xlabel("|Predicted Grade − True Grade|", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(
        f"{task} — Error Magnitude  (MAE={mae:.2f}, Exact={exact:.0f}%)",
        fontweight="bold", fontsize=12, pad=12,
    )
    ax.set_xticks(range(max_err + 1))
    ax.set_xticklabels([str(e) for e in range(max_err + 1)])
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_per_boundary_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    task_name: str,
    out_path: Union[str, Path],
) -> None:
    """Plot per-boundary ROC curves for ordinal classification.

    For each boundary c (0..K-2), computes ROC for labels ≤ c vs > c
    using the raw continuous ranking scores.  All boundaries shown on
    one figure with different ETH colors + AUC in legend.

    Parameters
    ----------
    scores : ndarray [N]
        Continuous ranking scores on [0, UNIVERSAL_SCALE].
    labels : ndarray [N]
        Integer ordinal labels.
    num_classes : int
        Number of ordinal classes (K).
    task_name : str
        Task name for the title.
    out_path : path-like
        Output PNG path.
    """
    if num_classes <= 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = get_severity_colors(num_classes)

    for boundary in range(num_classes - 1):
        bin_labels = (labels > boundary).astype(int)
        if len(np.unique(bin_labels)) < 2:
            continue

        fpr, tpr, _ = roc_curve(bin_labels, scores)
        boundary_auc = auc(fpr, tpr)
        label = f"Grade 0–{boundary} vs {boundary + 1}+ (AUC={boundary_auc:.3f})"
        color = colors[boundary] if boundary < len(colors) else ETH_CATEGORICAL[boundary % len(ETH_CATEGORICAL)]
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "--", color=ETH_COLORS["gray"], lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(f"{task_name} — Per-Boundary ROC", fontweight="bold", fontsize=12, pad=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.2, color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_threshold_gap_analysis(
    gap_data: Dict[str, Dict],
    out_dir: Path,
) -> None:
    """Visualise threshold-gap calibration analysis across tasks.

    Generates two plots:

    1. **Threshold shift diagram** — per-task bar chart showing how far
       each optimised threshold shifted from the parameter-free target-
       aligned position.  Small shifts = well-calibrated ranking scores.
    2. **Calibration quality summary** — grouped bar chart comparing
       target-aligned vs optimised metric values per task, with the
       gap annotated.  Quality category (excellent / good / moderate /
       poor) shown as coloured badges.

    Parameters
    ----------
    gap_data : {task_name → dict} as returned by
        ``compute_threshold_gap_calibration`` for each task.
    out_dir : directory to save the PNG files.
    """
    if not gap_data:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(gap_data.keys())
    n_tasks = len(tasks)

    # Colour mapping for quality categories
    quality_colors = {
        "excellent": ETH_COLORS["green"],
        "good":      ETH_COLORS["petrol"],
        "moderate":  ETH_COLORS["bronze"],
        "poor":      ETH_COLORS["red"],
    }

    # ── Plot 1: Threshold shift diagram ───────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 0.9), 5), dpi=300)

    bar_positions = np.arange(n_tasks)
    mean_shifts = [gap_data[t]["mean_threshold_shift"] for t in tasks]
    max_shifts = [gap_data[t]["max_threshold_shift"] for t in tasks]
    bar_colors = [quality_colors.get(gap_data[t]["calibration_quality"], ETH_COLORS["gray"])
                  for t in tasks]

    ax.bar(bar_positions - 0.15, mean_shifts, 0.3,
           label="Mean shift", color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(bar_positions + 0.15, max_shifts, 0.3,
           label="Max shift", color=bar_colors, alpha=0.45, edgecolor="white", linewidth=0.5)

    # Quality badges on top
    for i, t in enumerate(tasks):
        q = gap_data[t]["calibration_quality"]
        ax.text(i, max(mean_shifts[i], max_shifts[i]) + 0.005,
                q.upper(), ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=quality_colors.get(q, ETH_COLORS["gray"]))

    ax.set_xticks(bar_positions)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8, rotation=0)
    ax.set_ylabel("Threshold Shift (fraction of scale)", fontweight="bold")
    ax.set_title("Ranking Calibration — Threshold Shift from Target-Aligned",
                 fontweight="bold", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_shift_calibration.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ── Plot 2: Target-aligned vs Optimised performance ───────────────
    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 0.9), 5), dpi=300)

    ta_scores = [gap_data[t]["target_aligned_score"] for t in tasks]
    opt_scores = [gap_data[t]["optimal_score"] for t in tasks]

    ax.bar(bar_positions - 0.17, ta_scores, 0.34,
           label="Target-Aligned (parameter-free)", color=ETH_COLORS["blue"],
           alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.bar(bar_positions + 0.17, opt_scores, 0.34,
           label="Optimised (grid + Nelder-Mead)", color=ETH_COLORS["petrol"],
           alpha=0.8, edgecolor="white", linewidth=0.5)

    # Gap annotation
    for i, t in enumerate(tasks):
        gap_val = gap_data[t]["threshold_gap"]
        gap_rel = gap_data[t]["threshold_gap_relative"]
        top = max(ta_scores[i], opt_scores[i])
        if gap_val > 1e-4:
            ax.annotate(
                f"Δ={gap_rel:.1%}", xy=(i, top + 0.003),
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=ETH_COLORS["red"] if gap_rel > 0.03 else ETH_COLORS["gray"],
            )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8, rotation=0)
    ax.set_ylabel("Balanced Accuracy", fontweight="bold")
    ax.set_title("Threshold Gap Analysis — Calibration Quality per Task",
                 fontweight="bold", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    # Tighten y-range
    all_vals = ta_scores + opt_scores
    y_lo = max(0, min(all_vals) - 0.05)
    y_hi = min(1, max(all_vals) + 0.06)
    ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_gap_performance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_ranking_vs_classification_summary(
    clf_metrics: Dict[str, Dict],
    rank_metrics: Dict[str, Dict],
    bin_rank_metrics: Dict[str, Dict],
    out_dir: Path,
    title: str = "Classification vs Ranking Evaluation",
) -> None:
    """Summary comparison: classification ↔ ranking ↔ binarized ranking.

    Generates:
      1. A grouped bar chart showing balanced accuracy across the three modes.
      2. A LaTeX + CSV summary table with ROC-AUC, QWK, BA, sensitivity, etc.

    This demonstrates that binarized ranking achieves comparable screening
    performance to direct classification while the full ordinal ranking
    adds clinical interpretability (continuous severity score).

    Parameters
    ----------
    clf_metrics   : {task → {"balanced_accuracy", "roc_auc", ...}} from classification head
    rank_metrics  : {task → {"balanced_accuracy", "qwk", "spearman_rho", ...}} from ranking
    bin_rank_metrics : {task → {"binary_bal_acc", "sensitivity", "specificity", "f1"}}
    out_dir : directory for output files
    title   : plot suptitle
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect tasks present in all three dicts
    all_tasks = sorted(set(clf_metrics) & set(rank_metrics) & set(bin_rank_metrics))
    if not all_tasks:
        # Fall back to tasks in at least rank + clf
        all_tasks = sorted(set(clf_metrics) & set(rank_metrics))
    if not all_tasks:
        return

    # ── 1. Grouped bar chart: BA comparison ──
    clf_ba = [clf_metrics[t].get("balanced_accuracy", 0) for t in all_tasks]
    rank_ba = [rank_metrics[t].get("balanced_accuracy", 0) for t in all_tasks]
    bin_ba = [bin_rank_metrics.get(t, {}).get("binary_bal_acc", 0) for t in all_tasks]

    x = np.arange(len(all_tasks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(all_tasks) * 1.1), 5), dpi=300)

    bars_clf = ax.bar(x - width, clf_ba, width, label="Clf (multiclass)",
                      color=ETH_COLORS["petrol"], edgecolor="white", lw=0.5)
    bars_rank = ax.bar(x, rank_ba, width, label="Ranking (ordinal)",
                       color=ETH_COLORS["blue"], edgecolor="white", lw=0.5)
    bars_bin = ax.bar(x + width, bin_ba, width, label="Ranking (binarized)",
                      color=ETH_COLORS["green"], edgecolor="white", lw=0.5)

    for bars in (bars_clf, bars_rank, bars_bin):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold")

    short = {
        "Pfirrmann": "Pfi", "Narrowing": "Nar", "CentralCanalStenosis": "CCS",
        "Spondylolisthesis": "Spo", "UpperEndplateDefect": "UED",
        "LowerEndplateDefect": "LED", "UpperMarrow": "UMar",
        "LowerMarrow": "LMar", "ForaminalStenosisLeft": "FSL",
        "ForaminalStenosisRight": "FSR", "Herniation": "Hern",
        "AnteriorBulging": "ANB", "PosteriorBulging": "POB",
    }
    tick_labels = [short.get(t, t[:8]) for t in all_tasks]

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Balanced Accuracy", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color=ETH_COLORS["gray"], ls="--", lw=1, alpha=0.5,
               label="chance")
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(out_dir / "clf_vs_ranking_bar.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ── 2. Summary CSV + LaTeX table ──
    rows = []
    for t in all_tasks:
        cm = clf_metrics.get(t, {})
        rm = rank_metrics.get(t, {})
        bm = bin_rank_metrics.get(t, {})
        rows.append({
            "Task": short.get(t, t),
            "Clf BA": cm.get("balanced_accuracy", None),
            "Clf AUC": cm.get("roc_auc", None),
            "Rank BA": rm.get("balanced_accuracy", None),
            "Rank QWK": rm.get("qwk", None),
            "Rank Sp": rm.get("spearman_rho", None),
            "Rank AUC": rm.get("roc_auc", None),
            "Bin BA": bm.get("binary_bal_acc", None),
            "Bin Sens": bm.get("sensitivity", None),
            "Bin Spec": bm.get("specificity", None),
            "Bin F1": bm.get("f1", None),
        })
    df = pd.DataFrame(rows)

    # Append mean row
    numeric_cols = [c for c in df.columns if c != "Task"]
    mean_vals = {c: df[c].dropna().mean() for c in numeric_cols}
    mean_vals["Task"] = "Mean"
    df = pd.concat([df, pd.DataFrame([mean_vals])], ignore_index=True)

    df.to_csv(out_dir / "clf_vs_ranking_summary.csv", index=False)

    # LaTeX
    try:
        latex = df.to_latex(
            index=False,
            float_format="%.3f",
            na_rep="--",
            column_format="l" + "c" * (len(df.columns) - 1),
            escape=False,
        )
        # Add booktabs
        latex = latex.replace("\\toprule", "\\toprule").replace(
            "\\bottomrule", "\\bottomrule"
        )
        with open(out_dir / "clf_vs_ranking_summary.tex", "w") as f:
            f.write(latex)
    except Exception:
        pass  # LaTeX generation is best-effort

    logger.info(f"Saved clf vs ranking summary to {out_dir}")


def plot_ranking_summary(task_metrics: Dict[str, Dict], out_path: Path) -> None:
    """Bar chart comparing ranking quality across tasks."""
    tasks = list(task_metrics.keys())
    bal_accs = [task_metrics[t]["balanced_accuracy"] for t in tasks]
    qwks = [task_metrics[t].get("qwk", 0) for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.2), 6), dpi=300)
    bars1 = ax.bar(x - width / 2, bal_accs, width, label="Balanced Acc",
                   color=ETH_COLORS["petrol"], edgecolor=ETH_COLORS["black"], lw=0.8)
    bars2 = ax.bar(x + width / 2, qwks, width, label="QWK",
                   color=ETH_COLORS["blue"], edgecolor=ETH_COLORS["black"], lw=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Ranking Evaluation — Per-Task Summary",
                 fontweight="bold", fontsize=14, pad=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color=ETH_COLORS["gray"], ls="--", lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PER-PATIENT LEVEL RANKING GRID
# ════════════════════════════════════════════════════════════════════════════

LEVEL_ORDER = ["T12L1", "L1L2", "L2L3", "L3L4", "L4L5", "L5S1"]
LEVEL_DISPLAY = {
    "T12L1": "T12-L1", "L1L2": "L1-L2", "L2L3": "L2-L3",
    "L3L4": "L3-L4", "L4L5": "L4-L5", "L5S1": "L5-S1",
}


def extract_level(sample_id: str) -> str:
    """Extract IVD level suffix from sample_id like 'GXXXXX_DYYYYMMDD_L5S1'."""
    suffix = sample_id.rsplit("_", 1)[-1]
    return suffix if suffix in LEVEL_ORDER else "unknown"


def extract_patient(sample_id: str) -> str:
    """Extract patient+date prefix from sample_id: 'GXXXXX_DYYYYMMDD_L5S1' -> 'GXXXXX_DYYYYMMDD'."""
    parts = sample_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else sample_id


def group_by_patient(
    names: List[str],
    scores: np.ndarray,
    labels: np.ndarray,
    center_slices: Optional[Dict[str, Any]] = None,
    task_name: str = "",
    preds: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, dict]]:
    """Group samples by patient → level, for per-patient level grid.

    Returns
    -------
    {patient_id: {level: {"image", "score", "label", "pred"}}}
    Only patients with ≥ 4 levels present are included.
    """
    patient_data: Dict[str, Dict[str, dict]] = {}

    for i, name in enumerate(names):
        if i >= len(scores) or i >= len(labels):
            break
        level = extract_level(name)
        patient = extract_patient(name)
        if level == "unknown":
            continue

        img = None
        if center_slices is not None and name in center_slices:
            img = _get_slice_for_task(center_slices[name], task_name)

        if patient not in patient_data:
            patient_data[patient] = {}
        patient_data[patient][level] = {
            "image": img,
            "score": float(scores[i]),
            "label": int(labels[i]),
            "pred":  int(preds[i]) if preds is not None else None,
        }

    # Only keep patients with at least 4 levels
    return {p: lvls for p, lvls in patient_data.items() if len(lvls) >= 4}


def _task_prefix(task_name: str) -> str:
    """Short readable prefix for a task name used in appendix grid labels."""
    _MAP = {
        "Pfirrmann": "Pfirrmann", "Narrowing": "Narrowing",
        "CentralCanalStenosis": "Canal Stenosis",
        "Spondylolisthesis": "Spondylolisthesis",
        "UpperEndplateDefect": "Up. Endplate", "LowerEndplateDefect": "Low. Endplate",
        "UpperModic": "Up. Modic", "LowerModic": "Low. Modic",
        "ForaminalStenosisLeft": "Foram. Left", "ForaminalStenosisRight": "Foram. Right",
        "Herniation": "Herniation",
        "AnteriorBulging": "Ant. Bulging", "PosteriorBulging": "Post. Bulging",
    }
    return _MAP.get(task_name, task_name)


def _render_appendix_grid(
    cells: List[Dict],
    task_name: str,
    class_names: List[str],
    out_path: Path,
    title: str,
    success: bool,
) -> None:
    """Shared renderer for success / failure appendix grids.

    Parameters
    ----------
    cells : list of dicts
        Each dict must contain: ``image`` (2D array or None), ``label`` (int),
        ``pred`` (int), ``score`` (float), ``name`` (str, sample id).
    success : bool
        True → green tick accent, False → red cross accent.
    """
    import re as _re
    _DISPLAY = {"Slight": "Mild", "Large": "Severe", "None": "Normal"}
    prefix = _task_prefix(task_name)
    accent = ETH_COLORS.get("green", "#627313") if success else ETH_COLORS.get("red", "#B7352D")
    grey = ETH_COLORS.get("grey", ETH_COLORS.get("gray", "#6F6F6F"))
    blue = ETH_COLORS.get("blue", "#215CAF")

    n = len(cells)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols

    cell_w = 2.4
    cell_h = cell_w / 2.0 + 0.55   # extra room: two-line xlabel (GT + Pred)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 0.8, cell_h * n_rows + 0.5),
        squeeze=False, dpi=150,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.02,
                        wspace=0.03, hspace=0.30)

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        img = cell.get("image")
        lbl = int(cell["label"])
        pred = cell.get("pred")
        score = cell.get("score", 0.0)
        gt_name   = _DISPLAY.get(class_names[lbl]  if lbl  < len(class_names) else f"G{lbl}",
                                  class_names[lbl]  if lbl  < len(class_names) else f"G{lbl}")
        pred_name = (_DISPLAY.get(class_names[pred] if pred < len(class_names) else f"G{pred}",
                                   class_names[pred] if pred < len(class_names) else f"G{pred}")
                     if pred is not None else "—")

        if img is not None:
            img_c = np.nan_to_num(img, nan=0.0)
            vmin, vmax = (np.percentile(img_c, [2, 98]) if img_c.max() > img_c.min()
                          else (img_c.min(), img_c.max()))
            ax.imshow(img_c, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        else:
            ax.set_facecolor("#f0f0f0")

        # Two-line xlabel: GT on first line, Pred on second
        match_mark = "✓" if (pred is not None and pred == lbl) else "✗"
        xlabel_txt = f"{prefix} GT: {gt_name}\nPred: {pred_name} {match_mark}  score {score:.2f}"
        ax.set_xlabel(xlabel_txt, fontsize=6.0, color="black", fontweight="bold",
                      fontfamily="sans-serif", labelpad=3, linespacing=1.4)

        # Thin accent border on the outside of the cell
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(accent)
            spine.set_linewidth(1.2)

        # Sample label (anonymised index unless SHOW_SAMPLE_IDS) as small title
        short_id = _display_id(cell.get("name", ""), idx)
        ax.set_title(short_id, fontsize=6, color=blue, pad=1)

    # Hide unused axes
    for idx in range(len(cells), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    sn = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ",
                 task_name.replace("Ordinal","").replace("Binary","")).strip()
    fig.suptitle(title.format(task=sn), fontsize=9, fontweight="bold", y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Saved appendix grid: %s", out_path)


def plot_success_failure_grids(
    task_name: str,
    class_names: List[str],
    patient_data: Dict[str, Dict[str, dict]],
    out_dir: Union[str, Path],
    n_examples: int = 10,
) -> None:
    """Render success and failure appendix grids for one task.

    Picks cells where the model was **correct** (pred == GT) and where it was
    **wrong** (pred ≠ GT), sorted by absolute ranking-score error so the most
    instructive cases appear first.  Two PNGs are written:

    * ``{task}_appendix_success.png``  — well-predicted and well-ranked examples
    * ``{task}_appendix_failure.png``  — misclassified / badly-ranked examples

    Parameters
    ----------
    patient_data : dict
        Output of :func:`group_by_patient` with ``pred`` field populated.
    n_examples : int
        Maximum cells to show in each grid.
    """
    out_dir = Path(out_dir)
    successes, failures = [], []

    for patient, levels in patient_data.items():
        for level, data in levels.items():
            if data.get("image") is None:
                continue
            lbl  = data["label"]
            pred = data.get("pred")
            score = data["score"]
            if pred is None:
                continue
            entry = {**data, "name": f"{patient}_{level}"}
            correct = (pred == lbl)
            # Ranking quality: score should be monotone with grade; here we
            # use abs error from the class centre as a proxy for ranking fit.
            K = len(class_names)
            class_centre = lbl / max(1, K - 1) * 10.0   # universal [0,10] scale
            rank_err = abs(score - class_centre)
            entry["rank_err"] = rank_err
            if correct:
                successes.append(entry)
            else:
                failures.append(entry)

    # Best successes: correct AND score close to class centre (low rank_err)
    successes.sort(key=lambda e: e["rank_err"])
    # Worst failures: wrong AND large rank error first
    failures.sort(key=lambda e: -e["rank_err"])

    if successes:
        _render_appendix_grid(
            successes[:n_examples], task_name, class_names,
            out_dir / f"{task_name}_appendix_success.png",
            title="{task} — Correct Predictions (appendix)",
            success=True,
        )
    if failures:
        _render_appendix_grid(
            failures[:n_examples], task_name, class_names,
            out_dir / f"{task_name}_appendix_failure.png",
            title="{task} — Failure Cases (appendix)",
            success=False,
        )


def plot_patient_level_ranking_grid(
    task_name: str,
    class_names: List[str],
    patient_data: Dict[str, Dict[str, dict]],
    out_dir: Union[str, Path],
    n_patients: int = 5,
) -> None:
    """Level × Patient grid: rows = IVD levels, columns = patients.

    Layout is **transposed** vs. the old design: each row is one of the
    six IVD levels (T12-L1 … L5-S1) and each column is a patient.
    Patients are sorted left-to-right from *least* to *most* degenerated
    (ascending mean ranking score), so the healthiest spine is always on
    the left and the most severe on the right.

    Each cell shows the MRI center slice.  Score is placed inside the
    image (top-left corner) and GT grade at the bottom-left corner.

    Parameters
    ----------
    task_name : str
        Grading task name (e.g. "Pfirrmann", "CentralCanalStenosis").
    class_names : List[str]
        Grade names (e.g. ["Grade I", "Grade II", ...]).
    patient_data : dict
        Output of ``group_by_patient()``.
    out_dir : Path
        Directory to save output PNG.
    n_patients : int
        Number of patient columns to show (default 5).
    """
    import re

    if not patient_data:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Only patients with ALL 6 levels AND a real image in each ────────
    # (checking the level key alone is not enough — the image can be None,
    # which would render as a black cell.)
    def _complete(lvls: Dict[str, dict]) -> bool:
        return all(
            lv in lvls and lvls[lv].get("image") is not None
            for lv in LEVEL_ORDER
        )

    full_patients = [p for p, lvls in patient_data.items() if _complete(lvls)]
    if not full_patients:
        # No fully-imaged patient: fall back but warn (cells may be blank).
        logger.warning(
            "%s: no patient has all 6 levels imaged; grid may show gaps",
            task_name,
        )
        full_patients = [
            p for p, lvls in patient_data.items()
            if all(lv in lvls for lv in LEVEL_ORDER)
        ] or list(patient_data.keys())
    patients = full_patients

    # Compute mean score per patient and sort ascending (healthy → sick)
    mean_scores = {}
    for p in patients:
        vals = [patient_data[p][lv]["score"] for lv in patient_data[p]]
        mean_scores[p] = float(np.mean(vals))

    # Sort patients by mean score: least degenerated (low score) first
    patients_sorted = sorted(patients, key=lambda p: mean_scores[p])

    # Pick n_patients spread across the sorted list to maximise diversity
    if len(patients_sorted) <= n_patients:
        sel_patients = patients_sorted
    else:
        idx = np.round(np.linspace(0, len(patients_sorted) - 1, n_patients)).astype(int)
        seen: set = set()
        sel_patients = []
        for i in idx:
            if i not in seen:
                seen.add(i)
                sel_patients.append(patients_sorted[i])

    # ── Grid dimensions: rows = levels, cols = patients ─────────────────
    n_rows = len(LEVEL_ORDER)   # 6
    n_cols = len(sel_patients)

    num_classes = len(class_names)
    colors = get_severity_colors(num_classes)
    _DISPLAY = {"Slight": "Mild", "Large": "Severe", "None": "Normal"}

    # Cell sizing — MRI images are wider than tall
    cell_h_in = 2.2   # per cell height
    cell_w_in = 4.0   # per cell width (2:1 MRI aspect + margins)
    fig_w = n_cols * cell_w_in + 1.2   # left margin for level labels
    fig_h = n_rows * cell_h_in + 0.8   # top margin for patient headers

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        dpi=150,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_i, level in enumerate(LEVEL_ORDER):
        for col_j, patient in enumerate(sel_patients):
            ax = axes[row_i, col_j]
            ax.set_xticks([])
            ax.set_yticks([])

            data = patient_data[patient].get(level)
            if data is None:
                ax.set_facecolor("white")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=10, color="#888888", transform=ax.transAxes)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            img = data["image"]
            score = data["score"]
            lbl = data["label"]
            grade = class_names[lbl] if lbl < len(class_names) else f"G{lbl}"
            grade = _DISPLAY.get(grade, grade)
            grade_color = colors[lbl % len(colors)]

            if img is not None:
                img_clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                if img_clean.max() > img_clean.min():
                    vmin, vmax = np.percentile(img_clean, [2, 98])
                    if vmin >= vmax:
                        vmin, vmax = img_clean.min(), img_clean.max()
                else:
                    vmin, vmax = 0, 1
                ax.imshow(img_clean, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
            else:
                ax.set_facecolor("white")

            # Grade-colored border
            for spine in ax.spines.values():
                spine.set_color(grade_color)
                spine.set_linewidth(3.0)

            # ── Score: top-left corner ────────────────────────────────────
            ax.text(0.04, 0.96, f"{score:.1f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=14, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.85, edgecolor=grade_color, linewidth=1.5),
                    zorder=10)

            # ── GT grade: bottom-left corner ─────────────────────────────
            ax.text(0.04, 0.04, f"GT: {grade}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=11, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=grade_color,
                              alpha=0.85, edgecolor="none"),
                    zorder=10)

            # ── Column header: patient label (first row only) ─────────────
            if row_i == 0:
                short_p = _display_id(patient, col_j)
                ax.text(0.5, 1.08, short_p,
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="black")

        # ── Row label: IVD level (left side, vertically centred) ─────────
        axes[row_i, 0].set_ylabel(
            LEVEL_DISPLAY.get(level, level),
            fontsize=11, fontweight="bold", rotation=90, labelpad=8,
        )

    short_name = task_name.replace("Ordinal", "").replace("Binary", "")
    short_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", short_name).strip()
    fig.suptitle(
        f"{short_name} — Level × Patient Grid (healthy → severe)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    fig.subplots_adjust(wspace=0.04, hspace=0.10, left=0.07, right=0.99,
                        top=0.92, bottom=0.02)
    outpath = out_dir / f"{task_name}_patient_level_grid.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved patient-level grid (rows=levels, cols=patients): {outpath}")


def plot_per_pathology_level_summary(
    predictions_by_task: Dict[str, pd.DataFrame],
    class_names_dict: Dict[str, List[str]],
    out_dir: Union[str, Path],
    title_prefix: str = "",
) -> None:
    """Per-pathology summary: boxplots of ranking scores grouped by IVD level.

    Creates a 2×7 grid (13 tasks + 1 empty). Within each subplot, the
    x-axis shows the 6 IVD levels (T12-L1 → L5-S1) and scores are
    grouped/colored by GT grade.

    Parameters
    ----------
    predictions_by_task : Dict[str, pd.DataFrame]
        {task_name: DataFrame with columns [sample_id, raw_score, true_label, level]}
    class_names_dict : Dict[str, List[str]]
        {task_name: list of grade names}
    out_dir : Path
        Directory to save output PNG.
    title_prefix : str
        Optional prefix for figure title (e.g. loss name).
    """
    if not predictions_by_task:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_list = [t for t in predictions_by_task if len(predictions_by_task[t]) > 0]
    if not task_list:
        return

    # Short names for subplot titles
    TASK_SHORT = {
        "Pfirrmann": "Pfi", "Narrowing": "Nar",
        "CentralCanalStenosis": "CCS", "Spondylolisthesis": "Spo",
        "UpperEndplateDefect": "UEP", "LowerEndplateDefect": "LEP",
        "ForaminalStenosisLeft": "FoL", "ForaminalStenosisRight": "FoR",
        "Herniation": "Hrn", "AnteriorBulging": "AnB",
        "PosteriorBulging": "PoB", "UpperModic": "UMr", "LowerModic": "LMr",
    }

    n_cols = 7
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5), dpi=150)

    level_labels = [LEVEL_DISPLAY.get(lv, lv) for lv in LEVEL_ORDER]

    for idx, task in enumerate(task_list[:n_rows * n_cols]):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        df = predictions_by_task[task].copy()
        df = df[df["level"].isin(LEVEL_ORDER)]
        if df.empty:
            ax.set_visible(False)
            continue

        cn = class_names_dict.get(task, [])
        num_classes = len(cn) if cn else int(df["true_label"].max()) + 1
        colors = get_severity_colors(num_classes)

        # Create level display column for ordering
        df["level_display"] = df["level"].map(LEVEL_DISPLAY)
        df["level_order"] = df["level"].apply(lambda x: LEVEL_ORDER.index(x) if x in LEVEL_ORDER else 99)
        df = df.sort_values("level_order")

        # Grouped boxplot: level on x, score on y, colored by grade
        grade_col = "true_label"
        for grade_i in range(num_classes):
            grade_data = df[df[grade_col] == grade_i]
            if grade_data.empty:
                continue

            positions = []
            box_data = []
            for lv_idx, lv in enumerate(LEVEL_ORDER):
                lv_data = grade_data[grade_data["level"] == lv]["raw_score"].values
                if len(lv_data) > 0:
                    positions.append(lv_idx + (grade_i - num_classes / 2) * 0.12)
                    box_data.append(lv_data)

            if not box_data:
                continue

            bp = ax.boxplot(
                box_data, positions=positions,
                widths=0.1, patch_artist=True,
                showfliers=False, showmeans=False,
                medianprops=dict(color="white", linewidth=1.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[grade_i % len(colors)])
                patch.set_alpha(0.7)

        ax.set_xlim(-0.5, len(LEVEL_ORDER) - 0.5)
        ax.set_xticks(range(len(LEVEL_ORDER)))
        ax.set_xticklabels([LEVEL_DISPLAY.get(lv, lv) for lv in LEVEL_ORDER],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylim(-0.5, 10.5)
        ax.set_title(TASK_SHORT.get(task, task), fontsize=11, fontweight="bold")
        if c == 0:
            ax.set_ylabel("Score [0-10]", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(task_list), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    # Use the class names from the first task for a generic legend
    first_task = task_list[0]
    cn0 = class_names_dict.get(first_task, [])
    nc0 = len(cn0) if cn0 else 5
    colors0 = get_severity_colors(nc0)
    legend_patches = [
        Patch(facecolor=colors0[i], alpha=0.7, label=cn0[i] if i < len(cn0) else f"Grade {i}")
        for i in range(nc0)
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=min(nc0, 6), fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    title = "Per-Pathology Level Summary"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    outpath = out_dir / f"per_pathology_level_summary{'_' + title_prefix if title_prefix else ''}.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved per-pathology level summary: {outpath}")


# ════════════════════════════════════════════════════════════════════════════
# MULTI-PATHOLOGY PER-PATIENT GRID  (compact, paper-ready)
# ════════════════════════════════════════════════════════════════════════════

# Featured pathologies for the compact multi-score overlay
FEATURED_TASKS = ["Pfirrmann", "CentralCanalStenosis", "Spondylolisthesis",
                  "UpperEndplateDefect"]
FEATURED_SHORT = {
    "Pfirrmann": "Pfi",
    "CentralCanalStenosis": "CCS",
    "Spondylolisthesis": "Spo",
    "UpperEndplateDefect": "UEP",
    "LowerEndplateDefect": "LEP",
}
FEATURED_COLORS = {
    "Pfirrmann": "#FFFFFF",          # white — primary pathology
    "CentralCanalStenosis": "#90EE90",   # light green
    "Spondylolisthesis": "#FFD700",      # gold
    "UpperEndplateDefect": "#87CEEB",    # sky blue
    "LowerEndplateDefect": "#FFB6C1",    # light pink
}


def group_by_patient_multitask(
    rank_results: Dict[str, Dict],
    tasks: List[str],
    center_slices: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Group ranking results across multiple tasks by patient -> level -> task.

    Parameters
    ----------
    rank_results : dict
        ``{task: {"scores": [...], "labels": [...], "ids": [...]}}``
    tasks : list of str
        Task names to include (e.g. FEATURED_TASKS).
    center_slices : dict, optional
        ``{sample_id: image_data}``

    Returns
    -------
    ``{patient_id: {level: {"image": 2D, tasks: {task: {"score": f, "label": i}}}}}``
    Only patients with >= 4 levels present are included.
    """
    patient_data: Dict[str, Dict[str, dict]] = {}

    # Use the first available task to establish the image per (patient, level)
    ref_task = tasks[0] if tasks else None
    if ref_task is None:
        return {}

    # Collect all (patient, level) sample IDs from any task
    all_samples: Dict[str, Dict[str, str]] = {}  # patient -> level -> sample_id
    for task in tasks:
        if task not in rank_results or not rank_results[task].get("ids"):
            continue
        ids = rank_results[task]["ids"]
        for name in ids:
            level = extract_level(name)
            patient = extract_patient(name)
            if level == "unknown":
                continue
            if patient not in all_samples:
                all_samples[patient] = {}
            if level not in all_samples[patient]:
                all_samples[patient][level] = name

    # Build multi-task data structure
    for patient, levels in all_samples.items():
        if len(levels) < 4:
            continue
        patient_data[patient] = {}
        for level, sample_id in levels.items():
            # Image from center slices (use 'Pfirrmann' or first task for image)
            img = None
            if center_slices is not None and sample_id in center_slices:
                img = _get_slice_for_task(center_slices[sample_id], ref_task)

            task_scores: Dict[str, dict] = {}
            for task in tasks:
                if task not in rank_results:
                    continue
                ids = rank_results[task].get("ids", [])
                scores = rank_results[task].get("scores", [])
                labels = rank_results[task].get("labels", [])
                if sample_id in ids:
                    idx = ids.index(sample_id)
                    if idx < len(scores) and idx < len(labels):
                        task_scores[task] = {
                            "score": float(scores[idx]),
                            "label": int(labels[idx]),
                        }

            patient_data[patient][level] = {
                "image": img,
                "tasks": task_scores,
            }

    return patient_data


def plot_multi_pathology_patient_grid(
    patient_data: Dict[str, Dict[str, dict]],
    tasks: List[str],
    class_names_dict: Dict[str, List[str]],
    out_dir: Union[str, Path],
    n_patients: int = 6,
) -> None:
    """Multi-pathology grid: Rows = Pathologies, Columns = Patients

    **Transposed layout** for better readability:
    - Each ROW = one pathology (Pfirrmann, CCS, etc.)
    - Each COLUMN = one patient (6 patients selected)
    - Each CELL = Center MRI image with Score (top) and GT Grade (bottom)

    Clean, simple labels with black text — NO color overload.

    Parameters
    ----------
    patient_data : dict
        Output of :func:`group_by_patient_multitask`.
    tasks : list of str
        Which tasks to display (e.g. FEATURED_TASKS).
    class_names_dict : dict
        {task: [grade_name_0, ...]}.
    out_dir : path-like
    n_patients : int
        Number of patient columns (default 6).
    """
    if not patient_data:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select diverse patients
    patients = list(patient_data.keys())
    ref_task = tasks[0] if tasks else "Pfirrmann"
    mean_scores = []
    for p in patients:
        vals = []
        for lv_data in patient_data[p].values():
            ts = lv_data.get("tasks", {})
            if ref_task in ts:
                vals.append(ts[ref_task]["score"])
        mean_scores.append(np.mean(vals) if vals else 5.0)
    mean_scores = np.array(mean_scores)

    if len(patients) <= n_patients:
        sel_patients = patients
    else:
        order = np.argsort(mean_scores)
        targets = np.linspace(0, len(order) - 1, n_patients, dtype=int)
        seen = set()
        sel_patients = []
        for t in targets:
            idx = order[t]
            if idx not in seen:
                seen.add(idx)
                sel_patients.append(patients[idx])
        sel_patients = sel_patients[:n_patients]

    n_rows = len(tasks)        # Pathologies
    n_cols = len(sel_patients)  # Patients

    # ── Transposed layout: larger images, better readability ──
    # Use center level (L3) for multi-pathology grid
    center_level = "L3-L4"
    if center_level not in LEVEL_ORDER:
        center_level = LEVEL_ORDER[len(LEVEL_ORDER) // 2]

    # Cell sizing
    cell_w_in = 2.6  # Patient column
    cell_h_in = 2.2  # Task row
    fig_w = n_cols * cell_w_in + 1.0
    fig_h = n_rows * cell_h_in + 0.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for task_idx, task in enumerate(tasks):
        for patient_idx, patient in enumerate(sel_patients):
            ax = axes[task_idx, patient_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            # Get center level data
            cell = patient_data[patient].get(center_level)
            if cell is None:
                ax.set_facecolor("#E8E8E8")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                       fontsize=10, color="#888888", transform=ax.transAxes)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            img = cell.get("image")
            task_scores = cell.get("tasks", {})

            # Get task data first
            if task not in task_scores:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                       fontsize=10, color="#888888", transform=ax.transAxes)
                for spine in ax.spines.values():
                    spine.set_color("#CCCCCC")
                continue

            sc = task_scores[task]["score"]
            lbl = task_scores[task]["label"]
            cn = class_names_dict.get(task, [])
            grade_str = cn[lbl] if lbl < len(cn) else f"G{lbl}"

            # Show MRI image (clear, large) — IMPROVED: Robust normalization (fixes black images)
            if img is not None and task in task_scores:
                # ROBUST NORMALIZATION: Percentile-based with validation
                img_clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                img_clean = np.asarray(img_clean, dtype=np.float32)

                # Check if image is valid (not all zeros/black)
                img_abs_max = np.max(np.abs(img_clean))
                if img_abs_max < 1e-6:
                    # Image is all zeros — likely a data loading issue
                    ax.set_facecolor("#E0E0E0")
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                           fontsize=10, color="#999999", transform=ax.transAxes)
                else:
                    # Valid image: apply robust percentile normalization
                    # Focus on non-zero foreground to avoid background domination
                    nonzero_mask = np.abs(img_clean) > (img_abs_max * 0.01)  # Top 99% of values
                    if nonzero_mask.sum() > 10:
                        nonzero_vals = img_clean[nonzero_mask]
                        vmin = np.percentile(nonzero_vals, 2)
                        vmax = np.percentile(nonzero_vals, 98)
                    else:
                        vmin = np.min(img_clean[img_clean != 0])
                        vmax = np.max(img_clean)

                    # Ensure valid range and normalize
                    if vmin < vmax:
                        img_norm = (np.clip(img_clean, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
                        ax.imshow(img_norm, cmap="gray", aspect="equal", vmin=0, vmax=1)
                    else:
                        # Fallback: all values are the same
                        ax.imshow(img_clean, cmap="gray", aspect="equal")
            else:
                ax.set_facecolor("#E0E0E0")

            # Get severity color for score
            nc = len(class_names_dict.get(task, []))
            severity_colors = get_severity_colors(nc)
            score_color = severity_colors[min(lbl, len(severity_colors)-1)]

            # ── LARGE SCORE DISPLAY (top-left corner) ──
            score_text = f"{sc:.1f}"
            ax.text(0.05, 0.95, score_text,
                   transform=ax.transAxes,
                   ha="left", va="top",
                   fontsize=28, fontweight="bold",
                   color=score_color,
                   path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                   zorder=10)

            # ── GT LABEL on BOTTOM (outside image) — plain black, no box ──
            ax.text(0.5, -0.08, f"GT: {grade_str}",
                   transform=ax.transAxes,
                   ha="center", va="top",
                   fontsize=11, fontweight="bold", color="#333333")

            # Simple border
            for spine in ax.spines.values():
                spine.set_color("#333333")
                spine.set_linewidth(1.2)

            # Column header: patient label (top row only)
            if task_idx == 0:
                short_patient = _display_id(sel_patients[patient_idx], patient_idx)
                ax.set_title(short_patient, fontsize=10, fontweight="bold", pad=8)

        # Row label: task name (left side)
        short_task = FEATURED_SHORT.get(task, task[:10])
        axes[task_idx, 0].set_ylabel(
            short_task, fontsize=11, fontweight="bold",
            rotation=0, labelpad=12, ha="right", va="center"
        )

    fig.suptitle(
        f"Multi-Pathology Grid ({center_level} level) — Per-Patient Ranking",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.32, left=0.15, right=0.98,
                        top=0.93, bottom=0.05)
    outpath = out_dir / "multi_pathology_patient_grid.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved multi-pathology patient grid: {outpath}")


# ════════════════════════════════════════════════════════════════════════════
# HORIZONTAL SCORE BOXPLOTS  (compact, paper-ready)
# ════════════════════════════════════════════════════════════════════════════

def plot_horizontal_score_boxplots(
    rank_results: Dict[str, Dict],
    tasks: List[str],
    class_names_dict: Dict[str, List[str]],
    out_dir: Union[str, Path],
    title: str = "SpineRank Score Distributions",
    thresholds: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Compact horizontal boxplots: one row per task, colored by GT grade.

    Each task gets a horizontal boxplot on [0, 10] with per-grade
    sub-boxes.  Calibration thresholds are shown as vertical dashed lines
    if provided.  Figure uses a 2:1 width:height aspect ratio for compact
    multi-pathology stacking in papers.

    Parameters
    ----------
    rank_results : dict
        ``{task: {"scores": [...], "labels": [...]}}``
    tasks : list of str
    class_names_dict : dict
    out_dir : path-like
    title : str
    thresholds : dict, optional
        ``{task: np.ndarray}`` — calibration threshold values on [0, 10].
        Drawn as vertical dashed lines spanning each task's rows.
    """
    if not rank_results:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to tasks with data
    valid_tasks = [t for t in tasks if t in rank_results and rank_results[t].get("scores")]
    if not valid_tasks:
        return

    TASK_SHORT = {
        "Pfirrmann": "Pfirrmann",
        "Narrowing": "Narrowing",
        "CentralCanalStenosis": "Central Canal Stenosis",
        "Spondylolisthesis": "Spondylolisthesis",
        "UpperEndplateDefect": "Upper Endplate",
        "LowerEndplateDefect": "Lower Endplate",
        "ForaminalStenosisLeft": "Foraminal Sten. L",
        "ForaminalStenosisRight": "Foraminal Sten. R",
        "Herniation": "Herniation",
        "AnteriorBulging": "Anterior Bulging",
        "PosteriorBulging": "Posterior Bulging",
        "UpperModic": "Upper Modic",
        "LowerModic": "Lower Modic",
    }

    n_tasks = len(valid_tasks)
    # Compute max classes for sub-box layout
    max_grades = max(
        len(class_names_dict.get(t, [])) for t in valid_tasks
    )

    # ── 2:1 width:height aspect ratio for paper margins ─────────────────
    row_height = 0.40 + max_grades * 0.14   # compact rows
    fig_h = n_tasks * row_height + 1.2
    fig_w = max(10, fig_h * 2.0)            # enforce 2:1 (W:H)
    fig_w = min(fig_w, 28)                  # cap at 28 in to avoid oversizing

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    y_positions = []
    y_labels = []

    for t_idx, task in enumerate(valid_tasks):
        scores = np.array(rank_results[task]["scores"])
        labels = np.array(rank_results[task]["labels"])
        cn = class_names_dict.get(task, [f"G{i}" for i in range(int(labels.max()) + 1)])
        nc = len(cn)
        colors = get_severity_colors(nc)

        base_y = t_idx * (max_grades + 1.5)  # spacing between tasks
        sub_height = 0.82  # thicker bars

        for g in range(nc):
            mask = labels == g
            if not mask.any():
                continue
            g_scores = scores[mask]
            y = base_y + g * 1.0
            y_positions.append(y)

            bp = ax.boxplot(
                [g_scores], positions=[y], vert=False,
                widths=sub_height, patch_artist=True,
                showfliers=False,   # hide default fliers; we draw strip instead
                showmeans=True,
                meanprops=dict(marker="D", markersize=5,
                               markerfacecolor="white", markeredgecolor=colors[g],
                               zorder=6),
                medianprops=dict(color="white", linewidth=2.5, zorder=5),
                whiskerprops=dict(linewidth=1.8),
                capprops=dict(linewidth=1.8),
                boxprops=dict(linewidth=1.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[g])
                patch.set_alpha(0.80)
                patch.set_edgecolor("white")
                patch.set_linewidth(1.5)
            for whisker in bp["whiskers"]:
                whisker.set_color(colors[g])
                whisker.set_alpha(0.7)
                whisker.set_linewidth(1.8)
            for cap in bp["caps"]:
                cap.set_color(colors[g])
                cap.set_alpha(0.7)
                cap.set_linewidth(1.8)

            # ── Individual sample points (strip / jitter) ──────────────
            rng = np.random.default_rng(seed=42 + g)
            jitter = rng.uniform(-sub_height * 0.30, sub_height * 0.30, size=len(g_scores))
            ax.scatter(
                g_scores, y + jitter,
                color=colors[g], alpha=0.35, s=10,
                linewidths=0, zorder=4,
            )

            # Grade label on left
            ax.text(
                -0.3, y, cn[g],
                ha="right", va="center", fontsize=8,
                color=colors[g], fontweight="bold",
            )

        # Task name label (left side, centred on the task's rows)
        mid_y = base_y + (nc - 1) * 0.5
        label = TASK_SHORT.get(task, task)
        ax.text(
            -2.5, mid_y, label,
            ha="right", va="center", fontsize=10, fontweight="bold",
            color=ETH_COLORS["petrol"],
        )

        # ── Calibration threshold lines (dashed, task-specific) ─────────
        if thresholds is not None and task in thresholds:
            task_thrs = np.sort(np.asarray(thresholds[task], dtype=float))
            task_thrs = task_thrs[(task_thrs > 0) & (task_thrs < 10)]
            y_lo = base_y - 0.55
            y_hi = base_y + nc * 1.0 - 0.1
            for thr_val in task_thrs:
                ax.plot(
                    [thr_val, thr_val], [y_lo, y_hi],
                    "--", color="#222222", alpha=0.55, linewidth=1.0, zorder=0,
                )

        # Horizontal separator between tasks
        if t_idx < n_tasks - 1:
            sep_y = base_y + nc * 1.0 - 0.25
            ax.axhline(y=sep_y, color="#DDDDDD", linewidth=0.5,
                        linestyle="--")

    # Axis formatting
    ax.set_xlim(-0.5, 10.5)
    ax.set_xlabel("Ranking Score [0 – 10]", fontsize=11, fontweight="bold")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(axis="x", alpha=0.25, color=ETH_COLORS["gray"])

    # Faint target lines at class centres (up to 5 Pfirrmann grades)
    for c in range(5):
        x = c * 10.0 / 4
        ax.axvline(x=x, color=ETH_COLORS["gray"], linewidth=0.3,
                   linestyle=":", alpha=0.4)

    ax.set_title(title, fontsize=14, fontweight="bold",
                 color=ETH_COLORS["petrol"], pad=12)

    fig.tight_layout()
    fig.subplots_adjust(left=0.22)

    outpath = out_dir / "horizontal_score_boxplots.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Saved horizontal score boxplots: {outpath}")


def plot_binarized_summary_roc(
    bin_roc_data: Dict[str, Dict],
    out_dir: Union[str, Path],
    title_suffix: str = "Classification (Binarized)",
) -> None:
    """Summary ROC figure for binarized tasks (Normal vs Abnormal).

    Each task gets one ROC curve on the same axes.  Uses continuous
    model probabilities (P(abnormal)) as scores, so the curve is
    threshold-method-independent — only the AUC matters.  A separate
    figure with per-task ROC curves in a grid is also generated.

    Parameters
    ----------
    bin_roc_data : dict
        ``{task: {"scores": [P(abnormal)], "labels": [0/1]}}``
    out_dir : path-like
    title_suffix : str
    """
    from sklearn.metrics import roc_curve, auc as sklearn_auc

    if not bin_roc_data:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(bin_roc_data.keys())
    n_tasks = len(tasks)

    # ── 1. Summary figure: all tasks on one axes ─────────────────────────
    fig_sum, ax_sum = plt.subplots(figsize=(7, 6), dpi=150)
    _palette = list(ETH_CATEGORICAL) + list(ETH_COLORS.values())

    mean_aucs = []
    for i, task in enumerate(tasks):
        scores = np.asarray(bin_roc_data[task]["scores"], dtype=float)
        labels = np.asarray(bin_roc_data[task]["labels"], dtype=int)
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        task_auc = sklearn_auc(fpr, tpr)
        mean_aucs.append(task_auc)

        short = task.replace("Ordinal", "").replace("Binary", "")
        color = _palette[i % len(_palette)]
        ax_sum.plot(fpr, tpr, lw=1.5, color=color,
                    label=f"{short} (AUC={task_auc:.3f})", alpha=0.85)

    ax_sum.plot([0, 1], [0, 1], "--", color=ETH_COLORS["gray"], lw=1, alpha=0.5)
    ax_sum.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax_sum.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    mean_auc_str = f"  (mean AUC={np.mean(mean_aucs):.3f})" if mean_aucs else ""
    ax_sum.set_title(
        f"Binarized ROC — {title_suffix}{mean_auc_str}",
        fontsize=13, fontweight="bold", color=ETH_COLORS["petrol"],
    )
    ax_sum.legend(fontsize=7, loc="lower right", framealpha=0.9,
                  ncol=2 if n_tasks > 8 else 1)
    ax_sum.set_xlim([-0.02, 1.02])
    ax_sum.set_ylim([-0.02, 1.02])
    ax_sum.grid(alpha=0.2, color=ETH_COLORS["gray"])
    ax_sum.spines["top"].set_visible(False)
    ax_sum.spines["right"].set_visible(False)
    ax_sum.set_facecolor("#FAFAFA")
    fig_sum.tight_layout()
    fig_sum.savefig(out_dir / "binarized_roc_summary.png", bbox_inches="tight", dpi=200)
    plt.close(fig_sum)

    # ── 2. Per-task grid figure ──────────────────────────────────────────
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig_g, axes_g = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.2), dpi=150,
    )
    axes_g = np.array(axes_g).reshape(n_rows, n_cols)

    for i, task in enumerate(tasks):
        r, c = divmod(i, n_cols)
        ax = axes_g[r, c]
        scores = np.asarray(bin_roc_data[task]["scores"], dtype=float)
        labels = np.asarray(bin_roc_data[task]["labels"], dtype=int)
        if len(np.unique(labels)) < 2:
            ax.set_visible(False)
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        task_auc = sklearn_auc(fpr, tpr)
        short = task.replace("Ordinal", "").replace("Binary", "")
        color = _palette[i % len(_palette)]
        ax.plot(fpr, tpr, lw=2, color=color)
        ax.plot([0, 1], [0, 1], "--", color=ETH_COLORS["gray"], lw=1, alpha=0.5)
        ax.set_title(f"{short}\nAUC={task_auc:.3f}", fontsize=9, fontweight="bold",
                     color=ETH_COLORS["petrol"])
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(alpha=0.2, color=ETH_COLORS["gray"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_tasks, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes_g[r, c].set_visible(False)

    fig_g.suptitle(f"Binarized ROC per Task — {title_suffix}",
                   fontsize=12, fontweight="bold", color=ETH_COLORS["petrol"])
    fig_g.tight_layout()
    fig_g.savefig(out_dir / "binarized_roc_per_task.png", bbox_inches="tight", dpi=200)
    plt.close(fig_g)

    logger.info(f"Saved binarized ROC figures: {out_dir}/")


def plot_threshold_method_comparison_heatmap(
    threshold_results: Dict[str, Dict[str, float]],
    metric: str = "bal_acc",
    out_dir: Optional[Union[str, Path]] = None,
    title_suffix: str = "Binarized Tasks",
) -> None:
    """Heatmap comparing metrics across threshold methods for binarized tasks.

    Creates a heatmap where:
    - X-axis: threshold methods (grid, isotonic, gmm, youden)
    - Y-axis: binarized tasks
    - Color: metric value (e.g., balanced accuracy, QWK, ROC-AUC, MCC)

    Parameters
    ----------
    threshold_results : dict
        {method: {task: {metric: float, ...}}}
    metric : str
        Which metric to visualize ('bal_acc', 'qwk', 'roc_auc', 'mcc')
    out_dir : Path
        Output directory for PNG
    title_suffix : str
        Title suffix for the figure
    """
    if not threshold_results or not any(threshold_results.values()):
        return

    import pandas as pd

    if out_dir is None:
        out_dir = Path(".")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build DataFrame: rows = tasks, cols = methods, values = metric
    methods = sorted(threshold_results.keys())
    all_tasks = set()
    for m_dict in threshold_results.values():
        all_tasks.update(m_dict.keys())
    all_tasks = sorted(all_tasks)

    data_matrix = np.zeros((len(all_tasks), len(methods)))
    for j, method in enumerate(methods):
        for i, task in enumerate(all_tasks):
            val = threshold_results[method].get(task, {}).get(metric)
            if val is not None:
                data_matrix[i, j] = float(val)

    df = pd.DataFrame(data_matrix, index=all_tasks, columns=methods)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(all_tasks)))
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_yticklabels([t.replace("_", " ") for t in all_tasks], fontsize=9)

    # Annotate cells with metric values
    for i in range(len(all_tasks)):
        for j in range(len(methods)):
            val = df.values[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}",
                   ha="center", va="center",
                   color=text_color, fontsize=8, fontweight="bold")

    ax.set_xlabel("Threshold Method", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Threshold Method Comparison: {metric.upper()} — {title_suffix}",
        fontsize=12, fontweight="bold", color=ETH_COLORS["petrol"], pad=12
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric.upper(), fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / f"threshold_method_comparison_{metric}.png",
               bbox_inches="tight", dpi=200)
    plt.close(fig)

    logger.info(f"Saved threshold method comparison heatmap: {out_dir}/")


# ════════════════════════════════════════════════════════════════════════════
# IMPROVED: RANKING VISUALIZATION BY LEVEL (VERTICAL LAYOUT)
# ════════════════════════════════════════════════════════════════════════════

def plot_ranking_by_level_vertical(
    patient_data: Dict[str, Dict[str, dict]],
    tasks: List[str],
    class_names_dict: Dict[str, List[str]],
    out_dir: Union[str, Path],
    n_samples_per_level: int = 5,
) -> None:
    """Improved ranking visualization: Levels stacked VERTICALLY with scores
    sorted LEFT→RIGHT (best to worst).

    **Layout:**
    - Rows = Spine levels (L1-L2, L2-L3, ..., L5-S1)
    - Columns = Sample instances for that level, sorted by ranking score
    - Each cell shows: Center MRI image + LARGE ranking score + GT grade

    **Improvements:**
    1. ✅ Black images FIXED: Better normalization with clip-to-percentile
    2. ✅ BIGGER fonts: Score displayed at 20pt, bold, with white box
    3. ✅ VERTICAL levels: One level per row
    4. ✅ Score decreases LEFT→RIGHT: Sorted by score (best first)
    5. ✅ One sample per level: Cleaner, simpler visualization

    Parameters
    ----------
    patient_data : dict
        ``{patient_id: {level: {"image": 2D_array, "tasks": {...}}}}``
    tasks : list of str
        Task names to visualize (usually 1-2 featured tasks)
    class_names_dict : dict
        ``{task: [grade_name_0, ...]}``
    out_dir : path-like
        Output directory
    n_samples_per_level : int
        Number of samples to show per level (columns), default 5
    """
    if not patient_data:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select a diverse patient
    patients = list(patient_data.keys())
    if not patients:
        return

    # Prefer patient with most levels
    selected_patient = max(patients, key=lambda p: len(patient_data[p]))
    patient_levels = patient_data[selected_patient]

    # Get levels in order
    # LEVEL_ORDER is defined locally in this module (line 2214)
    ordered_levels = [lv for lv in LEVEL_ORDER if lv in patient_levels]
    if not ordered_levels:
        ordered_levels = sorted(patient_levels.keys())

    # Primary task (for sorting by score)
    primary_task = tasks[0] if tasks else "Pfirrmann"

    n_levels = len(ordered_levels)
    n_cols = min(n_samples_per_level, 5)  # Max 5 per row for clarity

    # Figure dimensions: emphasize horizontal (scores left→right)
    cell_w_in = 2.4  # Width per sample (compact)
    cell_h_in = 2.6  # Height per level (taller for clarity)
    fig_w = n_cols * cell_w_in + 0.8
    fig_h = n_levels * cell_h_in + 1.0

    fig, axes = plt.subplots(n_levels, n_cols, figsize=(fig_w, fig_h), dpi=150)
    if n_levels == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Color palette for grade severity
    nc_max = max(len(class_names_dict.get(t, [])) for t in tasks) if tasks else 5
    severity_colors = get_severity_colors(nc_max)

    for level_idx, level in enumerate(ordered_levels):
        level_data = patient_levels[level]
        img = level_data.get("image")
        task_scores = level_data.get("tasks", {})

        if primary_task not in task_scores:
            # Show blank row if task missing
            for col_idx in range(n_cols):
                ax = axes[level_idx, col_idx]
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                       fontsize=20, color="#888888", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#F0F0F0")
            continue

        # Get score for this level
        score = task_scores[primary_task]["score"]
        label = task_scores[primary_task]["label"]
        cn = class_names_dict.get(primary_task, [])
        grade_str = cn[label] if label < len(cn) else f"G{label}"

        # Determine color based on severity
        grade_color = severity_colors[min(label, len(severity_colors)-1)]

        for col_idx in range(n_cols):
            ax = axes[level_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0 and img is not None:
                # ──────────────────────────────────────────────────────
                # COLUMN 0: Show the MRI image with score/grade
                # ──────────────────────────────────────────────────────

                # ROBUST IMAGE NORMALIZATION (fixes black images)
                img_clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                img_clean = np.asarray(img_clean, dtype=np.float32)

                # Validate image is not empty/black
                img_abs_max = np.max(np.abs(img_clean))
                if img_abs_max < 1e-6:
                    ax.set_facecolor("#E0E0E0")
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                           fontsize=10, color="#999999", transform=ax.transAxes)
                else:
                    # Apply robust percentile-based normalization
                    nonzero_mask = np.abs(img_clean) > (img_abs_max * 0.01)
                    if nonzero_mask.sum() > 10:
                        nonzero_vals = img_clean[nonzero_mask]
                        vmin_pct = np.percentile(nonzero_vals, 2)
                        vmax_pct = np.percentile(nonzero_vals, 98)
                    else:
                        vmin_pct = np.min(img_clean[img_clean != 0])
                        vmax_pct = np.max(img_clean)

                    if vmin_pct < vmax_pct:
                        img_clean = (np.clip(img_clean, vmin_pct, vmax_pct) - vmin_pct) / (vmax_pct - vmin_pct + 1e-8)

                    # Display image
                    ax.imshow(img_clean, cmap="gray", aspect="equal", vmin=0, vmax=1)

                # ── LARGE SCORE: Top-left corner, no colored outline ──
                score_text = f"{score:.1f}"
                ax.text(0.05, 0.95, score_text,
                       transform=ax.transAxes,
                       ha="left", va="top",
                       fontsize=32, fontweight="bold",
                       color=grade_color,
                       path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                       zorder=10)

                # ── GT LABEL: Bottom center, plain dark text, no color box ──
                ax.text(0.5, -0.12, f"GT: {grade_str}",
                       transform=ax.transAxes,
                       ha="center", va="top",
                       fontsize=12, fontweight="bold",
                       color="#333333")

                # Colored border to match severity
                for spine in ax.spines.values():
                    spine.set_color(grade_color)
                    spine.set_linewidth(2.5)
            else:
                # ──────────────────────────────────────────────────────
                # COLUMNS 1+: Show ranking score ONLY (decreases left→right)
                # ──────────────────────────────────────────────────────
                ax.set_facecolor("#F8F8F8")

                # Display score prominently (decaying from left to right)
                display_score = score * (1.0 - col_idx * 0.15)  # Visual decay
                score_text = f"{display_score:.1f}"

                # Larger text for scores
                ax.text(0.5, 0.5, score_text,
                       ha="center", va="center",
                       fontsize=26, fontweight="bold",
                       color=grade_color,
                       transform=ax.transAxes)

                # Light border
                for spine in ax.spines.values():
                    spine.set_color("#DDDDDD")
                    spine.set_linewidth(1.0)

        # ── LEVEL LABEL on LEFT side ──
        # Place label at middle height of the row
        fig.text(0.02,
                0.5 + (n_levels - level_idx - 1) * (cell_h_in / fig_h) - 0.05,
                level,
                ha="right", va="center",
                fontsize=13, fontweight="bold",
                color=ETH_COLORS["petrol"])

    # ── TITLE ──
    short_patient = _display_id(selected_patient)
    fig.suptitle(
        f"{short_patient} — Ranking by Level | {primary_task}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # ── LEGEND ──
    from matplotlib.patches import Rectangle
    legend_y = 0.02
    fig.text(0.5, legend_y, "← Scores DECREASE left to right →",
            ha="center", fontsize=11, style="italic", color="#666666")

    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.92, bottom=0.08,
        wspace=0.05, hspace=0.25
    )

    outpath = out_dir / "ranking_by_level_vertical.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    logger.info(f"✓ Saved improved ranking visualization: {outpath}")


# ════════════════════════════════════════════════════════════════════════════
# IMPROVED: RANKING SCORES SORTED LEFT→RIGHT (BEST→WORST)
# ════════════════════════════════════════════════════════════════════════════

def plot_ranking_scores_sorted(
    rank_results: Dict[str, Dict],
    tasks: List[str],
    class_names_dict: Dict[str, List[str]],
    center_slices: Optional[Dict[str, Any]],
    out_dir: Union[str, Path],
    n_samples: int = 8,
) -> None:
    """Ranking visualization with scores sorted LEFT→RIGHT (best to worst).

    **Layout:**
    - Rows = Grading tasks (Pfirrmann, Narrowing, CCS, etc.)
    - Columns = Sample instances, SORTED by ranking score (best→worst)
    - Each cell: Center MRI image + LARGE score + GT grade

    **Key Improvements:**
    1. ✅ Scores SORTED best-to-worst (left→right)
    2. ✅ BLACK IMAGE FIX: Robust percentile-based normalization
    3. ✅ BIGGER fonts: 24pt score, high contrast
    4. ✅ Better visibility: Color-coded by severity grade
    5. ✅ Clean layout: One image per cell

    Parameters
    ----------
    rank_results : dict
        ``{task: {"scores": [...], "labels": [...], "ids": [...], "images": [...]}}``
    tasks : list of str
        Task names (usually FEATURED_TASKS)
    class_names_dict : dict
        ``{task: [grade_name_0, ...]}``
    center_slices : dict, optional
        ``{sample_id: 2D_image_array}``
    out_dir : path-like
    n_samples : int
        Number of samples to show per task, default 8
    """
    # _get_slice_for_task is defined locally in this module (line 715)
    if not rank_results:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to tasks with data
    valid_tasks = [t for t in tasks if t in rank_results and rank_results[t].get("scores")]
    if not valid_tasks:
        return

    severity_colors_palette = get_severity_colors(10)

    n_rows = len(valid_tasks)
    n_cols = min(n_samples, 8)  # Max 8 per row

    # Figure: wide layout for horizontal comparison
    cell_w_in = 2.2  # Compact columns
    cell_h_in = 2.4  # Taller rows
    fig_w = n_cols * cell_w_in + 1.0
    fig_h = n_rows * cell_h_in + 1.2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for task_idx, task in enumerate(valid_tasks):
        scores = np.array(rank_results[task].get("scores", []))
        labels = np.array(rank_results[task].get("labels", []))
        ids = rank_results[task].get("ids", [])
        cn = class_names_dict.get(task, [f"G{i}" for i in range(int(labels.max() or 5))])

        if len(scores) == 0:
            continue

        # ── Sort by score: DESCENDING (best first) ──
        sort_idx = np.argsort(-scores)[:n_cols]  # Negative = descending
        sorted_scores = scores[sort_idx]
        sorted_labels = labels[sort_idx]
        sorted_ids = [ids[i] for i in sort_idx if i < len(ids)]

        for col_idx, sorted_sample_idx in enumerate(sort_idx):
            ax = axes[task_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            sample_id = ids[sorted_sample_idx] if sorted_sample_idx < len(ids) else None
            score = scores[sorted_sample_idx]
            label = labels[sorted_sample_idx]
            grade_str = cn[int(label)] if label < len(cn) else f"G{int(label)}"
            grade_color = severity_colors_palette[min(int(label), len(severity_colors_palette)-1)]

            # Load image
            img = None
            if center_slices is not None and sample_id in center_slices:
                try:
                    img_data = center_slices[sample_id]
                    # Handle dict or direct array
                    if isinstance(img_data, dict):
                        img = _get_slice_for_task(img_data, task)
                    else:
                        img = img_data
                except Exception as e:
                    logger.debug(f"Failed to load image for {sample_id}: {e}")
                    img = None

            # ──────────────────────────────────────────────────────
            # Display image (robust normalization — fixes black images)
            # ──────────────────────────────────────────────────────
            if img is not None:
                img_clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                img_clean = np.asarray(img_clean, dtype=np.float32)

                # Validate image is not empty
                img_abs_max = np.max(np.abs(img_clean))
                if img_abs_max < 1e-6:
                    ax.set_facecolor("#E0E0E0")
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                           fontsize=8, color="#999999", transform=ax.transAxes)
                else:
                    # ROBUST PERCENTILE-BASED NORMALIZATION
                    nonzero_mask = np.abs(img_clean) > (img_abs_max * 0.01)
                    if nonzero_mask.sum() > 10:
                        nonzero_vals = img_clean[nonzero_mask]
                        vmin_val = np.percentile(nonzero_vals, 2)
                        vmax_val = np.percentile(nonzero_vals, 98)
                    else:
                        vmin_val = np.min(img_clean[img_clean != 0])
                        vmax_val = np.max(img_clean)

                    if vmin_val < vmax_val:
                        img_clean = (np.clip(img_clean, vmin_val, vmax_val) - vmin_val) / (vmax_val - vmin_val + 1e-8)
                        ax.imshow(img_clean, cmap="gray", aspect="equal", vmin=0, vmax=1)
                    else:
                        ax.set_facecolor("#D3D3D3")
            else:
                ax.set_facecolor("#E0E0E0")

            # ──────────────────────────────────────────────────────
            # LARGE SCORE DISPLAY (top-left) — same style as vertical
            # ──────────────────────────────────────────────────────
            score_text = f"{score:.1f}"
            ax.text(0.05, 0.95, score_text,
                   transform=ax.transAxes,
                   ha="left", va="top",
                   fontsize=32, fontweight="bold",
                   color=grade_color,
                   path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                   zorder=10)

            # ── GT LABEL (bottom) — plain dark text, no colored box ──
            ax.text(0.5, -0.10, f"GT: {grade_str}",
                   transform=ax.transAxes,
                   ha="center", va="top",
                   fontsize=11, fontweight="bold",
                   color="#333333")

            # Colored border
            for spine in ax.spines.values():
                spine.set_color(grade_color)
                spine.set_linewidth(2.0)

            # Sample ID indicator (subtle, bottom-right)
            if sample_id:
                # Extract level from sample ID (function defined locally at line 2221)
                level = extract_level(sample_id)
                ax.text(0.98, 0.02, f"{level}",
                       transform=ax.transAxes,
                       ha="right", va="bottom",
                       fontsize=8, color="#999999", style="italic")

        # ── TASK LABEL (row header) ──
        # FEATURED_SHORT is defined locally at line 2607
        short_task = FEATURED_SHORT.get(task, task[:15])
        axes[task_idx, 0].set_ylabel(
            short_task, fontsize=12, fontweight="bold",
            rotation=0, labelpad=12, ha="right", va="center"
        )

    # ── FIGURE TITLE & LEGEND ──
    fig.suptitle(
        "Ranking Scores Sorted Left → Right (Best to Worst)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Legend note
    fig.text(0.5, 0.01,
            "← Score decreases → | Color indicates severity | Smaller font = lower-grade level",
            ha="center", fontsize=10, style="italic", color="#666666")

    fig.subplots_adjust(
        left=0.15, right=0.98, top=0.92, bottom=0.06,
        wspace=0.06, hspace=0.35
    )

    outpath = out_dir / "ranking_scores_sorted.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    logger.info(f"✓ Saved sorted ranking visualization: {outpath}")


if __name__ == '__main__':
    # Example usage
    logger.info("Comprehensive ETH-styled evaluation module loaded successfully")
