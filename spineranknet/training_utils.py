"""Shared training utilities for the SpineRankNet paper trainers.

Used by :mod:`spineranknet.baseline.train_ranking_mse` to factor out
loss helpers, TTA forward passes, checkpoint IO, and uncertainty plots.
Data loading lives in the trainer (:func:`get_dataloaders_fast`, built on
:class:`spineranknet.dataloaders.Genodisc.GenodiscOrdinalDataset`).

Contents
~~~~~~~~
* Logger setup (:func:`setup_logger`).
* Test-time-augmentation forward (:func:`tta_forward_chunked`)
  with optional per-view return for uncertainty quantification.
* Classification loss helpers (:func:`get_loss_type`,
  :func:`get_task_predictions`, :func:`compute_clf_loss`).
* Checkpoint save / load (:func:`save_checkpoint`,
  :func:`load_checkpoint_into_model`) with old-format key remapping.
* TTA uncertainty analysis drivers
  (:func:`run_tta_uncertainty_analysis`,
  :func:`run_rank_tta_uncertainty_analysis`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from spineranknet.config import CLASS_NAMES_DICT, INVALID_LABEL, TASK_DEFINITIONS
from spineranknet.losses.classification import LOSS_REGISTRY, get_predictions


__all__ = [
    "setup_logger",
    "tta_forward_chunked",
    "get_loss_type",
    "get_task_predictions",
    "compute_clf_loss",
    "save_checkpoint",
    "load_checkpoint_into_model",
    "run_tta_uncertainty_analysis",
    "run_rank_tta_uncertainty_analysis",
]

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# LOGGER HELPER
# ════════════════════════════════════════════════════════════════════════════

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to console (and optionally to file).

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.
    log_file : str, optional
        If provided, an additional :class:`~logging.FileHandler` writes to
        this path (appending).
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    log = logging.getLogger(name)
    log.setLevel(level)
    log.handlers.clear()          # avoid duplicate handlers on re-init

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


# ════════════════════════════════════════════════════════════════════════════
# TTA BATCHED FORWARD
# ════════════════════════════════════════════════════════════════════════════

def tta_forward_chunked(
    model: torch.nn.Module,
    tta_images: torch.Tensor,
    chunk_size: int = 16,
    return_per_view: bool = False,
) -> Union[
    Tuple[tuple, Dict[str, torch.Tensor]],
    Tuple[tuple, Dict[str, torch.Tensor], tuple, Dict[str, torch.Tensor]],
]:
    """Run batched TTA forward and average predictions across views.

    Parameters
    ----------
    model : nn.Module
        Model that returns ``(clf_out, rank_out, features)`` where
        ``clf_out`` is a tuple of ``(B, K)`` tensors and ``rank_out``
        is a dict of ``{task: (B,)}`` tensors.
    tta_images : Tensor (N_views, D, H, W) or (N_views, C, D, H, W)
        Stacked TTA views; multi-channel inputs (e.g. ``USE_MULTI_CONTRAST``)
        arrive as ``(N_views, C, D, H, W)``.
    chunk_size : int
        Number of views to forward at once.  16 works well for A40 (48 GB).
    return_per_view : bool
        If ``True``, also return the **un-averaged** per-view logits/scores
        for uncertainty quantification.

    Returns
    -------
    clf_out : tuple of Tensor (1, K)
        Averaged classification logits per task head.
    rank_out : dict {task: Tensor (1,)}
        Averaged ranking scores per task.
    per_view_clf : tuple of Tensor (N_views, K)
        *(only when return_per_view=True)* Raw logits for every view.
    per_view_rank : dict {task: Tensor (N_views,)}
        *(only when return_per_view=True)* Raw ranking scores for every view.
    """
    n_views = tta_images.shape[0]
    all_clf: List[tuple] = []
    all_rank: List[Dict[str, torch.Tensor]] = []

    for start in range(0, n_views, chunk_size):
        chunk = tta_images[start : start + chunk_size]   # (chunk, [C,] D, H, W)
        c_out, r_out, _ = model(chunk)
        if not isinstance(c_out, (tuple, list)):
            c_out = (c_out,)
        all_clf.append(c_out)
        all_rank.append(r_out)

    # Concatenate per-view logits across all chunks: (N_views, K)
    n_heads = len(all_clf[0])
    per_view_clf = tuple(
        torch.cat([c[i] for c in all_clf], dim=0)
        for i in range(n_heads)
    )

    per_view_rank: Dict[str, torch.Tensor] = {}
    if all_rank and all_rank[0]:
        for task_key in all_rank[0]:
            per_view_rank[task_key] = torch.cat(
                [r[task_key] for r in all_rank], dim=0
            )

    # Average classification logits across views
    clf_out = tuple(t.mean(dim=0, keepdim=True) for t in per_view_clf)

    # Average ranking scores across views
    rank_out: Dict[str, torch.Tensor] = {
        k: v.mean(dim=0, keepdim=True) for k, v in per_view_rank.items()
    }

    if return_per_view:
        return clf_out, rank_out, per_view_clf, per_view_rank
    return clf_out, rank_out


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION LOSS HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_loss_type(task: str, config: Any) -> str:
    """Resolve the loss-type name for *task* from config overrides.

    Parameters
    ----------
    task : str
        Task name (must exist in :data:`spineranknet.config.TASK_DEFINITIONS`).
    config : object
        Config exposing an optional ``TASK_LOSS_OVERRIDE`` mapping.

    Returns
    -------
    str
        Loss-type key (e.g. ``"ce"``, ``"corn"``). The config override
        takes precedence over the task default.
    """
    td = TASK_DEFINITIONS[task]
    task_loss_override = getattr(config, "TASK_LOSS_OVERRIDE", {})
    return task_loss_override.get(task, td.get("default_loss", "ce"))


def get_task_predictions(
    logits: torch.Tensor,
    task: str,
    config: Any,
) -> torch.Tensor:
    """Convert raw logits to class predictions, accounting for loss type.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(B, K)`` (or ``(B, K-1)`` for cumulative losses).
    task : str
        Task name used to resolve the loss type.
    config : object
        Config passed to :func:`get_loss_type`.

    Returns
    -------
    torch.Tensor
        Integer class predictions of shape ``(B,)``.
    """
    return get_predictions(logits, get_loss_type(task, config))


def compute_clf_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    config: Any,
    class_weights: Dict[str, Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Compute the classification loss for a single task on valid samples.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(B, K)``.
    labels : torch.Tensor
        Shape ``(B,)`` integer labels; invalid (negative or
        out-of-range) entries are excluded.
    task : str
        Task name used to look up ``num_classes`` and the loss type.
    config : object
        Config exposing ``TASK_LOSS_OVERRIDE`` and ``USE_CLASS_WEIGHTS``.
    class_weights : dict of str to (torch.Tensor or None)
        Mapping task → class-weight tensor (or ``None`` for unweighted).

    Returns
    -------
    torch.Tensor or None
        Scalar loss tensor, or ``None`` when the batch contains no
        valid samples for *task*.
    """
    td = TASK_DEFINITIONS[task]
    nc = int(td["num_classes"])
    valid = (labels >= 0) & (labels < nc)
    if not valid.any():
        return None

    logits_v = logits[valid]
    labels_v = labels[valid]

    loss_type = get_loss_type(task, config)
    wt = class_weights.get(task)
    loss_fn = LOSS_REGISTRY.get(loss_type, LOSS_REGISTRY["ce"])
    return loss_fn(logits_v, labels_v, nc, wt)


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    acc: float,
    save_dir: str,
    model_type: str,
    prefix: str = "ckpt",
    remove_old: bool = True,
) -> Path:
    """Save a model checkpoint, optionally removing previous checkpoints.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose ``state_dict`` is serialised.
    epoch : int
        Epoch index, embedded in the filename and stored in the payload.
    acc : float
        Best metric (typically accuracy) associated with this snapshot.
    save_dir : str
        Parent directory; a subdirectory named *model_type* is created.
    model_type : str
        Backbone identifier used as the subdirectory name.
    prefix : str, optional
        Filename prefix (default ``"ckpt"``).
    remove_old : bool, optional
        If ``True`` (default), every ``*.pt`` file in the target
        directory is removed before saving.

    Returns
    -------
    pathlib.Path
        Path to the saved checkpoint file.
    """
    wdir = Path(save_dir) / model_type
    wdir.mkdir(parents=True, exist_ok=True)
    if remove_old:
        for old in wdir.glob("*.pt"):
            old.unlink()
    path = wdir / f"{prefix}_{epoch}.pt"
    state = {
        "model_weights": model.state_dict(),
        "epoch": epoch,
        "acc": acc,
    }
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")
    return path


def _remap_ranking_head_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remap old full-name ranking-head keys to abbreviated ``rank_*`` names.

    Old checkpoints store ranking heads as e.g.
    ``ranking_heads.Pfirrmann.net.0.weight``. New models use
    ``ranking_heads.rank_pf.net.0.weight``.

    Parameters
    ----------
    state : dict
        State-dict mapping parameter names to tensors.

    Returns
    -------
    dict
        State-dict with renamed keys (values are passed through
        unchanged).
    """
    _RANK_MAP = {td["ranking_head"]: td["ranking_head"]
                 for td in TASK_DEFINITIONS.values()
                 if "ranking_head" in td}
    _TASK_TO_RANK = {task: td["ranking_head"]
                     for task, td in TASK_DEFINITIONS.items()
                     if "ranking_head" in td}
    remapped = {}
    for k, v in state.items():
        new_k = k
        for task_name, rank_key in _TASK_TO_RANK.items():
            new_k = new_k.replace(f"ranking_heads.{task_name}.", f"ranking_heads.{rank_key}.")
        remapped[new_k] = v
    return remapped


def load_checkpoint_into_model(model: nn.Module, ckpt: Dict[str, Any]) -> None:
    """Load a checkpoint dict into *model*, handling old and new formats.

    Supported formats
    ~~~~~~~~~~~~~~~~~
    * **New unified**: ``{"model_weights": state_dict, ...}``
    * **Old hybrid**:  ``{"backbone": state_dict, "ranking_heads": state_dict}``

    Automatically remaps old full-name ranking-head keys (e.g.
    ``ranking_heads.Pfirrmann.*``) to abbreviated ``rank_*`` names and
    filters out parameters whose shapes do not match the target model.

    Parameters
    ----------
    model : torch.nn.Module
        Target model; loaded with ``strict=False``.
    ckpt : dict
        Checkpoint payload in either supported format.
    """
    if "backbone" in ckpt and "ranking_heads" in ckpt:
        merged = {}
        for k, v in ckpt["backbone"].items():
            merged[k] = v
        for k, v in ckpt["ranking_heads"].items():
            merged[f"ranking_heads.{k}"] = v
        merged = _remap_ranking_head_keys(merged)
        model.load_state_dict(merged, strict=False)
    else:
        state = ckpt.get("model_weights", ckpt)
        state = _remap_ranking_head_keys(state)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)


# ════════════════════════════════════════════════════════════════════════════
# TTA UNCERTAINTY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def run_tta_uncertainty_analysis(
    tta_per_view: Dict[str, Dict[str, list]],
    tasks: List[str],
    out_dir: Union[str, Path],
) -> None:
    """Run TTA uncertainty analysis: save CSVs and generate plots.

    Parameters
    ----------
    tta_per_view : dict
        Mapping ``{task: {"per_view_probs": [...], "ids": [...],
        "targets": [...]}}`` collected during TTA evaluation.
    tasks : list of str
        Tasks to analyse.
    out_dir : str or pathlib.Path
        Parent directory; an ``uncertainty/`` subdirectory is created.

    Notes
    -----
    Imports are deferred to avoid pulling matplotlib at module import
    time. Failures are caught and logged as warnings — the call is
    non-fatal.
    """
    if not any(tta_per_view[t]["per_view_probs"] for t in tasks):
        return

    try:
        from spineranknet.visualization.uncertainty import (
            save_tta_predictions_csv,
            compute_and_plot_uncertainty,
        )
        uq_dir = Path(out_dir) / "uncertainty"
        uq_dir.mkdir(exist_ok=True)
        logger.info("\nTTA Uncertainty Analysis")
        logger.info("-" * 40)
        for task in tasks:
            pv = tta_per_view[task]
            if not pv["per_view_probs"]:
                continue
            cn = CLASS_NAMES_DICT.get(
                task,
                [f"C{i}" for i in range(pv["per_view_probs"][0].shape[1])],
            )
            save_tta_predictions_csv(
                per_view_probs=pv["per_view_probs"],
                sample_ids=pv["ids"],
                true_labels=pv["targets"],
                class_names=cn,
                task=task,
                out_dir=uq_dir,
            )
            compute_and_plot_uncertainty(
                per_view_probs=pv["per_view_probs"],
                true_labels=pv["targets"],
                class_names=cn,
                task=task,
                out_dir=uq_dir,
            )
    except Exception as e:
        logger.warning(f"TTA uncertainty analysis failed (non-fatal): {e}")


def run_rank_tta_uncertainty_analysis(
    rank_results: Dict[str, Dict[str, list]],
    tasks: List[str],
    thresholds: Dict[str, Any],
    out_dir: Union[str, Path],
) -> None:
    """Compute and plot per-sample TTA standard deviation of ranking scores.

    For each sample *t*::

        σ_TTA(t) = sqrt( 1/N_TTA · Σ_v ( s_v(t) − s̄(t) )² )

    i.e. the standard deviation of per-view ranking scores.

    Parameters
    ----------
    rank_results : dict
        Mapping ``{task: {"scores": [...], "labels": [...],
        "sigma_tta": [...], ...}}``. The ``sigma_tta`` key is populated
        only when TTA was active during evaluation.
    tasks : list of str
        Tasks to analyse.
    thresholds : dict of str to numpy.ndarray
        Decision thresholds from val-set calibration.
    out_dir : str or pathlib.Path
        Parent directory; an ``uncertainty/ranking/`` subdirectory is
        created.

    Notes
    -----
    The call is a no-op when no task has populated ``sigma_tta``.
    Failures are caught and logged as warnings — non-fatal.
    """
    # Only proceed if at least one task has sigma_tta data
    has_sigma = any(
        rank_results.get(t, {}).get("sigma_tta") for t in tasks
    )
    if not has_sigma:
        return

    try:
        from spineranknet.visualization.uncertainty import compute_and_plot_rank_uncertainty
        uq_dir = Path(out_dir) / "uncertainty" / "ranking"
        uq_dir.mkdir(parents=True, exist_ok=True)
        logger.info("\nRanking TTA Uncertainty Analysis (σ_TTA)")
        logger.info("-" * 45)
        compute_and_plot_rank_uncertainty(rank_results, tasks, thresholds, uq_dir)
    except Exception as e:
        logger.warning(f"Ranking TTA uncertainty analysis failed (non-fatal): {e}")
