#!/usr/bin/env python3
"""
SpineRankNet — Unified Training / Evaluation Script (Classification + Ranking)
==============================================================================

Trains per-task classification and/or ranking heads on top of a 3D
ResNet-18/34/50 encoder (``MultiHeadGradingRanker``) and evaluates them.

Training modes (controlled via ``--mode`` or loss weights):
  - **train** (default): Hybrid classification + ranking end-to-end.
  - **clf**: Classification loss only (ranking & triplet weights zeroed).
  - **rank**: Ranking loss only (classification weight zeroed).
  - **eval**: Evaluate an existing checkpoint.

The combined loss is::

    L = lambda_clf * L_clf  +  lambda_rank * L_rank  +  lambda_triplet * L_triplet

Classification losses (from ``losses/classification.py``):
  - CE, EMD (Wasserstein), MAE, CORN, CORAL, Soft-CE, CLM

  Use ``TASK_LOSS_OVERRIDE`` with ``_all_`` wildcard to override all tasks.

Ranking losses (from ``losses/ranking_losses.py``):
  - **Pairwise**: RelativeAttributes, DeepRankSVM, DeepRelativeAttributes, JustNoticeableDifferences, RankNet, SpineRank
  - **Pointwise**: MSE, MAE, TripletOrdinal

Key features:
  - Default ``RANK_FROM_LOGITS=True``: ranking heads built on clf logits
  - Named ablation presets (``--ablation <name>``, see ``--list_ablations``)
  - Optional ``--encoder_only`` to load only encoder (+ clf heads) from a
    Phase-1 classification checkpoint
  - Comprehensive evaluation: threshold-optimised BalAcc, QWK, Spearman rho,
    Kendall tau, ROC-AUC, confusion matrices, density histograms, boxplots
  - T2w + S1 per-patient test filtering (default, disable with ``--no_test_filter``)

Data locations are read from the YAML config (``GENODICT_PATH``: genodict
pickle with Train/Val/Test splits; ``IVD_PATH``: directory of per-IVD ``.npy``
volumes) or from ``--genodict_path`` / ``--ivd_path``.  The shipped configs use
the relative layout ``data/GENODISCv2/...``; ``--data_root`` or the
``GENODISC_ROOT`` environment variable relocates that prefix.

Usage (paper protocol; ``scripts/run_all_experiments.py`` prints the full set)::

    # Phase 1: classification pretraining (CE; also --clf_loss corn / mae)
    python -m spineranknet.baseline.train_ranking_mse \\
        --config config/ranking_logit_heads.yaml --mode clf --clf_loss ce \\
        --backbone resnet18 --epochs 300 --batch_size 32 --oversample \\
        --selection_criterion bal_acc \\
        --save_dir results/phase1/ce/weights --eval_dir results/phase1/ce/eval

    # Phase 2: SpineRank fine-tuning from the Phase-1 checkpoint
    python -m spineranknet.baseline.train_ranking_mse \\
        --config config/ranking_logit_heads.yaml --mode train --backbone resnet18 \\
        --checkpoint results/phase1/ce/weights/resnet18/best.pt \\
        --encoder_only --rank_from_logits \\
        --ranking_loss SpineRank --ranking_head mlp \\
        --clf_loss_weight 0 --ranking_loss_weight 1 \\
        --spinerank_margin_base 0.5 --spinerank_margin_scale 1.5 --spinerank_c2 0.25 \\
        --epochs 200 --batch_size 32 --oversample --selection_criterion bal_acc \\
        --save_dir results/ranking/resnet18/spinerank_mlp/weights \\
        --eval_dir results/ranking/resnet18/spinerank_mlp/eval

    # Evaluate (54-view TTA is on by default; --no_tta disables it)
    python -m spineranknet.baseline.train_ranking_mse \\
        --config config/ranking_logit_heads.yaml --mode eval --backbone resnet18 \\
        --checkpoint results/ranking/resnet18/spinerank_mlp/weights/resnet18/best.pt \\
        --rank_from_logits --ranking_loss SpineRank --ranking_head mlp \\
        --threshold_method grid \\
        --eval_dir results/ranking/resnet18/spinerank_mlp/eval_grid

    # List all ablation presets
    python -m spineranknet.baseline.train_ranking_mse --list_ablations
"""

import os
import json
import random
import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    mean_absolute_error,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)
from scipy.stats import spearmanr, kendalltau

# ── project imports ──────────────────────────────────────────────────────
from spineranknet.config import (
    TASK_DEFINITIONS,
    INVALID_LABEL,
    CLASS_NAMES_DICT,
    ORDINAL_TASKS_13,
    STANDARD_TASKS_13,
    ETH_COLORS,
    expand_env_paths,
    resolve_tasks,
)
from spineranknet.losses.classification import (
    LOSS_REGISTRY,
    build_task_weighting,
)
from spineranknet.losses.ranking_losses import (
    TripletMarginOrdinalLoss,
    cross_task_ranking_loss,
)
# Full ranking-loss registry (paper loss + all comparison/ablation losses) lives
# in the experiments subpackage so the core package advertises only the paper loss.
from spineranknet.experiments.comparison_losses import RANKING_LOSSES
from spineranknet.networks.heads.ranking_heads import UNIVERSAL_SCALE
from spineranknet.rank_calibration import compute_agreement_loss, rank_calibrated_classify
from spineranknet.networks.MultiHeadGradingRanker import build_grading_ranker
from spineranknet.dataloaders.Genodisc import GenodiscOrdinalDataset
from spineranknet.visualization.evaluation import (
    compute_comprehensive_metrics,
    plot_confusion_matrix_eth,
    plot_roc_curves,
    plot_score_density_histogram,
    plot_score_boxplots,
    plot_ranked_example_grid,
    plot_ranked_image_grid,
    plot_ranking_summary,
    plot_threshold_search,
    plot_error_magnitude_histogram,
    plot_per_boundary_roc,
    group_by_patient,
    plot_patient_level_ranking_grid,
    plot_success_failure_grids,
    group_by_patient_multitask,
    plot_multi_pathology_patient_grid,
    plot_ranking_by_level_vertical,  # NEW: Vertical levels visualization
    plot_ranking_scores_sorted,        # NEW: Sorted scores visualization
    plot_horizontal_score_boxplots,
    plot_binarized_summary_roc,
    FEATURED_TASKS,
    save_viz_cache,
    set_show_sample_ids,
    TASK_DEFINITIONS as _VIZ_TASK_DEFINITIONS,
    CLASS_NAMES_DICT as _VIZ_CLASS_NAMES_DICT,
)
from spineranknet.training_utils import (
    get_task_predictions,
    compute_clf_loss,
    tta_forward_chunked,
)
from spineranknet.baseline.configs import HybridConfig
# Ablation presets live in the experiments subpackage; the ablation-capable
# trainer imports them to honour --ablation flags.
from spineranknet.experiments.presets import ABLATION_PRESETS, apply_ablation_preset

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "text.color": ETH_COLORS["black"],
        "axes.labelcolor": ETH_COLORS["black"],
        "xtick.color": ETH_COLORS["black"],
        "ytick.color": ETH_COLORS["black"],
        "font.family": "sans-serif",
        "font.size": 10,
    }
)


# ════════════════════════════════════════════════════════════════════════════
# FAST DATALOADER  (numpy-based, from pre-extracted IVD volumes)
#
# Uses GenodiscOrdinalDataset (reads .npy files via pickle genodict).
# Extended to produce UpperModic / LowerModic (4-class ordinal) labels that
# match the task definitions in config.py TASK_DEFINITIONS.
# ════════════════════════════════════════════════════════════════════════════

import pickle
from spineranknet.dataloaders.utils import (
    label_check as _label_check,
    label_check_marrow as _label_check_marrow,
)


# =============================================================================
# Training-config persistence  (dump alongside best.pt, auto-load at eval)
# =============================================================================

#: Filename used to persist the effective training config next to ``best.pt``.
#: The same YAML can be replayed via ``--config <path>`` for a full re-train.
TRAIN_CONFIG_FILENAME: str = "train_config.yaml"

#: Default relative location of the GENODISC data used in the shipped YAML
#: configs. ``--data_root`` / ``$GENODISC_ROOT`` replace this prefix.
DEFAULT_DATA_PREFIX: str = "data/GENODISCv2"


def _dump_train_config(config: Any, save_dir: Path) -> None:
    """Write a YAML capturing the effective HybridConfig used for this run.

    The file is dropped at ``save_dir/train_config.yaml`` so it sits next to
    ``best.pt`` / ``ckpt_*.pt``.  It is symmetric with ``--config``: any
    later run can be replayed with::

        python -m spineranknet.baseline.train_ranking_mse \\
            --config <save_dir>/train_config.yaml

    and ``--mode eval`` against the colocated checkpoint will pick it up
    automatically (see :func:`_maybe_load_train_config`).

    The dump contains the local paths used for the run (``*_PATH``,
    ``*_DIR``, ``CHECKPOINT``). Remove or replace them before sharing a run
    folder or releasing weights.
    """
    try:
        import yaml as _yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping train_config.yaml dump")
        return

    # Pull every public attribute off the (data)class instance.  We only want
    # JSON/YAML-friendly leaves; drop callables, modules, and private fields.
    payload: Dict[str, Any] = {}
    for k in sorted(set(vars(config)) | set(getattr(type(config), "__dataclass_fields__", {}))):
        if k.startswith("_"):
            continue
        v = getattr(config, k, None)
        if callable(v) or hasattr(v, "__module__") and not isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
            continue
        # tuples → lists for round-trip; everything else is already YAML-safe.
        payload[k] = list(v) if isinstance(v, tuple) else v

    out = save_dir / TRAIN_CONFIG_FILENAME
    try:
        with open(out, "w") as f:
            f.write(
                "# Effective training config. Contains local paths (*_PATH, *_DIR,\n"
                "# CHECKPOINT): remove or replace them before sharing this file.\n"
            )
            _yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)
        logger.info("Wrote effective config to %s", out)
    except (OSError, _yaml.YAMLError) as e:
        logger.warning("Could not write %s: %s", out, e)


def _maybe_load_train_config(args: argparse.Namespace) -> None:
    """For ``--mode eval``: if the user didn't pass ``--config``, look for a
    sibling ``train_config.yaml`` (or legacy ``train_config.json``) next to
    the checkpoint and set ``args.config`` so the normal YAML-load path takes
    over.  Silent no-op when nothing is found.
    """
    if getattr(args, "config", None):
        return  # user passed --config explicitly; respect it
    ckpt = getattr(args, "checkpoint", None)
    if not ckpt:
        return
    ckpt_path = Path(ckpt)
    for parent in (ckpt_path.parent, ckpt_path.parent.parent):
        cand = parent / TRAIN_CONFIG_FILENAME
        if cand.is_file():
            args.config = str(cand)
            logger.info("Auto-loaded training config from %s", cand)
            return


class _ExtendedOrdinalDataset(GenodiscOrdinalDataset):
    """GenodiscOrdinalDataset + UpperModic/LowerModic/IVDlevel extraction.

    The parent class produces labels using the old task naming (UpperMarrow,
    LowerMarrow are binary; no UpperModic/LowerModic).  This subclass adds:
      - UpperModic (0-3): derived from UpperModic1/2/3/M flags
      - LowerModic (0-3): derived from LowerModic1/2/3/M flags
      - IVDlevel  (0-5): derived from volume name suffix
    """

    @staticmethod
    def _derive_modic_type(score: Dict[str, Any], prefix: str) -> int:
        """Derive Modic type (0-3) from component flags, matching the canonical grading scheme."""
        type_keys = {1: f"{prefix}Modic1", 2: f"{prefix}Modic2", 3: f"{prefix}Modic3"}
        values = {}
        for cls, key in type_keys.items():
            raw = score.get(key, float("nan"))
            try:
                v = int(raw) if not (isinstance(raw, float) and np.isnan(raw)) else -100
            except (TypeError, ValueError):
                v = -100
            if v >= 0:
                values[cls] = v
        if not values:
            return -100
        present = [cls for cls, v in values.items() if v > 0]
        return max(present) if present else 0

    def _extract_labels(self, score: Dict[str, Any]) -> Dict[str, int]:
        labels = super()._extract_labels(score)
        # Add UpperModic / LowerModic (4-class ordinal: 0=None, 1=I, 2=II, 3=III)
        labels["UpperModic"] = self._derive_modic_type(score, "Upper")
        labels["LowerModic"] = self._derive_modic_type(score, "Lower")
        return labels


def _fast_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for _ExtendedOrdinalDataset -> (B, 1, D, H, W) images.

    The old GenodiscOrdinalDataset returns images as (1, S, H, W) numpy.
    We stack them into (B, 1, S, H, W) to match the (B, C, D, H, W)
    format expected by MultiHeadGradingRanker / ResNet3D encoders.
    """
    images = torch.from_numpy(
        np.stack([s["image"] for s in batch], axis=0)
    ).float()
    names = [s["name"] for s in batch]
    label_keys = batch[0]["labels"].keys()
    labels = {
        k: torch.tensor([s["labels"][k] for s in batch], dtype=torch.long)
        for k in label_keys
    }
    return {"image": images, "labels": labels, "name": names}


class _ExtendedTTAOrdinalDataset(_ExtendedOrdinalDataset):
    """TTA variant of _ExtendedOrdinalDataset for fast eval with augmentation.

    Always loads the T2_S1 sequence (canonical for evaluation) and generates
    a deterministic 54-view augmentation grid:
    2 flips × 3 translation-x × 3 translation-y × 3 slice-shifts = 54 views.

    Inherits ``_extract_labels`` from ``_ExtendedOrdinalDataset`` (ordinal +
    UpperModic/LowerModic) via MRO.

    Returns ``{'images': [list of (1,S,H,W) np.arrays], 'labels': dict,
    'name': str}``.
    """

    def get_vb_by_name(self, curr_vb_name: str) -> Dict[str, Any]:
        import cv2

        curr_ = self.genodict[curr_vb_name]
        score = curr_['score']
        labels = self._extract_labels(score)

        # Always load T2_S1 (canonical evaluation sequence)
        curr_seq = 'T2_S1'
        path = self.scan_path + curr_vb_name + '_' + curr_seq + '.npy'
        if os.path.isfile(path):
            image = np.load(path, allow_pickle=True)
        else:
            image = np.zeros(
                (self.original_height, self.original_width, self.original_slices),
                np.float32,
            )
            labels = self._invalidate_all(labels)

        if not np.isfinite(image).all():
            logger.error(f'Image contains NaN: {curr_vb_name}')

        # ── Generate 54-view TTA augmentation grid ──
        images = []
        orig_image = image.copy()
        for use_flip in [False, True]:
            for delta_x in [-16, 0, 16]:
                for delta_y in [-16, 0, 16]:
                    for delta_mid in [-1, 0, 1]:
                        img = orig_image.copy()
                        num_rows, num_cols, num_slices = img.shape
                        max_cols = num_cols - self.margin_cols
                        min_cols = self.margin_cols
                        max_rows = num_rows - self.margin_rows
                        min_rows = self.margin_rows

                        # Slice shift
                        if delta_mid > 0:
                            img = np.roll(img, delta_mid, axis=2)
                            img[:, :, 0:delta_mid] = 0
                        elif delta_mid < 0:
                            img = np.roll(img, delta_mid, axis=2)
                            img[:, :, num_slices + delta_mid:num_slices] = 0

                        # Flip (axial slice reversal)
                        if use_flip:
                            img = np.flip(img, axis=2).copy()

                        # Empty-slice mask (before crop)
                        slice_start = max(0, (num_slices - self.slices) // 2)
                        slice_end = slice_start + self.slices
                        zero_slice = np.mean(img, axis=(0, 1))
                        zero_slice = zero_slice[slice_start:slice_end]

                        # Translation
                        max_cols += delta_y
                        min_cols += delta_y
                        max_rows += delta_x
                        min_rows += delta_x

                        # Boundary clamping
                        if max_cols >= num_cols:
                            min_cols += (max_cols - num_cols)
                            max_cols = num_cols - 1
                        if max_rows >= num_rows:
                            min_rows += (max_rows - num_rows)
                            max_rows = num_rows - 1
                        if min_cols < 0:
                            max_cols += abs(min_cols)
                            min_cols = 0
                        if min_rows < 0:
                            max_rows += abs(min_rows)
                            min_rows = 0
                        max_cols = int(np.round(max_cols))
                        min_cols = int(np.round(min_cols))
                        max_rows = int(np.round(max_rows))
                        min_rows = int(np.round(min_rows))

                        # Select slices, crop, resize
                        img = img[:, :, slice_start:slice_end]
                        img = cv2.resize(
                            img[min_rows:max_rows, min_cols:max_cols, :],
                            (self.width, self.height),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        img[:, :, zero_slice == 0] = 0
                        img = np.transpose(img, (2, 0, 1))[None, :, :, :]
                        images.append(img)

        # Derive VertebralLevel
        if labels.get('Pfirrmann', 0) == -100 and labels.get('Narrowing', 0) == -100:
            labels['VertebralLevel'] = -100
        else:
            labels['VertebralLevel'] = self._vertebral_level_from_name(curr_vb_name)

        return {'images': images, 'labels': labels, 'name': curr_vb_name}


def _fast_tta_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for _ExtendedTTAOrdinalDataset → tta_images tensor.

    The TTA dataset returns ``{'images': [list of (1,S,H,W) np.arrays], ...}``
    for each sample.  With batch_size=1, we stack the list into a single
    ``tta_images`` tensor of shape ``(N_views, 1, S, H, W)`` compatible with
    ``tta_forward_chunked``.  Labels become (1,) tensors for the single sample.
    """
    assert len(batch) == 1, (
        f"TTA collate expects batch_size=1, got {len(batch)}"
    )
    s = batch[0]
    # Stack list of (1, S, H, W) arrays → (N_views, 1, S, H, W)
    tta_images = torch.from_numpy(
        np.stack(s["images"], axis=0)
    ).float()
    # Also provide CENTER (non-augmented) view as 'image' for viz
    # TTA grid: 2 flips × 3 dx × 3 dy × 3 dz = 54 views
    # Non-augmented (flip=False, dx=0, dy=0, dz=0) is at:
    #   flip=0 (first), dx=1 (middle), dy=1 (middle), dz=1 (middle)
    #   index = 0*27 + 1*9 + 1*3 + 1*1 = 13
    CENTER_VIEW_IDX = 13  # No augmentation: False, 0, 0, 0
    image = tta_images[CENTER_VIEW_IDX:CENTER_VIEW_IDX+1]  # (1, 1, S, H, W) ✓ CENTERED
    labels = {
        k: torch.tensor([v], dtype=torch.long) for k, v in s["labels"].items()
    }
    return {
        "image": image,
        "tta_images": tta_images,
        "labels": labels,
        "name": [s["name"]],
    }


def get_dataloaders_fast(
    config: Any,
    tasks: List[str],
    data_fraction: float = 1.0,
    seed: int = 42,
    use_tta_test: bool = False,
) -> Dict[str, DataLoader]:
    """Build dataloaders using the fast numpy pipeline (GenodiscOrdinalDataset).

    Reads pre-extracted .npy IVD volumes from ``config.IVD_PATH`` using
    the genodict pickle at ``config.GENODICT_PATH``.

    Parameters
    ----------
    use_tta_test : bool
        If True, wraps the test set with ``_ExtendedTTAOrdinalDataset``
        (54-view TTA grid: 2 flips × 3 dx × 3 dy × 3 slice-shifts).
        The test DataLoader uses batch_size=1 with ``_fast_tta_collate_fn``
        so that each sample yields ``tta_images (N_views, 1, D, H, W)``.
    """
    genodict_path = getattr(config, "GENODICT_PATH", "")
    ivd_path = getattr(config, "IVD_PATH", "")

    if not genodict_path or not ivd_path:
        raise ValueError(
            "Fast dataloader requires GENODICT_PATH and IVD_PATH in config. "
            "Set them in the YAML or via --genodict_path / --ivd_path."
        )

    with open(genodict_path, "rb") as f:
        geno = pickle.load(f)

    slices = getattr(config, "N_SLICES", 12)
    height = config.HEIGHT
    width = config.WIDTH
    bs = config.BATCH_SIZE
    nw = config.NUM_WORKERS

    # Train
    train_ds = _ExtendedOrdinalDataset(
        geno["Train"], ivd_path, transform=True,
        slices=slices, height=height, width=width,
    )
    if 0 < data_fraction < 1.0:
        from torch.utils.data import Subset
        n = len(train_ds)
        k = max(1, int(n * data_fraction))
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)[:k].tolist()
        train_ds = Subset(train_ds, indices)
        logger.info(f"  Data fraction={data_fraction:.0%}: {k}/{n} training samples")

    # Val
    val_ds = _ExtendedOrdinalDataset(
        geno["Val"], ivd_path, transform=False,
        slices=slices, height=height, width=width,
    )

    # Test — use TTA variant when requested
    if use_tta_test:
        # transform=True triggers the 54-view TTA augmentation grid
        test_ds = _ExtendedTTAOrdinalDataset(
            geno["Test"], ivd_path, transform=True,
            slices=slices, height=height, width=width,
        )
        test_bs = 1  # TTA requires batch_size=1
        test_collate = _fast_tta_collate_fn
        logger.info("  TTA enabled: 54-view augmentation grid on test set")
    else:
        test_ds = _ExtendedOrdinalDataset(
            geno["Test"], ivd_path, transform=False,
            slices=slices, height=height, width=width,
        )
        test_bs = bs
        test_collate = _fast_collate_fn

    loaders = {
        "Train": DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=True, drop_last=True,
            collate_fn=_fast_collate_fn,
        ),
        "Val": DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=True,
            collate_fn=_fast_collate_fn,
        ),
        "Test": DataLoader(
            test_ds, batch_size=test_bs, shuffle=False,
            num_workers=nw, pin_memory=True,
            collate_fn=test_collate,
        ),
    }

    # Store genodict on the loader for class weight computation
    loaders["_genodict_train"] = geno["Train"]

    return loaders


def compute_class_weights_fast(
    genodict: dict,
    tasks: List[str],
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute class weights from a genodict (fast, no image loading).

    Handles all task variants including UpperModic/LowerModic (4-class).
    """
    from collections import Counter

    result: Dict[str, Optional[torch.Tensor]] = {}

    for task in tasks:
        td = TASK_DEFINITIONS.get(task)
        if td is None:
            continue
        nc = td["num_classes"]
        is_binary = td.get("binary", False)
        counts = Counter()

        if task == "IVDlevel":
            from spineranknet.dataloaders.Genodisc import GenodiscDataset as _GD
            for vb_name in genodict:
                lbl = _GD._vertebral_level_from_name(vb_name)
                if 0 <= lbl < nc:
                    counts[lbl] += 1
        elif task in ("UpperModic", "LowerModic"):
            prefix = "Upper" if task == "UpperModic" else "Lower"
            for vb in genodict.values():
                lbl = _ExtendedOrdinalDataset._derive_modic_type(vb["score"], prefix)
                if 0 <= lbl < nc:
                    counts[lbl] += 1
        elif task in ("UpperMarrow", "LowerMarrow"):
            for vb in genodict.values():
                s = vb["score"]
                if task == "UpperMarrow":
                    lbl = _label_check_marrow(s["UpperModic1"], s["UpperModic2"],
                                              s["UpperModic3"], s["UpperModicM"])
                else:
                    lbl = _label_check_marrow(s["LowerModic1"], s["LowerModic2"],
                                              s["LowerModic3"], s["LowerModicM"])
                if 0 <= lbl < nc:
                    counts[lbl] += 1
        else:
            score_key = td.get("dataset_key", task)
            for vb in genodict.values():
                s = vb["score"]
                try:
                    if score_key == "Pfirrmann":
                        lbl = _label_check(s["Pfirrmann"]) - 1
                    elif score_key == "Spondylolisthesis" and is_binary:
                        raw = _label_check(s["Spondylolisthesis"])
                        lbl = (1 if raw > 0 else 0) if raw >= 0 else -100
                    elif score_key == "Spondylolisthesis":
                        raw = _label_check(s["Spondylolisthesis"])
                        lbl = min(int(raw), 2) if raw >= 0 else -100
                    elif is_binary:
                        raw = _label_check(s[score_key])
                        lbl = (1 if raw > 0 else 0) if raw >= 0 else -100
                    else:
                        lbl = _label_check(s[score_key])
                except (KeyError, TypeError):
                    continue
                if 0 <= lbl < nc:
                    counts[lbl] += 1

        total = sum(counts.values())
        w = np.ones(nc, dtype=np.float32)
        for c, cnt in counts.items():
            if cnt > 0 and c < nc:
                w[c] = total / (len(counts) * cnt)
        result[task] = torch.tensor(w, dtype=torch.float32).cuda()
        logger.info(f"  {task:25s}: {total:5d} samples -> {[f'{x:.2f}' for x in w.tolist()]}")

    return result


# ════════════════════════════════════════════════════════════════════════════
# PAIR GENERATION
# ════════════════════════════════════════════════════════════════════════════


def generate_pairs(
    labels: torch.Tensor,
    scores: torch.Tensor,
    max_pairs: int = 500,
    hard_ratio: float = 0.7,
    hard_margin: float = 1.0,
    num_classes: int = 5,
    skip_equal: bool = False,
    level_labels: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Build (i, j) pair indices with y_ij, pair_type, severity_weight.

    Hard-negative mining: ``hard_ratio`` fraction of pairs are the
    hardest (highest ``hardness``), the rest are randomly sampled
    from the easy pool.

    Severity weights are normalized to the universal [0, UNIVERSAL_SCALE]
    range: ``sev_normalized = |la - lb| * (UNIVERSAL_SCALE / max(K-1, 1))``.

    Parameters
    ----------
    skip_equal : bool
        If ``True``, skip pairs where both samples have the same label
        (i.e. only use ordinal pairs with at least one grade difference).
    level_labels : Optional[torch.Tensor]
        If provided, only form pairs between samples at the same IVD level.
        Must be aligned with ``labels`` (same length, same mask applied).
    """
    labs = labels.detach().cpu().numpy()
    scrs = scores.detach().cpu().numpy()
    lvls = level_labels.detach().cpu().numpy() if level_labels is not None else None

    valid = np.where(labs >= 0)[0]
    if len(valid) < 2:
        return None

    vl = labs[valid]
    vs = scrs[valid]
    v_lvl = lvls[valid] if lvls is not None else None

    # Severity normalization factor: maps |la-lb| from [0, K-1] to [0, SCALE]
    sev_scale = UNIVERSAL_SCALE / max(num_classes - 1, 1)

    pairs, yij, ptypes, sev, hardness = [], [], [], [], []

    for a in range(len(vl)):
        for b in range(a + 1, len(vl)):
            # Level-aware: skip cross-level pairs
            if v_lvl is not None and v_lvl[a] != v_lvl[b]:
                continue

            la, lb = vl[a], vl[b]
            sa, sb = vs[a], vs[b]

            if la == lb:
                if skip_equal:
                    continue
                y = 0.0
                ptype = 0  # similarity pair
            elif la > lb:
                y = 1.0
                ptype = 1  # ordinal pair
            else:
                y = -1.0
                ptype = 1

            sev_w = abs(float(la - lb)) * sev_scale
            h = abs(float(sa - sb))
            if ptype == 1 and y * (sa - sb) < hard_margin:
                h = hard_margin - y * (sa - sb)

            pairs.append((valid[a], valid[b]))
            yij.append(y)
            ptypes.append(ptype)
            sev.append(sev_w)
            hardness.append(h)

    if not pairs:
        return None

    n_hard = int(max_pairs * hard_ratio)
    n_easy = max_pairs - n_hard

    order = np.argsort(hardness)[::-1]
    hard_idx = order[:n_hard].tolist()
    easy_pool = order[n_hard:]
    if len(easy_pool) > n_easy and n_easy > 0:
        easy_idx = np.random.choice(easy_pool, n_easy, replace=False).tolist()
    else:
        easy_idx = easy_pool.tolist()

    sel = hard_idx + easy_idx
    if not sel:
        return None

    device = scores.device
    pidx = torch.tensor([pairs[i] for i in sel], dtype=torch.long, device=device)
    y_out = torch.tensor([yij[i] for i in sel], dtype=torch.float32, device=device)
    pt_out = torch.tensor([ptypes[i] for i in sel], dtype=torch.long, device=device)
    sv_out = torch.tensor([sev[i] for i in sel], dtype=torch.float32, device=device)

    return pidx, y_out, pt_out, sv_out


# ════════════════════════════════════════════════════════════════════════════
# SCORE DISCRETIZATION
# ════════════════════════════════════════════════════════════════════════════


def concordance_index(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, int]:
    """Fraction of comparable pairs where predicted scores agree on direction.

    Concordance index (a.k.a. C-index, pairwise ranking accuracy) — the
    textbook ordinal-ranking metric.  For every pair ``(i, j)`` with
    ``g_i > g_j`` (comparable), the model is "concordant" if ``s_i > s_j``,
    "discordant" if ``s_i < s_j``, and gets half-credit for ties
    (``s_i == s_j``).  Returns the fraction of comparable pairs that are
    concordant (with tied-score half-credit).

    Equivalent to AUC when labels are binary, and to ``(1 + τ_a) / 2`` for
    Kendall's τ_a — but always interpretable as "fraction of pairs ranked
    correctly". 0.5 is chance, 1.0 is perfect.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Continuous predicted ranking scores.
    labels : np.ndarray, shape (N,)
        Integer ordinal grades (only relative order matters).

    Returns
    -------
    Tuple[float, int]
        ``(c_index, n_comparable_pairs)``.  Returns ``(0.5, 0)`` if no
        comparable pairs exist (all labels identical).
    """
    s = np.asarray(scores, dtype=np.float64)
    l = np.asarray(labels, dtype=np.float64)
    if s.size < 2:
        return 0.5, 0
    # Pairwise differences via broadcasting — only count each pair once via g_i > g_j.
    dl = l[:, None] - l[None, :]
    ds = s[:, None] - s[None, :]
    comparable = dl > 0
    n_pairs = int(comparable.sum())
    if n_pairs == 0:
        return 0.5, 0
    concordant = (ds > 0) & comparable
    tied      = (ds == 0) & comparable
    c = (concordant.sum() + 0.5 * tied.sum()) / n_pairs
    return float(c), n_pairs


def discretize_ranking_scores(
    scores: np.ndarray,
    labels: np.ndarray | None,
    nc: int,
    mode: str = "target_aligned",
    scale: float = UNIVERSAL_SCALE,
) -> np.ndarray:
    """Map continuous ``[0, scale]`` ranking scores to ``[0, K-1]`` class indices.

    Parameters
    ----------
    scores : array of shape (N,)
        Continuous severity scores on ``[0, scale]``.
    labels : array of shape (N,) or None
        Ground-truth class labels (needed for ``"centroid"`` mode).
    nc : int
        Number of ordinal classes *K*.
    mode : str
        ``"uniform"``  – legacy: ``bin_width = scale / K`` (misaligned with
        training targets).
        ``"target_aligned"``  – midpoints between training targets
        ``c * scale / (K-1)``.  Deterministic, zero-cost, exactly corrects the
        misalignment for pointwise losses.
        ``"centroid"``  – midpoints between per-class median scores. Data-driven;
        best for pairwise losses where no explicit targets exist.
    scale : float
        Universal score scale (default 10.0).

    Returns
    -------
    preds : ndarray of shape (N,), dtype int
        Predicted class indices in ``[0, K-1]``.
    """
    if mode == "centroid" and labels is not None:
        centroids: list[float] = []
        for c in range(nc):
            mask = labels == c
            if mask.sum() >= 3:
                centroids.append(float(np.median(scores[mask])))
            else:
                # Fallback for under-represented classes
                centroids.append(c * scale / max(nc - 1, 1))
        # Ensure monotonicity
        for i in range(1, len(centroids)):
            if centroids[i] <= centroids[i - 1]:
                centroids[i] = centroids[i - 1] + 1e-4
        thresholds = np.array(
            [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(nc - 1)]
        )
        return np.clip(np.searchsorted(thresholds, scores), 0, nc - 1)

    if mode == "target_aligned":
        targets = np.array(
            [c * scale / max(nc - 1, 1) for c in range(nc)]
        )
        thresholds = (targets[:-1] + targets[1:]) / 2.0
        return np.clip(np.searchsorted(thresholds, scores), 0, nc - 1)

    # "uniform" — legacy behaviour
    bin_width = scale / nc
    return np.clip((scores / bin_width).astype(int), 0, nc - 1)


# ════════════════════════════════════════════════════════════════════════════
# OPTIMAL THRESHOLD SEARCH
# ════════════════════════════════════════════════════════════════════════════


def _thresholds_isotonic(
    scores: np.ndarray, labels: np.ndarray, num_classes: int,
) -> np.ndarray:
    """Isotonic-regression thresholds for ordinal score discretisation.

    Fits a monotonic mapping from continuous scores → class labels, then
    derives K-1 thresholds at the transition points.
    """
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)

    # Walk through the fitted values in sorted score order to find class
    # transition points.
    order = np.argsort(scores)
    fitted = ir.predict(scores[order])
    sorted_scores = scores[order]

    thresholds = []
    for c in range(num_classes - 1):
        boundary = c + 0.5
        above = np.where(fitted >= boundary)[0]
        if len(above):
            thresholds.append(float(sorted_scores[above[0]]))
        else:
            thresholds.append(float(c + 1) * UNIVERSAL_SCALE / max(num_classes - 1, 1))
    return np.array(thresholds)


def _thresholds_gmm(
    scores: np.ndarray, labels: np.ndarray, num_classes: int,
) -> np.ndarray:
    """Gaussian-mixture-model thresholds.

    Fits *K* 1-D Gaussians to the score distribution (initialised from class
    statistics), then places thresholds at posterior cross-over points between
    adjacent components sorted by mean.
    """
    from sklearn.mixture import GaussianMixture

    # Initialise means from class medians for stable convergence
    means_init = np.array([
        np.median(scores[labels == c]) if (labels == c).sum() >= 2
        else c * UNIVERSAL_SCALE / max(num_classes - 1, 1)
        for c in range(num_classes)
    ]).reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=num_classes,
        means_init=means_init,
        max_iter=200,
        n_init=1,
        random_state=42,
    )
    gmm.fit(scores.reshape(-1, 1))

    # Sort components by mean
    comp_order = np.argsort(gmm.means_.ravel())
    sorted_means = gmm.means_.ravel()[comp_order]
    sorted_stds = np.sqrt(gmm.covariances_.ravel()[comp_order])
    sorted_weights = gmm.weights_[comp_order]

    # Thresholds at posterior cross-over between adjacent components
    thresholds = []
    for i in range(num_classes - 1):
        mu_a, sig_a, w_a = sorted_means[i], sorted_stds[i], sorted_weights[i]
        mu_b, sig_b, w_b = sorted_means[i + 1], sorted_stds[i + 1], sorted_weights[i + 1]
        # Search for crossover in [mu_a, mu_b]
        x_grid = np.linspace(mu_a, mu_b, 500)
        from scipy.stats import norm
        log_p_a = np.log(w_a + 1e-12) + norm.logpdf(x_grid, mu_a, sig_a + 1e-8)
        log_p_b = np.log(w_b + 1e-12) + norm.logpdf(x_grid, mu_b, sig_b + 1e-8)
        diff = log_p_a - log_p_b
        # Find zero crossing (a > b then b > a)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes):
            thresholds.append(float(x_grid[sign_changes[-1]]))
        else:
            # Fallback: midpoint between means
            thresholds.append(float((mu_a + mu_b) / 2.0))
    return np.array(thresholds)


def _thresholds_youden(
    scores: np.ndarray, labels: np.ndarray, num_classes: int,
) -> np.ndarray:
    """Youden's-J thresholds via one-vs-all ROC analysis.

    For each ordinal boundary (class ≤ c vs > c), finds the threshold
    maximising Youden's J = sensitivity + specificity − 1.
    """
    from sklearn.metrics import roc_curve

    thresholds = []
    for c in range(num_classes - 1):
        binary = (labels > c).astype(int)
        if len(np.unique(binary)) < 2:
            # Degenerate: use midpoint between target scores
            thresholds.append(
                (c + 0.5) * UNIVERSAL_SCALE / max(num_classes - 1, 1)
            )
            continue
        fpr, tpr, thr = roc_curve(binary, scores)
        j = tpr - fpr
        best_idx = int(j.argmax())
        thresholds.append(float(thr[best_idx]))
    return np.sort(np.array(thresholds))


def find_optimal_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    metric: str = "balanced_accuracy",
    method: str = "grid",
    grid_steps: int = 200,
) -> Tuple[np.ndarray, float]:
    """Find K-1 optimal thresholds for discretising continuous scores.

    Parameters
    ----------
    metric : str
        Objective for grid/Nelder-Mead: ``"balanced_accuracy"``, ``"qwk"``,
        ``"accuracy"``, ``"composite"``.
    method : str
        ``"grid"``     – 3-pass coordinate-descent grid search + Nelder-Mead
        (default, most thorough).
        ``"isotonic"`` – isotonic regression → class-transition thresholds.
        ``"gmm"``      – Gaussian mixture model → posterior cross-over points.
        ``"youden"``   – Youden's J on one-vs-all ROC curves.

    Returns
    -------
    thresholds : ndarray of shape (K-1,)
    score : float
        Balanced accuracy (or ``metric`` value) evaluated at those thresholds.
    """
    def _eval(thr: np.ndarray) -> float:
        """Evaluate a set of thresholds using the selected metric."""
        preds = np.searchsorted(np.sort(thr), scores)
        preds = np.clip(preds, 0, num_classes - 1)
        if metric == "qwk":
            return cohen_kappa_score(labels, preds, weights="quadratic")
        if metric == "accuracy":
            return accuracy_score(labels, preds)
        if metric == "composite":
            ba = balanced_accuracy_score(labels, preds)
            qwk = (cohen_kappa_score(labels, preds, weights="quadratic")
                   if num_classes > 2 else cohen_kappa_score(labels, preds))
            mae_norm = mean_absolute_error(labels, preds) / max(num_classes - 1, 1)
            return 0.5 * ba + 0.3 * max(qwk, 0.0) + 0.2 * (1.0 - mae_norm)
        return balanced_accuracy_score(labels, preds)

    # ── dispatch alternative methods ──────────────────────────────────
    if method == "isotonic":
        thresholds = _thresholds_isotonic(scores, labels, num_classes)
        return thresholds, _eval(thresholds)
    if method == "gmm":
        thresholds = _thresholds_gmm(scores, labels, num_classes)
        return thresholds, _eval(thresholds)
    if method == "youden":
        thresholds = _thresholds_youden(scores, labels, num_classes)
        return thresholds, _eval(thresholds)

    # ── grid search (default method) ──────────────────────────────────
    lo, hi = float(scores.min()) - 0.5, float(scores.max()) + 0.5
    thresholds = np.linspace(lo, hi, num_classes + 1)[1:-1]

    # 3-pass coordinate descent with increasing resolution
    for pass_idx in range(3):
        resolution = grid_steps if pass_idx == 0 else grid_steps * 2
        if pass_idx > 0:
            margin = (hi - lo) / (grid_steps * 0.5)
        for k in range(num_classes - 1):
            best_t, best_s = thresholds[k], _eval(thresholds)
            if pass_idx > 0:
                t_lo = max(lo, thresholds[k] - margin)
                t_hi = min(hi, thresholds[k] + margin)
            else:
                t_lo, t_hi = lo, hi
            for t in np.linspace(t_lo, t_hi, resolution):
                trial = thresholds.copy()
                trial[k] = t
                trial = np.sort(trial)
                s = _eval(trial)
                if s > best_s:
                    best_s = s
                    best_t = t
            thresholds[k] = best_t
            thresholds = np.sort(thresholds)

    # Nelder-Mead polish
    try:
        from scipy.optimize import minimize

        def _neg_score(thr_flat: np.ndarray) -> float:
            return -_eval(np.sort(thr_flat))

        result = minimize(
            _neg_score, thresholds, method="Nelder-Mead",
            options={"maxiter": 500 * (num_classes - 1), "xatol": 1e-4,
                     "fatol": 1e-6, "adaptive": True},
        )
        candidate = np.sort(result.x)
        if _eval(candidate) >= _eval(thresholds):
            thresholds = candidate
    except Exception:
        pass

    return thresholds, _eval(thresholds)


# ════════════════════════════════════════════════════════════════════════════
# TEST SET FILTERING: T2w-only + one S1 per patient
# ════════════════════════════════════════════════════════════════════════════


def filter_test_t2_s1_per_patient(dataloader: DataLoader) -> DataLoader:
    """Create a filtered test DataLoader: T2w + first S1 per (patient, level)."""
    ds = dataloader.dataset
    entries = getattr(ds, "entries", None)
    if entries is None:
        logger.warning("Cannot filter: dataset has no 'entries' attribute")
        return dataloader

    seen = set()
    keep_indices = []
    for idx, entry in enumerate(entries):
        if entry.sequence.upper() != "T2":
            continue
        if entry.plane.upper() != "S1":
            continue
        key = (entry.patient_id, entry.level)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(idx)

    if not keep_indices:
        logger.warning("T2+S1 filter -> 0 samples -- falling back to unfiltered")
        return dataloader

    from torch.utils.data import Subset

    filtered_ds = Subset(ds, keep_indices)
    logger.info(
        f"  Test filter: {len(entries)} -> {len(keep_indices)} samples "
        f"(T2w + S1, {len(seen)} unique patient-level combos)"
    )
    return DataLoader(
        filtered_ds,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=False,
    )


# ════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ════════════════════════════════════════════════════════════════════════════

# Legacy key remapping for old classification checkpoints
_LEGACY_HEAD_MAP = {
    "Cen": "clf_ccs", "Spo": "clf_spn", "Upp": "clf_ued", "Low": "clf_led",
    "Her": "clf_hrn", "Ant": "clf_anb", "Pos": "clf_pob", "Ver": "clf_ivd",
    "Pfi": "clf_pf", "Nar": "clf_nar", "CCS": "clf_ccs", "Spn": "clf_spn",
    "UED": "clf_ued", "LED": "clf_led", "UMo": "clf_umc", "LMo": "clf_lmc",
    "FSL": "clf_fsl", "FSR": "clf_fsr", "Hrn": "clf_hrn", "AnB": "clf_anb",
    "PoB": "clf_pob", "AnT": "clf_ant", "FJL": "clf_fjl", "FJR": "clf_fjr",
    "IVD": "clf_ivd", "UMa": "clf_uma", "LMa": "clf_lma", "SpB": "clf_spb",
    "UEB": "clf_ueb", "LEB": "clf_leb", "FLB": "clf_flb", "FRB": "clf_frb",
    "HrB": "clf_hrb", "ABB": "clf_abb", "PBB": "clf_pbb",
}

_LEGACY_RANK_HEAD_MAP = {
    "Pfirrmann": "rank_pf", "Narrowing": "rank_nar",
    "CentralCanalStenosis": "rank_ccs", "Spondylolisthesis": "rank_spn",
    "UpperEndplateDefect": "rank_ued", "LowerEndplateDefect": "rank_led",
    "UpperModic": "rank_umc", "LowerModic": "rank_lmc",
    "ForaminalStenosisLeft": "rank_fsl", "ForaminalStenosisRight": "rank_fsr",
    "Herniation": "rank_hrn", "AnteriorBulging": "rank_anb",
    "PosteriorBulging": "rank_pob", "AnnularTears": "rank_ant",
    "FacetJointArthropathyLeft": "rank_fjl", "FacetJointArthropathyRight": "rank_fjr",
    "UpperMarrow": "rank_uma", "LowerMarrow": "rank_lma",
    "SpondylolisthesisBinary": "rank_spb",
    "UpperEndplateDefectBinary": "rank_ueb", "LowerEndplateDefectBinary": "rank_leb",
    "ForaminalStenosisLeftBinary": "rank_flb", "ForaminalStenosisRightBinary": "rank_frb",
    "HerniationBinary": "rank_hrb",
    "AnteriorBulgingBinary": "rank_abb", "PosteriorBulgingBinary": "rank_pbb",
}


def _remap_legacy_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap old head abbreviations -> new clf_*/rank_* naming scheme."""
    remapped = {}
    for k, v in state.items():
        new_k = k
        for old, new in _LEGACY_HEAD_MAP.items():
            new_k = new_k.replace(f".{old}.", f".{new}.")
        for old, new in _LEGACY_RANK_HEAD_MAP.items():
            new_k = new_k.replace(f"ranking_heads.{old}.", f"ranking_heads.{new}.")
        remapped[new_k] = v
    return remapped


def build_model(
    config: HybridConfig,
    tasks: List[str],
    skip_checkpoint: bool = False,
) -> Tuple[nn.Module, int, float]:
    """Build MultiHeadGradingRanker, optionally resume from checkpoint.

    Supports:
      - Full checkpoint loading (all weights)
      - Encoder-only loading (``--encoder_only``): re-init clf + ranking heads
      - Encoder + clf heads loading (``--encoder_only --rank_from_logits``):
        keeps clf heads, re-inits only ranking heads
    """
    in_channels = 1
    if getattr(config, "USE_MULTI_CONTRAST", False):
        in_channels = len(getattr(config, "CONTRASTS", ["T1", "T2"]))

    model = build_grading_ranker(
        arch=config.MODEL_TYPE,
        tasks=tasks,
        task_defs=TASK_DEFINITIONS,
        task_loss_override=getattr(config, "TASK_LOSS_OVERRIDE", {}),
        head_type=getattr(config, "HEAD_TYPE", "linear"),
        hidden_dim=getattr(config, "HEAD_HIDDEN_DIM", 256),
        ranking_head_type=getattr(config, "RANKING_HEAD_TYPE", "linear"),
        rank_hidden_dim=getattr(config, "RANK_HIDDEN_DIM", 128),
        rank_dropout=getattr(config, "RANK_DROPOUT", 0.3),
        bounded=getattr(config, "BOUNDED", True),
        bounding_mode=getattr(config, "BOUNDING_MODE", "softplus"),
        rank_from_logits=getattr(config, "RANK_FROM_LOGITS", False),
        rank_from_concat_logits=getattr(config, "RANK_FROM_CONCAT_LOGITS", False),
        cross_task_mode=getattr(config, "CROSS_TASK_MODE", "none"),
        cross_task_d_model=getattr(config, "CROSS_TASK_D_MODEL", 64),
        cross_task_nhead=getattr(config, "CROSS_TASK_NHEAD", 4),
        cross_task_num_layers=getattr(config, "CROSS_TASK_NUM_LAYERS", 2),
        cross_task_dim_ff=getattr(config, "CROSS_TASK_DIM_FF", 128),
        cross_task_dropout=getattr(config, "CROSS_TASK_DROPOUT", 0.1),
        cross_task_store_attn=getattr(config, "CROSS_TASK_STORE_ATTN", False),
        in_channels=in_channels,
    )

    model = nn.DataParallel(model)
    model.cuda()

    epoch_start = 1
    best_metric = 0.0

    encoder_only = getattr(config, "ENCODER_ONLY", False)
    rank_from_logits = getattr(config, "RANK_FROM_LOGITS", False)
    rank_from_concat = getattr(config, "RANK_FROM_CONCAT_LOGITS", False)
    ckpt_path = getattr(config, "CHECKPOINT", "")

    if encoder_only and (rank_from_logits or rank_from_concat):
        logger.info(
            "NOTE: --encoder_only + logit-based ranking: "
            "loading encoder + clf heads, re-init ranking heads only."
        )

    if ckpt_path and os.path.isfile(ckpt_path) and not skip_checkpoint:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        # Helper: check if a key belongs to ranking heads (per-task or concat)
        def _is_ranking_key(k: str) -> bool:
            return "ranking_heads" in k or "_concat_ranking_head" in k

        load_enc_clf = encoder_only and (rank_from_logits or rank_from_concat)

        if "backbone" in ckpt and "ranking_heads" in ckpt:
            # Old hybrid format
            if encoder_only and not (rank_from_logits or rank_from_concat):
                enc_state = {
                    k: v for k, v in ckpt["backbone"].items() if "encoder" in k
                }
                model.load_state_dict(enc_state, strict=False)
                logger.info(f"Loaded ENCODER-ONLY from old hybrid ckpt: {ckpt_path}")
            elif load_enc_clf:
                backbone_state = ckpt["backbone"]
                model_state = model.state_dict()
                filtered = {
                    k: v
                    for k, v in backbone_state.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                model.load_state_dict(filtered, strict=False)
                logger.info(
                    f"Loaded encoder + clf heads from old hybrid ckpt "
                    f"({len(filtered)} loaded): {ckpt_path}"
                )
            else:
                merged = dict(ckpt["backbone"])
                for k, v in ckpt["ranking_heads"].items():
                    merged[f"ranking_heads.{k}"] = v
                merged = _remap_legacy_keys(merged)
                model.load_state_dict(merged, strict=False)
                logger.info(f"Loaded old hybrid checkpoint: {ckpt_path}")
        else:
            # New unified / clf-only format
            state = ckpt.get("model_weights", ckpt)
            state = _remap_legacy_keys(state)

            if encoder_only and not (rank_from_logits or rank_from_concat):
                enc_state = {k: v for k, v in state.items() if "encoder" in k}
                model.load_state_dict(enc_state, strict=False)
                logger.info(
                    f"Loaded ENCODER-ONLY from {ckpt_path} "
                    f"({len(enc_state)} params)"
                )
            elif load_enc_clf:
                model_state = model.state_dict()
                filtered = {
                    k: v
                    for k, v in state.items()
                    if k in model_state
                    and v.shape == model_state[k].shape
                    and not _is_ranking_key(k)
                }
                model.load_state_dict(filtered, strict=False)
                logger.info(
                    f"Loaded encoder + clf heads from {ckpt_path} "
                    f"({len(filtered)} loaded, ranking heads re-init)"
                )
            else:
                model_state = model.state_dict()
                filtered = {
                    k: v
                    for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                model.load_state_dict(filtered, strict=False)
                logger.info(
                    f"Loaded checkpoint from {ckpt_path} "
                    f"({len(filtered)}/{len(state)} params)"
                )

        best_metric = float(ckpt.get("acc", ckpt.get("best_metric", 0.0)))
        epoch_start = int(ckpt.get("epoch_no", ckpt.get("epoch", 0))) + 1
        logger.info(f"  Checkpoint epoch {epoch_start - 1}, metric={best_metric:.3f}")

        if encoder_only:
            # When loading only encoder (± clf heads) for fine-tuning, reset
            # the epoch counter — this is a NEW training run, not a resume.
            logger.info("  --encoder_only: resetting epoch_start=1, best_metric=0")
            epoch_start = 1
            best_metric = 0.0

    return model, epoch_start, best_metric


# ════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _loss_still_decreasing(history: dict, window: int = 20) -> bool:
    """Check if smoothed training loss is still trending downward.

    Compares the mean loss over the most recent *window* epochs against
    the previous *window* epochs.  Returns ``True`` (keep training) when
    there is not enough data yet (< 2·window epochs).
    """
    losses = history.get("train_loss", [])
    if len(losses) < 2 * window:
        return True  # not enough data — be conservative, keep training
    recent = float(np.mean(losses[-window:]))
    previous = float(np.mean(losses[-2 * window : -window]))
    return recent < previous


# ════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP  (hybrid: classification + ranking + triplet)
# ════════════════════════════════════════════════════════════════════════════


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    tasks: List[str],
    config: HybridConfig,
    class_weights: Dict[str, Optional[torch.Tensor]],
    ranking_loss_fn: Any,
    triplet_loss_fn: Optional[TripletMarginOrdinalLoss],
    epoch: int,
    task_weighting: Optional[Any] = None,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Train one epoch. Returns (total_loss, clf_loss, rank_loss, per_task_clf_losses)."""
    model.train()
    total_loss_sum = 0.0
    clf_loss_sum = 0.0
    rank_loss_sum = 0.0
    triplet_loss_sum = 0.0
    agree_loss_sum = 0.0
    task_loss_sums: Dict[str, float] = {t: 0.0 for t in tasks}
    n_batches = 0

    ranking_weight = float(getattr(config, "RANKING_LOSS_WEIGHT", 1.0))
    triplet_weight = float(getattr(config, "TRIPLET_LOSS_WEIGHT", 0.5))
    clf_weight = float(getattr(config, "CLF_LOSS_WEIGHT", 1.0))
    agree_weight = float(getattr(config, "AGREEMENT_LOSS_WEIGHT", 0.0))

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for sample in pbar:
        images = sample["image"].cuda()
        labels = {k: v.cuda().long() for k, v in sample["labels"].items()}

        optimizer.zero_grad()
        clf_out, rank_out, features = model(images)

        # ── classification loss ──────────────────────────────────────
        per_task_clf_losses: Dict[str, torch.Tensor] = {}
        if clf_weight > 0:
            for i, task in enumerate(tasks):
                if i >= len(clf_out) or task not in labels:
                    continue
                loss = compute_clf_loss(
                    clf_out[i], labels[task], task, config, class_weights
                )
                if loss is not None:
                    per_task_clf_losses[task] = loss

        if per_task_clf_losses:
            if task_weighting is not None:
                clf_total = task_weighting.reweight(per_task_clf_losses, epoch=epoch)
            else:
                clf_total = sum(per_task_clf_losses.values()) / max(
                    len(per_task_clf_losses), 1
                )
        else:
            clf_total = torch.tensor(0.0, device="cuda")

        # ── ranking loss (skipped when ranking_weight == 0) ──────────
        rank_total = torch.tensor(0.0, device="cuda")
        n_rank_tasks = 0

        if ranking_weight > 0 and ranking_loss_fn is not None:
            _pointwise = getattr(ranking_loss_fn, "pointwise", False)
            _level_aware = getattr(config, "LEVEL_AWARE_PAIRS", False)
            _batch_levels = labels.get("VertebralLevel", None) if _level_aware else None

            for task in tasks:
                if task not in labels or task not in rank_out:
                    continue
                mask = labels[task] >= 0
                if mask.sum() < 2:
                    continue

                task_scores = rank_out[task][mask]
                task_labels = labels[task][mask]
                task_levels = _batch_levels[mask] if _batch_levels is not None else None
                nc = TASK_DEFINITIONS[task]["num_classes"]

                if _pointwise:
                    scaled_labels = task_labels.float() * (
                        UNIVERSAL_SCALE / max(nc - 1, 1)
                    )
                    rl = ranking_loss_fn(task_scores, scaled_labels)
                else:
                    pair_data = generate_pairs(
                        task_labels,
                        task_scores,
                        max_pairs=getattr(config, "MAX_PAIRS_PER_BATCH", 500),
                        hard_ratio=getattr(config, "HARD_NEGATIVE_RATIO", 0.7),
                        hard_margin=getattr(config, "HARD_MARGIN", 1.0),
                        num_classes=nc,
                        skip_equal=getattr(config, "SKIP_EQUAL_PAIRS", False),
                        level_labels=task_levels,
                    )
                    if pair_data is None:
                        continue

                    pidx, yij, ptypes, sev = pair_data
                    si = task_scores[pidx[:, 0]]
                    sj = task_scores[pidx[:, 1]]
                    rl = ranking_loss_fn(si, sj, yij, ptypes, sev)

                rank_total = rank_total + rl
                n_rank_tasks += 1

            if n_rank_tasks > 0:
                rank_total = rank_total / n_rank_tasks

        # ── cross-task ranking loss (SpineRank cross-task concordance) ──
        ct_weight = float(getattr(config, "CROSS_TASK_LOSS_WEIGHT", 0.0))
        if ct_weight > 0 and ranking_weight > 0:
            ct_max_pairs = int(getattr(config, "CROSS_TASK_MAX_PAIRS", 200))
            ct_loss = cross_task_ranking_loss(
                rank_out, labels, tasks, TASK_DEFINITIONS,
                max_pairs=ct_max_pairs,
            )
            rank_total = rank_total + ct_weight * ct_loss

        # ── triplet loss (skipped when triplet_weight == 0) ──────────
        triplet_total = torch.tensor(0.0, device="cuda")
        if triplet_weight > 0 and triplet_loss_fn is not None:
            n_triplet_tasks = 0
            for task in tasks:
                if task not in labels or task not in rank_out:
                    continue
                nc = TASK_DEFINITIONS[task]["num_classes"]
                mask = (labels[task] >= 0) & (labels[task] < nc)
                if mask.sum() < 3:
                    continue

                task_scores = rank_out[task][mask]
                task_labels = labels[task][mask]

                tl = triplet_loss_fn(task_scores, task_labels)
                triplet_total = triplet_total + tl
                n_triplet_tasks += 1

            if n_triplet_tasks > 0:
                triplet_total = triplet_total / n_triplet_tasks

        # ── agreement regularization (rank ↔ clf consistency) ────────
        agree_total = torch.tensor(0.0, device="cuda")
        if agree_weight > 0:
            n_agree = 0
            for i, task in enumerate(tasks):
                if task not in labels or task not in rank_out or i >= len(clf_out):
                    continue
                mask = labels[task] >= 0
                if mask.sum() < 1:
                    continue
                nc = TASK_DEFINITIONS[task]["num_classes"]
                al = compute_agreement_loss(
                    rank_out[task][mask], clf_out[i][mask],
                    nc, detach_clf=getattr(config, "AGREEMENT_DETACH_CLF", True),
                )
                agree_total = agree_total + al
                n_agree += 1
            if n_agree > 0:
                agree_total = agree_total / n_agree

        # ── combined loss ────────────────────────────────────────────
        total = (
            clf_weight * clf_total
            + ranking_weight * rank_total
            + triplet_weight * triplet_total
            + agree_weight * agree_total
        )

        if total.requires_grad:
            total.backward()
            optimizer.step()

        total_loss_sum += total.item()
        clf_loss_sum += clf_total.item()
        rank_loss_sum += rank_total.item()
        triplet_loss_sum += triplet_total.item()
        agree_loss_sum += agree_total.item()
        for t, l in per_task_clf_losses.items():
            task_loss_sums[t] += l.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total.item():.4f}",
            clf=f"{clf_total.item():.3f}",
            rank=f"{rank_total.item():.3f}",
        )

    n = max(n_batches, 1)
    avg_total = total_loss_sum / n
    avg_clf = clf_loss_sum / n
    avg_rank = rank_loss_sum / n
    avg_task = {t: v / n for t, v in task_loss_sums.items()}

    return avg_total, avg_clf, avg_rank, avg_task


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION  (classification + ranking balanced accuracy)
# ════════════════════════════════════════════════════════════════════════════


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    tasks: List[str],
    config: HybridConfig,
    epoch: int,
    set_type: str = "Val",
    binning_mode: str = "target_aligned",
) -> Tuple[float, Dict[str, float], float, Dict[str, float],
           Dict[str, Dict[str, float]]]:
    """Validate classification + ranking performance.

    Returns
    -------
    clf_overall : float
        Mean classification balanced accuracy (%).
    clf_per_task : dict
        Per-task classification balanced accuracy (%).
    rank_overall : float
        Mean ranking balanced accuracy (%).
    rank_per_task : dict
        Per-task ranking balanced accuracy (%).
    rank_extra : dict
        Per-task ranking extras: ``{task: {"qwk": ..., "spearman": ...,
        "kendall": ...}}``.  Also includes ``"__mean__"`` key with
        aggregated means.
    """
    model.eval()
    clf_preds: Dict[str, list] = {t: [] for t in tasks}
    clf_labels: Dict[str, list] = {t: [] for t in tasks}
    rank_scores: Dict[str, list] = {t: [] for t in tasks}
    rank_labels: Dict[str, list] = {t: [] for t in tasks}

    with torch.no_grad():
        for sample in tqdm(loader, desc=f"Eval {set_type}", leave=False):
            images = sample["image"].cuda()
            labels = {k: v.cuda().long() for k, v in sample["labels"].items()}

            clf_out, rank_out, _ = model(images)

            for i, task in enumerate(tasks):
                if i >= len(clf_out) or task not in labels:
                    continue
                preds = get_task_predictions(clf_out[i], task, config)
                clf_preds[task].extend(preds.cpu().numpy().tolist())
                clf_labels[task].extend(labels[task].cpu().numpy().tolist())

                if task in rank_out:
                    mask = labels[task] >= 0
                    if mask.sum() > 0:
                        rank_scores[task].extend(
                            rank_out[task][mask].cpu().numpy().tolist()
                        )
                        rank_labels[task].extend(
                            labels[task][mask].cpu().numpy().tolist()
                        )

    # classification balanced accuracy
    clf_acc_dict: Dict[str, float] = {}
    for task in tasks:
        if not clf_preds[task]:
            continue
        y_true = np.array(clf_labels[task])
        y_pred = np.array(clf_preds[task])
        valid = (y_true != INVALID_LABEL) & (y_true >= 0)
        if not valid.any():
            continue
        acc = balanced_accuracy_score(y_true[valid], y_pred[valid]) * 100.0
        clf_acc_dict[task] = acc

    clf_overall = (
        float(np.mean(list(clf_acc_dict.values()))) if clf_acc_dict else 0.0
    )

    # ranking balanced accuracy (scores on universal [0, SCALE] -> class indices)
    rank_acc_dict: Dict[str, float] = {}
    rank_extra: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        if not rank_scores[task]:
            continue
        nc = TASK_DEFINITIONS[task]["num_classes"]
        s = np.array(rank_scores[task])
        labs = np.array(rank_labels[task])
        preds = discretize_ranking_scores(s, labs, nc, mode=binning_mode)
        acc = balanced_accuracy_score(labs, preds) * 100.0
        rank_acc_dict[task] = acc

        # ── ranking-specific metrics for model selection ──
        qwk = (
            cohen_kappa_score(labs, preds, weights="quadratic")
            if nc > 2
            else cohen_kappa_score(labs, preds)
        )
        sp_r, _ = spearmanr(labs, s) if len(np.unique(labs)) > 1 else (0.0, 1.0)
        kt_t, _ = kendalltau(labs, s) if len(np.unique(labs)) > 1 else (0.0, 1.0)
        mae_val = float(np.mean(np.abs(labs - preds)))

        # ── ROC-AUC (multiclass, one-vs-rest) ──
        try:
            targets = np.array(
                [c * UNIVERSAL_SCALE / max(nc - 1, 1) for c in range(nc)]
            )
            dists = np.abs(s[:, None] - targets[None, :])
            probs = np.exp(-dists) / np.exp(-dists).sum(axis=1, keepdims=True)
            roc_auc = float(roc_auc_score(
                labs, probs, multi_class="ovr", average="macro",
            ))
        except (ValueError, IndexError):
            roc_auc = 0.0

        # ── MCC (multiclass Matthews correlation) ──
        try:
            mcc = float(matthews_corrcoef(labs, preds))
        except ValueError:
            mcc = 0.0

        rank_extra[task] = {
            "qwk": float(qwk),
            "spearman": float(sp_r),
            "kendall": float(kt_t),
            "mae": mae_val,
            "roc_auc": roc_auc,
            "mcc": mcc,
        }

    rank_overall = (
        float(np.mean(list(rank_acc_dict.values()))) if rank_acc_dict else 0.0
    )

    # aggregate means across tasks (exclude __mean__ key itself)
    if rank_extra:
        task_extras = {t: v for t, v in rank_extra.items()
                       if not t.startswith("__")}
        rank_extra["__mean__"] = {
            k: float(np.mean([v[k] for v in task_extras.values() if k in v]))
            for k in ["qwk", "spearman", "kendall", "mae", "roc_auc", "mcc"]
        }

    return clf_overall, clf_acc_dict, rank_overall, rank_acc_dict, rank_extra


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY ROC CURVE (macro OVR)
# ════════════════════════════════════════════════════════════════════════════


def plot_summary_roc_curve(
    results: Dict[str, Dict[str, Any]],
    class_names_dict: Dict[str, List[str]],
    out_dir: Path,
    model_name: str,
    title_suffix: str = "",
) -> Optional[Dict[str, float]]:
    """Summary ROC: one macro-OVR curve per task + grand average.

    Parameters
    ----------
    results : dict
        ``{task: {"targets": list[int], "probs": list[np.ndarray]}}``
    class_names_dict : dict
        ``{task: [class_name, ...]}``
    out_dir : Path
        Directory for the PNG output.
    model_name : str
        Prefix for the output filename.
    title_suffix : str, optional
        Extra text appended to the plot title.

    Returns
    -------
    dict or None
        ``{task: macro_auc, ...}`` if any tasks were plotted, else *None*.
    """
    from sklearn.metrics import roc_curve, auc as sk_auc

    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    task_colors = list(plt.cm.tab20.colors)
    all_mean_tpr: List[np.ndarray] = []
    mean_fpr = np.linspace(0, 1, 200)
    task_aucs: List[float] = []
    auc_dict: Dict[str, float] = {}

    for i, (task, data) in enumerate(results.items()):
        if not data["probs"]:
            continue

        y_true = np.array(data["targets"])
        y_probs = np.vstack(data["probs"])
        valid_mask = y_true >= 0
        if not valid_mask.any():
            continue
        y_true = y_true[valid_mask]
        y_probs = y_probs[valid_mask]

        n_classes = y_probs.shape[1]
        if len(np.unique(y_true)) < 2:
            continue

        # Macro-average OVR ROC for this task
        task_tprs: List[np.ndarray] = []
        task_auc_sum = 0.0
        n_valid_classes = 0
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
                continue
            fpr_c, tpr_c, _ = roc_curve(y_bin, y_probs[:, c])
            auc_c = sk_auc(fpr_c, tpr_c)
            tpr_interp = np.interp(mean_fpr, fpr_c, tpr_c)
            tpr_interp[0] = 0.0
            task_tprs.append(tpr_interp)
            task_auc_sum += auc_c
            n_valid_classes += 1

        if n_valid_classes == 0:
            continue

        mean_tpr_task = np.mean(task_tprs, axis=0)
        mean_tpr_task[-1] = 1.0
        macro_auc = task_auc_sum / n_valid_classes

        color = task_colors[i % len(task_colors)]
        short_name = task.replace("Ordinal", "").replace("Binary", "")
        ax.plot(
            mean_fpr, mean_tpr_task, lw=1.5, alpha=0.65, color=color,
            label=f"{short_name} ({macro_auc:.2f})",
        )

        all_mean_tpr.append(mean_tpr_task)
        task_aucs.append(macro_auc)
        auc_dict[task] = macro_auc

    # Grand macro-average across tasks
    if all_mean_tpr:
        grand_mean_tpr = np.mean(all_mean_tpr, axis=0)
        grand_mean_tpr[-1] = 1.0
        grand_auc = np.mean(task_aucs)
        grand_std = np.std(task_aucs)

        ax.plot(
            mean_fpr, grand_mean_tpr, lw=3.5, color=ETH_COLORS["petrol"],
            label=f"Mean ({grand_auc:.3f} ± {grand_std:.3f})",
            zorder=10,
        )
        ax.fill_between(
            mean_fpr, grand_mean_tpr, alpha=0.08, color=ETH_COLORS["petrol"],
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.4, label="Random")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    title = "Summary ROC — All Tasks (macro OVR)"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.legend(
        fontsize=8, loc="lower right", ncol=2, framealpha=0.9,
        handlelength=1.5, columnspacing=0.8,
    )
    ax.grid(alpha=0.2, linestyle=":", color=ETH_COLORS["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{model_name}_summary_roc.png", bbox_inches="tight", dpi=300,
    )
    plt.close(fig)

    if task_aucs:
        logger.info(
            f"  Summary ROC ({model_name}): Mean AUC = {np.mean(task_aucs):.4f} "
            f"± {np.std(task_aucs):.4f} across {len(task_aucs)} tasks"
        )
        return auc_dict
    return None


# ════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE EVALUATION
# ════════════════════════════════════════════════════════════════════════════


def comprehensive_evaluation(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    tasks: List[str],
    config: HybridConfig,
    out_dir: Path,
    set_type: str = "Test",
    scores_cache: str = "",
    dataset_cache: str = "",
    save_dataset_cache: bool = False,
    binarize_tasks: Optional[set] = None,
    show_threshold_dashes: bool = False,
    heatmap_metric: str = "bal_acc",
) -> Tuple[float, float]:
    """Full evaluation: classification + ranking metrics, plots, CSVs.

    Applies T2+S1 per-patient filtering when ``config.TEST_T2_S1_ONLY``
    is True (default).

    Parameters
    ----------
    scores_cache : str
        Path to a pickle file for caching raw model outputs (scores/labels/
        probs).  If the file exists, inference is skipped and results are
        loaded from cache — enabling fast re-evaluation with different
        threshold methods.  If empty string, caching is disabled.
    dataset_cache : str
        Path to pre-cached dataset tensors. If exists, skips NIFTI loading.
    save_dataset_cache : bool
        Whether to save preprocessed dataset to cache on first run.
    binarize_tasks : Optional[set]
        Task names to EXCLUDE from binarization. Others will be binarized.
    show_threshold_dashes : bool
        Overlay threshold dashes on boxplots.
    heatmap_metric : str
        Metric for threshold method heatmap visualization.

    Returns (clf_overall_balacc, rank_overall_balacc).
    """
    if binarize_tasks is None:
        binarize_tasks = set()
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply T2+S1 per-patient filter
    test_t2_s1 = getattr(config, "TEST_T2_S1_ONLY", True)
    if test_t2_s1 and set_type in loaders:
        loaders = dict(loaders)  # don't mutate caller's dict
        loaders[set_type] = filter_test_t2_s1_per_patient(loaders[set_type])

    # Containers
    rank_results: Dict[str, Dict[str, list]] = {
        t: {"scores": [], "labels": [], "ids": []} for t in tasks
    }
    clf_results: Dict[str, Dict[str, list]] = {
        t: {"preds": [], "targets": [], "probs": [], "logits": [], "ids": []}
        for t in tasks
    }
    # Center slices for ranked-grid image visualisation (capped for memory)
    # Multi-slice storage: center + lateral for foraminal stenosis
    _VIZ_MAX = 500
    _center_slices: Dict[str, Dict[str, np.ndarray]] = {}

    def _collect(loader: DataLoader, split_name: str) -> None:
        with torch.no_grad():
            for sample in tqdm(loader, desc=f"Collecting {split_name}"):
                sample_ids = sample.get("name", ["unknown"])
                if isinstance(sample_ids, str):
                    sample_ids = [sample_ids]

                labels = {}
                for k, v in sample["labels"].items():
                    t = v.cuda().long()
                    if t.dim() == 0:
                        t = t.unsqueeze(0)
                    labels[k] = t

                _cur_per_view_rank: Dict[str, torch.Tensor] = {}
                if "tta_images" in sample:
                    tta_batch = sample["tta_images"].cuda()
                    if tta_batch.dim() == 4:
                        tta_batch = tta_batch.unsqueeze(1)
                    clf_out, rank_out, _, _cur_per_view_rank = tta_forward_chunked(
                        model, tta_batch, return_per_view=True,
                    )
                else:
                    images = sample["image"].cuda()
                    clf_out, rank_out, _ = model(images)

                for i, task in enumerate(tasks):
                    if task not in labels:
                        continue

                    # Classification
                    if i < len(clf_out) and split_name == set_type:
                        logits = clf_out[i]
                        preds = get_task_predictions(logits, task, config)
                        probs = F.softmax(logits, dim=1)
                        clf_results[task]["preds"].extend(
                            preds.cpu().numpy().tolist()
                        )
                        clf_results[task]["targets"].extend(
                            labels[task].cpu().numpy().tolist()
                        )
                        clf_results[task]["probs"].append(probs.cpu().numpy())
                        clf_results[task]["logits"].append(
                            logits.cpu().numpy()
                        )
                        # Per-sample IDs for downstream per-center / per-site
                        # analyses (parallels rank_results[task]["ids"]).
                        clf_results[task]["ids"].extend(list(sample_ids))

                    # Ranking scores
                    if task in rank_out:
                        mask = labels[task] >= 0
                        if mask.sum() > 0:
                            scores_key = f"scores_{split_name}"
                            labels_key = f"labels_{split_name}"
                            rank_results[task].setdefault(scores_key, []).extend(
                                rank_out[task][mask].cpu().numpy().tolist()
                            )
                            rank_results[task].setdefault(labels_key, []).extend(
                                labels[task][mask].cpu().numpy().tolist()
                            )
                            if split_name == set_type:
                                rank_results[task]["scores"].extend(
                                    rank_out[task][mask].cpu().numpy().tolist()
                                )
                                rank_results[task]["labels"].extend(
                                    labels[task][mask].cpu().numpy().tolist()
                                )
                                rank_results[task]["ids"].extend(
                                    [
                                        n
                                        for n, m in zip(
                                            sample_ids, mask.cpu().numpy()
                                        )
                                        if m
                                    ]
                                )
                                # σ_TTA: std of ranking scores across N_TTA views
                                # Only computed when TTA is active (batch_size=1)
                                if task in _cur_per_view_rank:
                                    pv_np = _cur_per_view_rank[task].cpu().float().numpy()
                                    # pv_np shape: (N_views,) for the single TTA sample
                                    mask_np = mask.cpu().numpy()  # (1,)
                                    for is_valid in mask_np:
                                        if is_valid:
                                            sigma = float(pv_np.std())
                                            rank_results[task].setdefault("sigma_tta", []).append(sigma)

                # Store multi-slice images for ranked-grid visualisation (test only)
                # center + lateral slices for foraminal stenosis tasks
                # NOTE: When TTA is used, sample["image"] contains the CENTER (non-augmented)
                # view extracted at index 13 from the 54-view augmentation grid
                if split_name == set_type and len(_center_slices) < _VIZ_MAX:
                    if "image" not in sample:
                        continue  # Skip if image not available (e.g., eval-only mode)

                    raw_imgs = sample["image"]  # (B, 1, D, H, W) — CENTER view from TTA or regular
                    if raw_imgs.dim() == 5:
                        img_np = raw_imgs[:, 0].cpu().float().numpy()
                    else:
                        img_np = raw_imgs.cpu().float().numpy()

                    for bi in range(img_np.shape[0]):
                        nm = sample_ids[bi] if bi < len(sample_ids) else ""
                        if nm and nm not in _center_slices:
                            D_vol = img_np.shape[1]
                            mid = D_vol // 2
                            left_idx = max(0, D_vol // 4)
                            right_idx = min(D_vol - 1, 3 * D_vol // 4)

                            # Extract slices (ensure no black/NaN values)
                            center_slice = img_np[bi, mid]
                            left_slice = img_np[bi, left_idx]
                            right_slice = img_np[bi, right_idx]

                            # Validate slices (not completely black)
                            if np.max(np.abs(center_slice)) < 1e-6:
                                logger.warning(f"Skipping {nm}: center slice appears to be all zeros/black")
                                continue

                            _center_slices[nm] = {
                                "center": center_slice,
                                "left": left_slice,
                                "right": right_slice,
                            }
                            if len(_center_slices) >= _VIZ_MAX:
                                break

    logger.info(
        f"\n{'=' * 80}\nCOMPREHENSIVE EVALUATION on {set_type}\n{'=' * 80}"
    )

    # ── Scores cache: load if exists, otherwise run inference + save ──────
    _cache_loaded = False
    if scores_cache:
        _cache_path = Path(scores_cache)
        if _cache_path.exists():
            try:
                with open(_cache_path, "rb") as _cf:
                    _cached = pickle.load(_cf)
                rank_results.update(_cached.get("rank_results", {}))
                for t, rd in _cached.get("clf_results", {}).items():
                    clf_results[t].update(rd)
                _center_slices.update(_cached.get("center_slices", {}))
                _cache_loaded = True
                logger.info(f"  [cache] Loaded inference results from {_cache_path}")
            except Exception as _ce:
                logger.warning(f"  [cache] Load failed ({_ce}); re-running inference")

    if not _cache_loaded:
        _collect(loaders["Val"], "Val")
        _collect(loaders[set_type], set_type)

        # Save cache for subsequent threshold-method runs
        if scores_cache:
            _cache_path = Path(scores_cache)
            _cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(_cache_path, "wb") as _cf:
                    pickle.dump({
                        "rank_results": rank_results,
                        "clf_results": {
                            t: {k: v for k, v in r.items() if k != "probs"}
                            | {"probs": r["probs"]}
                            for t, r in clf_results.items()
                        },
                        "center_slices": _center_slices,
                    }, _cf, protocol=4)
                logger.info(f"  [cache] Saved inference results to {_cache_path}")
            except Exception as _ce:
                logger.warning(f"  [cache] Save failed: {_ce}")

    # ════════════════════════════════════════════════════════════════════
    # CLASSIFICATION METRICS
    # ════════════════════════════════════════════════════════════════════

    clf_dir = out_dir / "classification"
    clf_dir.mkdir(exist_ok=True)

    clf_task_metrics = {}
    all_clf_metrics = []
    roc_results_clf = {}

    for task in tasks:
        y_true = np.array(clf_results[task]["targets"])
        y_pred = np.array(clf_results[task]["preds"])
        valid = (y_true != INVALID_LABEL) & (y_true >= 0)
        if not valid.any():
            continue
        y_true_v, y_pred_v = y_true[valid], y_pred[valid]

        nc = TASK_DEFINITIONS[task]["num_classes"]
        cn = CLASS_NAMES_DICT.get(task, [f"C{i}" for i in range(nc)])

        bal_acc = balanced_accuracy_score(y_true_v, y_pred_v)
        acc = accuracy_score(y_true_v, y_pred_v)
        mae_val = mean_absolute_error(y_true_v, y_pred_v)
        qwk = (
            cohen_kappa_score(y_true_v, y_pred_v, weights="quadratic")
            if nc > 2
            else cohen_kappa_score(y_true_v, y_pred_v)
        )

        clf_task_metrics[task] = {
            "balanced_accuracy": float(bal_acc),
            "accuracy": float(acc),
            "mae": float(mae_val),
            "qwk": float(qwk),
            "n_samples": int(valid.sum()),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true_v, y_pred_v)
        plot_confusion_matrix_eth(
            cm,
            cn,
            title=f"{task} (Classification)",
            bal_acc=bal_acc,
            out_path=clf_dir / f"clf_{task}_cm.png",
        )

        # ROC curves
        if clf_results[task]["probs"]:
            probs = np.concatenate(clf_results[task]["probs"], axis=0)
            probs = probs[valid]
            roc_results_clf[task] = {
                "targets": y_true_v.tolist(),
                "probs": [probs[j] for j in range(len(probs))],
            }
            eval_result = compute_comprehensive_metrics(
                y_true_v, y_pred_v, probs, cn
            )
            mdf = eval_result["metrics"]
            mdf["Task"] = task
            all_clf_metrics.append(mdf)
            roc_auc = eval_result.get("macro_roc_auc")
            clf_task_metrics[task]["roc_auc"] = (
                float(roc_auc) if roc_auc is not None else None
            )
            mcc_val = eval_result.get("macro_mcc")
            clf_task_metrics[task]["mcc"] = (
                float(mcc_val) if mcc_val is not None else None
            )

        _clf_mcc = clf_task_metrics[task].get("mcc")
        _clf_mcc_s = f"{_clf_mcc:.4f}" if _clf_mcc is not None else "N/A"
        logger.info(
            f"  CLF  {task:30s}: bal_acc={bal_acc:.4f}  acc={acc:.4f}  "
            f"mae={mae_val:.4f}  qwk={qwk:.4f}  mcc={_clf_mcc_s}"
        )

    # Classification ROC curves (per-task + summary)
    if roc_results_clf:
        plot_roc_curves(roc_results_clf, CLASS_NAMES_DICT, clf_dir, "clf")
        clf_auc_dict = plot_summary_roc_curve(
            roc_results_clf, CLASS_NAMES_DICT, clf_dir, "clf",
            title_suffix="Classification (logits)",
        )

    if all_clf_metrics:
        combined_clf = pd.concat(all_clf_metrics, ignore_index=True)
        combined_clf.to_csv(clf_dir / "clf_all_metrics.csv", index=False)

    # Classification predictions CSV
    clf_pred_dir = clf_dir / "predictions"
    clf_pred_dir.mkdir(exist_ok=True)
    for task in tasks:
        y_true = np.array(clf_results[task]["targets"])
        y_pred = np.array(clf_results[task]["preds"])
        valid = (y_true != INVALID_LABEL) & (y_true >= 0)
        if not valid.any():
            continue
        nc = TASK_DEFINITIONS[task]["num_classes"]
        cn = CLASS_NAMES_DICT.get(task, [f"C{i}" for i in range(nc)])
        # Per-sample IDs (may be empty when the loader didn't provide names;
        # we fall back to row indices to keep the column shape stable).
        ids = clf_results[task].get("ids") or list(range(len(y_true)))
        pred_data = {
            "sample_id": [ids[i] for i in np.where(valid)[0]],
            "true_label": y_true[valid].tolist(),
            "predicted_label": y_pred[valid].tolist(),
        }
        if clf_results[task]["probs"]:
            probs = np.concatenate(clf_results[task]["probs"], axis=0)[valid]
            for c in range(nc):
                pred_data[f"prob_{cn[c]}"] = probs[:, c].tolist()
        pd.DataFrame(pred_data).to_csv(
            clf_pred_dir / f"clf_{task}_predictions.csv", index=False
        )

    # ════════════════════════════════════════════════════════════════════
    # BINARIZED CLASSIFICATION (for SpineNet v2 comparison)
    #
    # Binarize all tasks EXCEPT those in binarize_tasks parameter.
    # Uses clf P(abnormal) = 1 − P(class=0) from softmax for ROC/AUC.
    # ════════════════════════════════════════════════════════════════════
    # Default: keep Pfirrmann, CentralCanalStenosis, Narrowing as ordinal
    if not binarize_tasks:
        binarize_tasks = {"Pfirrmann", "CentralCanalStenosis", "Narrowing"}
    bin_clf_tasks = [t for t in tasks if t not in binarize_tasks]
    logger.info(f"\nBinarizing tasks: {sorted(bin_clf_tasks)}")
    logger.info(f"Keeping ordinal: {sorted(binarize_tasks)}")
    bin_clf_dir = clf_dir / "binarized"
    bin_clf_dir.mkdir(exist_ok=True)
    bin_clf_metrics: Dict[str, Dict] = {}
    bin_clf_roc: Dict[str, Dict] = {}

    for task in bin_clf_tasks:
        y_true = np.array(clf_results[task]["targets"])
        y_pred = np.array(clf_results[task]["preds"])
        valid = (y_true != INVALID_LABEL) & (y_true >= 0)
        if not valid.any():
            continue
        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]
        nc = TASK_DEFINITIONS[task]["num_classes"]

        bin_labels = (y_true_v >= 1).astype(int)
        bin_preds = (y_pred_v >= 1).astype(int)

        p_abnormal = None
        if clf_results[task]["probs"]:
            probs_all = np.concatenate(clf_results[task]["probs"], axis=0)[valid]
            # P(abnormal) = 1 - P(grade=0)
            p_abnormal = 1.0 - probs_all[:, 0]

        ba = balanced_accuracy_score(bin_labels, bin_preds)
        cm_bin = confusion_matrix(bin_labels, bin_preds, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel() if cm_bin.size == 4 else (0, 0, 0, 0)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1_val = f1_score(bin_labels, bin_preds, zero_division=0)

        try:
            bin_roc_auc = (
                float(roc_auc_score(bin_labels, p_abnormal))
                if p_abnormal is not None and len(np.unique(bin_labels)) > 1
                else None
            )
        except ValueError:
            bin_roc_auc = None

        bin_clf_metrics[task] = {
            "binary_bal_acc": float(ba),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1": float(f1_val),
            "binary_roc_auc": bin_roc_auc,
        }

        plot_confusion_matrix_eth(
            cm_bin, ["Normal", "Abnormal"],
            title=f"{task} (Binarized CLF)",
            bal_acc=ba,
            out_path=bin_clf_dir / f"bin_clf_{task}_cm.png",
        )

        if p_abnormal is not None and len(np.unique(bin_labels)) > 1:
            bin_clf_roc[task] = {
                "scores": p_abnormal.tolist(),
                "labels": bin_labels.tolist(),
            }

        _auc_str = f"{bin_roc_auc:.4f}" if bin_roc_auc is not None else "N/A"
        logger.info(
            f"  BIN_CLF  {task:25s}: bal_acc={ba:.4f}  AUC={_auc_str}  "
            f"sens={sens:.4f}  spec={spec:.4f}  f1={f1_val:.4f}"
        )

    if bin_clf_metrics:
        with open(bin_clf_dir / "binarized_clf_metrics.json", "w") as f:
            json.dump(bin_clf_metrics, f, indent=2)

    if bin_clf_roc:
        plot_binarized_summary_roc(
            bin_clf_roc, bin_clf_dir,
            title_suffix="Classification (Binarized)",
        )

    # ════════════════════════════════════════════════════════════════════
    # RANKING METRICS
    # ════════════════════════════════════════════════════════════════════

    rank_dir = out_dir / "ranking"
    rank_dir.mkdir(exist_ok=True)

    # Optimal thresholds on val
    thresholds = {}
    for task in tasks:
        val_scores = rank_results[task].get("scores_Val", [])
        val_labels = rank_results[task].get("labels_Val", [])
        if not val_scores:
            continue
        nc = TASK_DEFINITIONS[task]["num_classes"]
        thr, score = find_optimal_thresholds(
            np.array(val_scores),
            np.array(val_labels),
            nc,
            metric=config.THRESHOLD_METRIC,
            method=getattr(config, "THRESHOLD_METHOD", "grid"),
            grid_steps=config.THRESHOLD_GRID_STEPS,
        )
        thresholds[task] = thr
        logger.info(
            f"  RANK {task:30s}: val_thresholds={np.round(thr, 3)}  "
            f"val_{config.THRESHOLD_METRIC}={score:.4f}"
        )

    # Apply to test
    rank_task_metrics = {}
    roc_results_rank = {}
    all_rank_metrics = []

    for task in tasks:
        if not rank_results[task]["scores"]:
            continue

        nc = TASK_DEFINITIONS[task]["num_classes"]
        cn = CLASS_NAMES_DICT.get(task, [f"C{i}" for i in range(nc)])
        s = np.array(rank_results[task]["scores"])
        l = np.array(rank_results[task]["labels"])

        thr = thresholds.get(task)
        if thr is not None:
            preds = np.searchsorted(np.sort(thr), s)
            preds = np.clip(preds, 0, nc - 1)
        else:
            preds = discretize_ranking_scores(s, l, nc, mode="target_aligned")

        bal_acc = balanced_accuracy_score(l, preds)
        acc = accuracy_score(l, preds)
        mae_val = mean_absolute_error(l, preds)
        qwk = (
            cohen_kappa_score(l, preds, weights="quadratic")
            if nc > 2
            else cohen_kappa_score(l, preds)
        )
        sp_r, _ = spearmanr(l, s) if len(np.unique(l)) > 1 else (0.0, 1.0)
        kt_t, _ = kendalltau(l, s) if len(np.unique(l)) > 1 else (0.0, 1.0)
        c_idx, n_pairs = concordance_index(s, l)

        rank_task_metrics[task] = {
            "balanced_accuracy": float(bal_acc),
            "accuracy": float(acc),
            "mae": float(mae_val),
            "qwk": float(qwk),
            "spearman_rho": float(sp_r),
            "kendall_tau": float(kt_t),
            "concordance_index": float(c_idx),
            "concordance_n_pairs": int(n_pairs),
            "n_samples": len(l),
        }

        # Confusion matrix
        cm = confusion_matrix(l, preds)
        plot_confusion_matrix_eth(
            cm,
            cn,
            title=f"{task} (Ranking)",
            bal_acc=bal_acc,
            out_path=rank_dir / f"rank_{task}_cm.png",
        )

        # Score density + boxplots
        plot_score_density_histogram({task: s}, {task: l}, rank_dir, task, cn)
        plot_score_boxplots({task: s}, {task: l}, rank_dir, task, cn)

        # Ranked example grid
        task_names = rank_results[task].get(
            "ids", [str(i) for i in range(len(s))]
        )
        plot_ranked_example_grid(s, l, task_names, task, cn, rank_dir)

        # ── ranked image grid — 6 examples per IVD level ──────────────
        if _center_slices:
            plot_ranked_image_grid(
                s, l, task_names, task, cn, rank_dir,
                images=_center_slices,
                n_cols=6, n_rows=1,
            )

        # ── per-patient level ranking grid (rows=levels, cols=patients) ──
        if _center_slices:
            _patient_data = group_by_patient(
                task_names, s, l,
                center_slices=_center_slices, task_name=task,
                preds=preds,
            )
            if _patient_data:
                plot_patient_level_ranking_grid(
                    task, cn, _patient_data, rank_dir, n_patients=5,
                )
                # ── appendix: success & failure example grids ─────────────
                plot_success_failure_grids(
                    task, cn, _patient_data, rank_dir, n_examples=10,
                )

        # ── error-magnitude histogram (ordinal tasks only) ──
        plot_error_magnitude_histogram(
            l, preds, task, cn,
            rank_dir / f"rank_{task}_error_magnitude.png",
        )

        # ── threshold sensitivity plot ──
        if thr is not None and nc > 2:
            plot_threshold_search(
                s, l, thr, task, cn,
                rank_dir / f"rank_{task}_threshold_sensitivity.png",
            )

        # Soft probs for ROC (class centres on universal [0, SCALE])
        class_centres = np.linspace(0, UNIVERSAL_SCALE, nc)
        probs = np.zeros((len(s), nc))
        for i in range(len(s)):
            dists = np.abs(s[i] - class_centres)
            probs[i] = np.exp(-dists) / np.exp(-dists).sum()
        roc_results_rank[task] = {
            "targets": l.tolist(),
            "probs": [probs[j] for j in range(len(probs))],
        }

        # Per-task evaluation via evaluation.py
        eval_result = compute_comprehensive_metrics(l, preds, probs, cn)
        mdf = eval_result["metrics"]
        mdf["Task"] = task
        all_rank_metrics.append(mdf)

        rank_roc_auc = eval_result.get("macro_roc_auc")
        rank_task_metrics[task]["roc_auc"] = (
            float(rank_roc_auc) if rank_roc_auc is not None else None
        )
        rank_mcc = eval_result.get("macro_mcc")
        rank_task_metrics[task]["mcc"] = (
            float(rank_mcc) if rank_mcc is not None else None
        )

        # Predictions CSV
        rank_pred_dir = rank_dir / "predictions"
        rank_pred_dir.mkdir(exist_ok=True)
        pred_data = {
            "sample_id": task_names[: len(s)],
            "true_label": l.tolist(),
            "predicted_label": preds.tolist(),
            "raw_score": s.tolist(),
        }
        for c in range(nc):
            pred_data[f"prob_{cn[c]}"] = probs[:, c].tolist()
        pd.DataFrame(pred_data).to_csv(
            rank_pred_dir / f"rank_{task}_predictions.csv", index=False
        )

        _roc_str = (
            f"{rank_roc_auc:.4f}" if rank_roc_auc is not None else "N/A"
        )
        _mcc_str = (
            f"{rank_mcc:.4f}" if rank_mcc is not None else "N/A"
        )
        logger.info(
            f"  RANK {task:30s}: bal_acc={bal_acc:.4f}  roc_auc={_roc_str}  "
            f"mcc={_mcc_str}  qwk={qwk:.4f}  spearman={sp_r:.4f}  "
            f"kendall={kt_t:.4f}"
        )

    # ── binarized ranking evaluation (Normal vs Abnormal) ───────────
    bin_rank_dir = rank_dir / "binarized"
    bin_rank_dir.mkdir(exist_ok=True)
    bin_rank_metrics: Dict[str, Dict] = {}
    bin_rank_preds_all, bin_rank_labels_all = [], []

    for task in tasks:
        if not rank_results[task]["scores"]:
            continue
        nc = TASK_DEFINITIONS[task]["num_classes"]
        if nc <= 2:
            continue

        s = np.array(rank_results[task]["scores"])
        l = np.array(rank_results[task]["labels"])

        thr = thresholds.get(task)
        if thr is not None:
            ord_preds = np.searchsorted(np.sort(thr), s)
            ord_preds = np.clip(ord_preds, 0, nc - 1)
        else:
            ord_preds = discretize_ranking_scores(s, l, nc, mode="target_aligned")

        bin_labels = (l >= 1).astype(int)
        bin_preds = (ord_preds >= 1).astype(int)
        bin_rank_preds_all.append(bin_preds)
        bin_rank_labels_all.append(bin_labels)

        ba = balanced_accuracy_score(bin_labels, bin_preds)
        cm_bin = confusion_matrix(bin_labels, bin_preds, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel() if cm_bin.size == 4 else (0, 0, 0, 0)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = f1_score(bin_labels, bin_preds, zero_division=0)

        # Binary ROC AUC using raw continuous ranking scores
        try:
            bin_roc_auc = float(roc_auc_score(bin_labels, s))
        except ValueError:
            bin_roc_auc = None

        bin_rank_metrics[task] = {
            "binary_bal_acc": float(ba),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1": float(f1),
            "binary_roc_auc": bin_roc_auc,
        }

        plot_confusion_matrix_eth(
            cm_bin, ["Normal", "Abnormal"],
            title=f"{task} (Binarized Ranking)",
            bal_acc=ba,
            out_path=bin_rank_dir / f"bin_rank_{task}_cm.png",
        )
        _auc_str = f"{bin_roc_auc:.4f}" if bin_roc_auc is not None else "N/A"
        logger.info(
            f"  BIN  {task:30s}: bal_acc={ba:.4f}  AUC={_auc_str}  "
            f"sens={sens:.4f}  spec={spec:.4f}  f1={f1:.4f}"
        )

    if bin_rank_preds_all:
        all_bp = np.concatenate(bin_rank_preds_all)
        all_bl = np.concatenate(bin_rank_labels_all)
        agg_cm = confusion_matrix(all_bl, all_bp, labels=[0, 1])
        agg_ba = balanced_accuracy_score(all_bl, all_bp)
        plot_confusion_matrix_eth(
            agg_cm, ["Normal", "Abnormal"],
            title=f"Aggregated Binarized ({len(bin_rank_metrics)} tasks)",
            bal_acc=agg_ba,
            out_path=bin_rank_dir / "bin_rank_aggregated_cm.png",
        )

    if bin_rank_metrics:
        with open(bin_rank_dir / "binarized_ranking_metrics.json", "w") as f:
            json.dump(bin_rank_metrics, f, indent=2)
        # Merge into per-task ranking metrics
        for task, bm in bin_rank_metrics.items():
            if task in rank_task_metrics:
                rank_task_metrics[task]["binary_bal_acc"] = bm["binary_bal_acc"]
                rank_task_metrics[task]["binary_sensitivity"] = bm["sensitivity"]
                rank_task_metrics[task]["binary_specificity"] = bm["specificity"]
                rank_task_metrics[task]["binary_f1"] = bm["f1"]
                if bm.get("binary_roc_auc") is not None:
                    rank_task_metrics[task]["binary_roc_auc"] = bm["binary_roc_auc"]

    # ── per-boundary ordinal binary metrics (grade ≤c vs >c) ─────────
    boundary_dir = rank_dir / "per_boundary"
    boundary_dir.mkdir(exist_ok=True)
    boundary_metrics: Dict[str, Dict] = {}

    for task in tasks:
        if not rank_results[task]["scores"]:
            continue
        nc = TASK_DEFINITIONS[task]["num_classes"]
        if nc <= 2:
            continue

        s = np.array(rank_results[task]["scores"])
        l = np.array(rank_results[task]["labels"])
        task_boundaries: Dict[str, Dict] = {}

        for boundary in range(nc - 1):
            # Binary split: grades <= boundary vs grades > boundary
            bin_labels = (l > boundary).astype(int)
            if len(np.unique(bin_labels)) < 2:
                continue

            # Hard predictions via thresholded ordinal
            thr = thresholds.get(task)
            if thr is not None:
                ord_preds = np.searchsorted(np.sort(thr), s)
                ord_preds = np.clip(ord_preds, 0, nc - 1)
            else:
                ord_preds = discretize_ranking_scores(
                    s, l, nc, mode="target_aligned"
                )
            bin_preds = (ord_preds > boundary).astype(int)

            ba = balanced_accuracy_score(bin_labels, bin_preds)
            cm_bin = confusion_matrix(bin_labels, bin_preds, labels=[0, 1])
            tn, fp, fn, tp = (
                cm_bin.ravel() if cm_bin.size == 4 else (0, 0, 0, 0)
            )
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            b_f1 = f1_score(bin_labels, bin_preds, zero_division=0)

            # ROC AUC using raw continuous ranking scores
            try:
                boundary_auc = float(roc_auc_score(bin_labels, s))
            except ValueError:
                boundary_auc = None

            boundary_key = f"grade_0-{boundary}_vs_{boundary + 1}+"
            task_boundaries[boundary_key] = {
                "boundary": boundary,
                "bal_acc": float(ba),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "f1": float(b_f1),
                "roc_auc": boundary_auc,
                "n_negative": int((bin_labels == 0).sum()),
                "n_positive": int((bin_labels == 1).sum()),
            }

            # Per-boundary confusion matrix
            cn = [f"Grade 0–{boundary}", f"Grade {boundary + 1}+"]
            plot_confusion_matrix_eth(
                cm_bin, cn,
                title=f"{task} (boundary {boundary}/{boundary + 1})",
                bal_acc=ba,
                out_path=boundary_dir / f"boundary_{task}_b{boundary}_cm.png",
            )

            _auc_s = f"{boundary_auc:.4f}" if boundary_auc is not None else "N/A"
            logger.info(
                f"  BOUNDARY {task:25s} b={boundary}: "
                f"AUC={_auc_s}  BA={ba:.4f}  "
                f"sens={sens:.4f}  spec={spec:.4f}"
            )

        if task_boundaries:
            boundary_metrics[task] = task_boundaries

            # Per-boundary ROC curve (all boundaries on one figure)
            plot_per_boundary_roc(
                s, l, nc, task,
                out_path=boundary_dir / f"boundary_{task}_roc.png",
            )

    if boundary_metrics:
        with open(boundary_dir / "per_boundary_metrics.json", "w") as f:
            json.dump(boundary_metrics, f, indent=2)

        # Merge best boundary AUC into rank_task_metrics
        for task, bdata in boundary_metrics.items():
            aucs = [
                v["roc_auc"] for v in bdata.values()
                if v["roc_auc"] is not None
            ]
            if aucs and task in rank_task_metrics:
                rank_task_metrics[task]["best_boundary_auc"] = max(aucs)

    # ROC curves (per-task + summary)
    if roc_results_rank:
        plot_roc_curves(roc_results_rank, CLASS_NAMES_DICT, rank_dir, "rank")
        rank_auc_dict = plot_summary_roc_curve(
            roc_results_rank, CLASS_NAMES_DICT, rank_dir, "rank",
            title_suffix="Ranking (thresholded scores)",
        )

    if all_rank_metrics:
        combined = pd.concat(all_rank_metrics, ignore_index=True)
        combined.to_csv(rank_dir / "rank_all_metrics.csv", index=False)

    # Ranking summary plot
    if rank_task_metrics:
        plot_ranking_summary(rank_task_metrics, rank_dir / "ranking_summary.png")

    # ── Multi-pathology per-patient grid (4 pathologies on same MRI) ──
    if _center_slices and rank_results:
        featured = [t for t in FEATURED_TASKS if t in rank_results
                    and rank_results[t].get("scores")]
        if len(featured) >= 2:
            logger.info("Generating multi-pathology per-patient grid...")
            multi_patient_data = group_by_patient_multitask(
                rank_results, featured, center_slices=_center_slices,
            )
            if multi_patient_data:
                plot_multi_pathology_patient_grid(
                    multi_patient_data, featured,
                    CLASS_NAMES_DICT, rank_dir, n_patients=6,
                )

                # ── NEW: Ranking visualizations (improved layouts) ──
                logger.info("Generating improved ranking visualizations...")

                # Vertical levels visualization
                plot_ranking_by_level_vertical(
                    multi_patient_data, featured[:1] if featured else ["Pfirrmann"],
                    CLASS_NAMES_DICT, rank_dir,
                )

                # Sorted scores visualization
                plot_ranking_scores_sorted(
                    rank_results, featured,
                    CLASS_NAMES_DICT, _center_slices,
                    rank_dir, n_samples=8,
                )

    # ── Horizontal score boxplots (compact paper-ready overview) ──
    if rank_results:
        valid_tasks_for_box = [
            t for t in tasks
            if t in rank_results and rank_results[t].get("scores")
        ]
        if valid_tasks_for_box:
            logger.info("Generating horizontal score boxplots...")
            plot_horizontal_score_boxplots(
                rank_results, valid_tasks_for_box,
                CLASS_NAMES_DICT, rank_dir,
                title="SpineRank Score Distributions by Grade",
                thresholds=thresholds,
            )

    # ════════════════════════════════════════════════════════════════════
    # RANK-CALIBRATED CLASSIFICATION METRICS
    # ════════════════════════════════════════════════════════════════════
    if getattr(config, "USE_RANK_CALIBRATION", False):
        cal_dir = out_dir / "rank_calibrated"
        cal_dir.mkdir(exist_ok=True)
        cal_sigma = getattr(config, "CALIBRATION_SIGMA", 1.0)
        cal_alpha = getattr(config, "CALIBRATION_ALPHA", 0.5)
        all_cal_metrics = []

        logger.info(
            f"\n{'='*80}\nRANK-CALIBRATED CLASSIFICATION "
            f"(σ={cal_sigma}, α={cal_alpha})\n{'='*80}"
        )

        for task in tasks:
            if not clf_results[task]["logits"] or not rank_results[task]["scores"]:
                continue

            logits_all = np.vstack(clf_results[task]["logits"])
            scores_all = np.array(rank_results[task]["scores"])
            targets = np.array(clf_results[task]["targets"])

            n_min = min(len(logits_all), len(scores_all), len(targets))
            if n_min == 0:
                continue
            logits_all = logits_all[:n_min]
            scores_all = scores_all[:n_min]
            targets = targets[:n_min]

            valid = (targets != INVALID_LABEL) & (targets >= 0)
            if not valid.any():
                continue

            nc = TASK_DEFINITIONS[task]["num_classes"]
            cn = CLASS_NAMES_DICT.get(task, [f"C{i}" for i in range(nc)])

            fused_probs, fused_preds = rank_calibrated_classify(
                torch.from_numpy(logits_all[valid]).float(),
                torch.from_numpy(scores_all[valid]).float(),
                nc, sigma=cal_sigma, alpha=cal_alpha,
            )
            fused_probs_np = fused_probs.numpy()
            fused_preds_np = fused_preds.numpy()
            y_true = targets[valid]

            bal_acc = balanced_accuracy_score(y_true, fused_preds_np)
            acc = accuracy_score(y_true, fused_preds_np)
            mae_val = mean_absolute_error(y_true, fused_preds_np)
            qwk = (
                cohen_kappa_score(y_true, fused_preds_np, weights="quadratic")
                if nc > 2
                else cohen_kappa_score(y_true, fused_preds_np)
            )

            logger.info(
                f"  CAL {task:30s}: bal_acc={bal_acc:.4f}  acc={acc:.4f}  "
                f"mae={mae_val:.4f}  qwk={qwk:.4f}"
            )

            cm = confusion_matrix(y_true, fused_preds_np)
            plot_confusion_matrix_eth(
                cm, cn,
                title=f"{task} (Rank-Calibrated)",
                bal_acc=bal_acc,
                out_path=cal_dir / f"cal_{task}_cm.png",
            )

            eval_result = compute_comprehensive_metrics(
                y_true, fused_preds_np, fused_probs_np, cn,
            )
            mdf = eval_result["metrics"]
            mdf["Task"] = task
            all_cal_metrics.append(mdf)

            cal_pred_dir = cal_dir / "predictions"
            cal_pred_dir.mkdir(exist_ok=True)
            pred_data = {
                "sample_id": rank_results[task].get(
                    "ids", list(range(n_min))
                )[:len(y_true)],
                "true_label": y_true.tolist(),
                "predicted_label": fused_preds_np.tolist(),
            }
            for c in range(nc):
                col = f"prob_{cn[c]}" if c < len(cn) else f"prob_c{c}"
                pred_data[col] = fused_probs_np[:, c].tolist()
            pd.DataFrame(pred_data).to_csv(
                cal_pred_dir / f"cal_{task}_predictions.csv", index=False,
            )

        if all_cal_metrics:
            combined_cal = pd.concat(all_cal_metrics, ignore_index=True)
            combined_cal.to_csv(cal_dir / "cal_all_metrics.csv", index=False)

    # ════════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════════

    logger.info(f"\n{'=' * 140}")
    logger.info("CLASSIFICATION SUMMARY")
    logger.info(
        f"{'Task':<30} {'N':>6} {'BalAcc':>8} {'ROC_AUC':>9} {'MCC':>8} "
        f"{'Acc':>8} {'MAE':>8} {'QWK':>8}"
    )
    logger.info("-" * 140)
    for task in sorted(clf_task_metrics):
        m = clf_task_metrics[task]
        _rauc = m.get("roc_auc")
        _rauc_s = f"{_rauc:>9.4f}" if _rauc is not None else f"{'N/A':>9}"
        _mcc = m.get("mcc")
        _mcc_s = f"{_mcc:>8.4f}" if _mcc is not None else f"{'N/A':>8}"
        logger.info(
            f"{task:<30} {m['n_samples']:>6} {m['balanced_accuracy']:>8.4f} "
            f"{_rauc_s} {_mcc_s} {m['accuracy']:>8.4f} {m['mae']:>8.4f} "
            f"{m['qwk']:>8.4f}"
        )
    if clf_task_metrics:
        valid_roc = [
            m["roc_auc"]
            for m in clf_task_metrics.values()
            if m.get("roc_auc") is not None
        ]
        avg_roc = float(np.mean(valid_roc)) if valid_roc else None
        valid_mcc = [
            m["mcc"]
            for m in clf_task_metrics.values()
            if m.get("mcc") is not None
        ]
        avg_mcc = float(np.mean(valid_mcc)) if valid_mcc else None
        clf_avgs = {
            k: float(np.mean([m[k] for m in clf_task_metrics.values()]))
            for k in ["balanced_accuracy", "accuracy", "mae", "qwk"]
        }
        clf_avgs["roc_auc"] = avg_roc
        clf_avgs["mcc"] = avg_mcc
        _avg_rauc_s = (
            f"{avg_roc:>9.4f}" if avg_roc is not None else f"{'N/A':>9}"
        )
        _avg_mcc_s = (
            f"{avg_mcc:>8.4f}" if avg_mcc is not None else f"{'N/A':>8}"
        )
        logger.info("-" * 140)
        logger.info(
            f"{'CLF AVERAGE':<30} {'':>6} {clf_avgs['balanced_accuracy']:>8.4f} "
            f"{_avg_rauc_s} {_avg_mcc_s} {clf_avgs['accuracy']:>8.4f} "
            f"{clf_avgs['mae']:>8.4f} {clf_avgs['qwk']:>8.4f}"
        )

    logger.info(f"\n{'=' * 140}")
    logger.info("RANKING SUMMARY")
    logger.info(
        f"{'Task':<30} {'N':>6} {'BalAcc':>8} {'ROC_AUC':>9} {'MCC':>8} "
        f"{'Acc':>8} {'MAE':>8} {'QWK':>8} {'Spearman':>10} {'Kendall':>10} "
        f"{'C-index':>9}"
    )
    logger.info("-" * 150)
    for task in sorted(rank_task_metrics):
        m = rank_task_metrics[task]
        _rauc = m.get("roc_auc")
        _rauc_s = f"{_rauc:>9.4f}" if _rauc is not None else f"{'N/A':>9}"
        _mcc = m.get("mcc")
        _mcc_s = f"{_mcc:>8.4f}" if _mcc is not None else f"{'N/A':>8}"
        _cidx = m.get("concordance_index")
        _cidx_s = f"{_cidx:>9.4f}" if _cidx is not None else f"{'N/A':>9}"
        logger.info(
            f"{task:<30} {m['n_samples']:>6} {m['balanced_accuracy']:>8.4f} "
            f"{_rauc_s} {_mcc_s} {m['accuracy']:>8.4f} {m['mae']:>8.4f} "
            f"{m['qwk']:>8.4f} {m['spearman_rho']:>10.4f} "
            f"{m['kendall_tau']:>10.4f} {_cidx_s}"
        )
    if rank_task_metrics:
        valid_roc = [
            m["roc_auc"]
            for m in rank_task_metrics.values()
            if m.get("roc_auc") is not None
        ]
        avg_roc = float(np.mean(valid_roc)) if valid_roc else None
        valid_mcc = [
            m["mcc"]
            for m in rank_task_metrics.values()
            if m.get("mcc") is not None
        ]
        avg_mcc = float(np.mean(valid_mcc)) if valid_mcc else None
        rank_avgs = {
            k: float(np.mean([m[k] for m in rank_task_metrics.values()]))
            for k in [
                "balanced_accuracy", "accuracy", "mae", "qwk",
                "spearman_rho", "kendall_tau", "concordance_index",
            ]
        }
        rank_avgs["roc_auc"] = avg_roc
        rank_avgs["mcc"] = avg_mcc
        _avg_rauc_s = (
            f"{avg_roc:>9.4f}" if avg_roc is not None else f"{'N/A':>9}"
        )
        _avg_mcc_s = (
            f"{avg_mcc:>8.4f}" if avg_mcc is not None else f"{'N/A':>8}"
        )
        logger.info("-" * 150)
        logger.info(
            f"{'RANK AVERAGE':<30} {'':>6} {rank_avgs['balanced_accuracy']:>8.4f} "
            f"{_avg_rauc_s} {_avg_mcc_s} {rank_avgs['accuracy']:>8.4f} "
            f"{rank_avgs['mae']:>8.4f} {rank_avgs['qwk']:>8.4f} "
            f"{rank_avgs['spearman_rho']:>10.4f} "
            f"{rank_avgs['kendall_tau']:>10.4f} "
            f"{rank_avgs['concordance_index']:>9.4f}"
        )
    logger.info("=" * 150)

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY ROC-AUC COMPARISON (classification vs ranking)
    # ════════════════════════════════════════════════════════════════════
    clf_roc_vals = [
        m["roc_auc"] for m in clf_task_metrics.values()
        if m.get("roc_auc") is not None
    ]
    rank_roc_vals = [
        m["roc_auc"] for m in rank_task_metrics.values()
        if m.get("roc_auc") is not None
    ]
    if clf_roc_vals or rank_roc_vals:
        logger.info(f"\n{'=' * 80}")
        logger.info("SUMMARY ROC-AUC  (Classification logits  vs  Ranking scores)")
        logger.info("-" * 80)
        if clf_roc_vals:
            logger.info(
                f"  Classification (logits):  Mean AUC = {np.mean(clf_roc_vals):.4f} "
                f"± {np.std(clf_roc_vals):.4f}  ({len(clf_roc_vals)} tasks)"
            )
        if rank_roc_vals:
            logger.info(
                f"  Ranking (thresholded):    Mean AUC = {np.mean(rank_roc_vals):.4f} "
                f"± {np.std(rank_roc_vals):.4f}  ({len(rank_roc_vals)} tasks)"
            )
        logger.info("=" * 80)

    # Save JSON
    combined_metrics = {
        "classification": clf_task_metrics,
        "ranking": rank_task_metrics,
    }
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)
    # Also save ranking-only as hybrid_metrics.json (compat with aggregation)
    with open(out_dir / "hybrid_metrics.json", "w") as f:
        json.dump(rank_task_metrics, f, indent=2)

    clf_overall = (
        float(np.mean([m["balanced_accuracy"] for m in clf_task_metrics.values()]))
        if clf_task_metrics
        else 0.0
    )
    rank_overall = (
        float(np.mean([m["balanced_accuracy"] for m in rank_task_metrics.values()]))
        if rank_task_metrics
        else 0.0
    )
    logger.info(
        f"\nClassification BalAcc (avg): {clf_overall:.4f}"
        f" | Ranking BalAcc (avg): {rank_overall:.4f}"
    )

    # ── Save visualization cache (allows re-generating plots without re-eval) ──
    try:
        save_viz_cache(
            out_dir,
            rank_results=rank_results,
            clf_results=clf_results,
            center_slices=_center_slices,
            thresholds=thresholds,
            metrics=rank_task_metrics,
            class_names_dict=_VIZ_CLASS_NAMES_DICT,
            task_definitions=_VIZ_TASK_DEFINITIONS,
            tasks=tasks,
        )
    except Exception as _e:
        logger.warning(f"Could not save viz cache (non-fatal): {_e}")

    # ── Ranking TTA uncertainty analysis (σ_TTA) ──────────────────────────
    from spineranknet.training_utils import run_rank_tta_uncertainty_analysis
    run_rank_tta_uncertainty_analysis(rank_results, tasks, thresholds, out_dir)

    return clf_overall, rank_overall


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT COMPATIBILITY STUDY
# ════════════════════════════════════════════════════════════════════════════


def print_compatibility_info() -> None:
    """Log a short summary of the checkpoint formats this script can load."""
    logger.info(
        "\n"
        "=" * 70 + "\n"
        "  CHECKPOINT COMPATIBILITY\n"
        "=" * 70 + "\n"
        "\n"
        "  Checkpoints written by this script:\n"
        "    [OK] Format: {model_weights: MultiHeadGradingRanker.state_dict(), ...}\n"
        "    [OK] Full load (encoder + clf_* + rank_* heads)\n"
        "    [OK] --encoder_only: load encoder (+ clf heads) only, e.g. to\n"
        "         initialise Phase-2 ranking from a Phase-1 clf checkpoint\n"
        "    [OK] rank_from_logits: LogitRankingHead(K->1) on clf logits\n"
        "\n"
        "  Older formats:\n"
        "    [OK] Hybrid format {backbone:, ranking_heads:}\n"
        "    [OK] Legacy 3-letter task-key abbreviations are remapped\n"
        "\n"
        "=" * 70
    )


# ════════════════════════════════════════════════════════════════════════════
# CLI / MAIN
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Command-line entry point for SpineRankNet baseline training.

    Parses ``sys.argv`` via :class:`argparse.ArgumentParser`, builds
    the :class:`HybridConfig`, and dispatches to one of the four modes
    (``train``, ``clf``, ``rank``, ``eval``).

    See the module docstring for the full CLI surface.
    """
    parser = argparse.ArgumentParser(
        description="SpineRankNet -- Unified Baseline Training "
        "(Classification + Ranking)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config (HybridConfig format)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "clf", "rank", "eval"],
        default="train",
        help="'train' = hybrid, 'clf' = classification only, "
        "'rank' = ranking only, 'eval' = evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint to load (classification or ranking)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint with full state (model + optimizer + "
        "scheduler + counters). Pass path to .pt file, or 'auto' to "
        "auto-discover latest checkpoint in save_dir. "
        "Overrides --checkpoint and --encoder_only.",
    )
    parser.add_argument(
        "--encoder_only",
        action="store_true",
        help="Load only encoder weights from checkpoint (re-init heads)",
    )
    parser.add_argument(
        "--stop_on_loss",
        action="store_true",
        help="Loss-aware early stopping: don't stop if training loss is still "
        "decreasing (smoothed over a window), even if the selection metric "
        "hasn't improved.",
    )
    parser.add_argument(
        "--loss_window",
        type=int,
        default=20,
        help="Window size (epochs) for smoothed loss trend check (default: 20).",
    )
    parser.add_argument(
        "--rank_from_logits",
        action="store_true",
        help="Derive ranking scores from clf logits (per-task head types respected)",
    )
    parser.add_argument(
        "--rank_from_concat_logits",
        action="store_true",
        help="Concatenate ALL task logits and derive ranking from shared head "
        "(inter-task modelling). Mutually exclusive with --rank_from_logits.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="",
        help="Backbone architecture: resnet18|resnet34|resnet50",
    )

    # ── loss configuration ──
    parser.add_argument(
        "--ranking_loss",
        type=str,
        default="",
        help=f"Ranking loss: {list(RANKING_LOSSES.keys())}",
    )
    parser.add_argument(
        "--ranking_head",
        type=str,
        default="",
        help="Ranking head type: mlp|linear|transformer|kan",
    )
    parser.add_argument(
        "--level_aware_pairs",
        action="store_true",
        help="Only form ranking pairs within the same IVD level (no cross-level pairs)",
    )
    parser.add_argument(
        "--clf_head",
        type=str,
        default="",
        help="Classification head type: linear|mlp|kan|transformer",
    )
    parser.add_argument(
        "--clf_loss",
        type=str,
        default="",
        help="Shorthand for TASK_LOSS_OVERRIDE._all_: "
        f"{list(LOSS_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--clf_loss_weight",
        type=float,
        default=None,
        help="Classification loss weight (default from config, typically 1.0). Use 0 to disable.",
    )
    parser.add_argument(
        "--ranking_loss_weight",
        type=float,
        default=None,
        help="Ranking loss weight (default from config, typically 1.0). Use 0 to disable.",
    )
    parser.add_argument(
        "--use_triplet",
        action="store_true",
        help="Enable triplet ordinal loss",
    )
    parser.add_argument(
        "--triplet_weight",
        type=float,
        default=None,
        help="Triplet loss weight (default from config, typically 0.5). Use 0 to disable.",
    )
    parser.add_argument(
        "--cross_task_loss_weight",
        type=float,
        default=None,
        help="Cross-task ranking concordance loss weight (SpineRank). "
        "Encourages consistent severity ordering across pathologies. "
        "Disabled by default (0.0). Recommended: 0.1.",
    )

    # ── SpineRank component ablation flags ──
    parser.add_argument(
        "--spinerank_margin_base",
        type=float,
        default=None,
        help="SpineRank: base margin (default 0.5). "
        "Only used when --ranking_loss SpineRank.",
    )
    parser.add_argument(
        "--spinerank_margin_scale",
        type=float,
        default=None,
        help="SpineRank: margin scale (default 1.5). "
        "Set to 0 to disable adaptive margin.",
    )
    parser.add_argument(
        "--spinerank_lambda_conc",
        type=float,
        default=None,
        help="SpineRank: concordance regulariser weight (default 0.25). "
        "Set to 0 to disable concordance term.",
    )
    parser.add_argument(
        "--spinerank_c2",
        type=float,
        default=None,
        help="SpineRank: same-grade similarity (L2) weight C2 (default 0.25). "
        "Set to 0 to disable the similarity regulariser.",
    )
    parser.add_argument(
        "--spinerank_no_adaptive_margin",
        action="store_true",
        help="SpineRank ablation: disable adaptive margin (use fixed margin_base).",
    )
    parser.add_argument(
        "--spinerank_no_severity_weight",
        action="store_true",
        help="SpineRank ablation: disable severity weighting on loss terms.",
    )
    parser.add_argument(
        "--spinerank_no_concordance",
        action="store_true",
        help="SpineRank ablation: disable concordance regulariser.",
    )
    parser.add_argument(
        "--spinerank_no_similarity",
        action="store_true",
        help="SpineRank ablation: disable similarity (same-grade L2) term.",
    )
    parser.add_argument(
        "--task_weighting",
        type=str,
        default="",
        help="Multi-task clf loss weighting: "
        "uniform|uncertainty|dwa|difficulty",
    )

    # ── training ──
    parser.add_argument("--lr", type=float, default=0.0, help="Override LR")
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=0.0,
        help="Override encoder LR (differential LR)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Override batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="Override epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs without improvement). "
        "Note: 0 is currently treated as 30, so early stopping "
        "cannot be disabled.",
    )
    parser.add_argument(
        "--selection_criterion",
        type=str,
        default="bal_acc",
        choices=["bal_acc", "qwk", "spearman", "composite_rank", "roc_auc", "mcc"],
        help="Model selection criterion. "
        "'bal_acc': legacy combined balanced accuracy (sensitive to bins). "
        "'qwk': mean Quadratic Weighted Kappa across tasks (recommended). "
        "'spearman': mean Spearman rank correlation. "
        "'composite_rank': 0.5*QWK + 0.3*Spearman + 0.2*BalAcc. "
        "'roc_auc': mean macro-averaged ROC-AUC (stable, threshold-free). "
        "'mcc': mean Matthews Correlation Coefficient (balanced). "
        "(default: bal_acc)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Debug/smoke-test option: randomly subsample this fraction "
        "of the training set (default 1.0 = all data; not used in the paper).",
    )

    # ── data mode ──
    parser.add_argument(
        "--label_mode",
        choices=["ordinal", "standard"],
        default="",
        help="Label mode: ordinal (default) or standard (binary)",
    )
    parser.add_argument(
        "--raw_labels",
        action="store_true",
        help="Use ordinal multi-class labels for all tasks "
        "(alias for --label_mode ordinal)",
    )
    parser.add_argument(
        "--multi_contrast",
        action="store_true",
        help="Enable T1+T2 multi-channel input",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversample rare/difficult label combos",
    )
    parser.add_argument(
        "--use_tta", action="store_true", default=True,
        help="TTA for test evaluation (enabled by default)",
    )
    parser.add_argument(
        "--no_tta", dest="use_tta", action="store_false",
        help="Disable TTA for test evaluation",
    )

    # ── output ──
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Override output directory",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="",
        help="Override evaluation output directory",
    )
    parser.add_argument(
        "--scores_cache",
        type=str,
        default="",
        help="Path to scores cache pickle for fast re-evaluation.  "
        "If the file exists, model inference is skipped and raw scores/labels "
        "are loaded from cache.  On first run the cache is created automatically.  "
        "Delete the file or use FORCE=1 to force re-inference.",
    )
    parser.add_argument(
        "--no_test_filter",
        action="store_true",
        help="Disable T2w+S1-per-patient test filtering",
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default=None,
        choices=["balanced_accuracy", "qwk", "accuracy", "composite"],
        help="Metric for ranking-score threshold optimisation (default: from config)",
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default=None,
        choices=["grid", "isotonic", "gmm", "youden"],
        help="Threshold search method: grid (default), isotonic, gmm, youden",
    )
    parser.add_argument(
        "--binning_mode",
        type=str,
        default="target_aligned",
        choices=["uniform", "target_aligned", "centroid"],
        help="Score discretisation mode for ranking BalAcc/QWK during "
        "training (default: target_aligned).",
    )

    # ── dataset caching (for fast evaluation) ──
    parser.add_argument(
        "--dataset_cache",
        type=str,
        default="",
        help="Path to pre-cached dataset tensors (NIFTI preprocessing cached). "
        "If exists, skips disk I/O for repeated evaluations. "
        "Can be created with --save_dataset_cache.",
    )
    parser.add_argument(
        "--save_dataset_cache",
        action="store_true",
        help="Save preprocessed images + labels to dataset_cache on first run. "
        "Speeds up subsequent evaluations.",
    )

    # ── binarization & visualization ──
    parser.add_argument(
        "--binarize_tasks",
        type=str,
        default="Pfirrmann CentralCanalStenosis Narrowing",
        help="Space-separated list of tasks to EXCLUDE from binarization. "
        "All other tasks will be binarized (0=normal, 1=abnormal). "
        "Default: keep Pfirrmann, CCS, Narrowing as multi-class.",
    )
    parser.add_argument(
        "--show_threshold_dashes",
        action="store_true",
        help="Overlay learned thresholds as dashed lines on boxplots.",
    )
    parser.add_argument(
        "--heatmap_metric",
        type=str,
        default="bal_acc",
        choices=["bal_acc", "qwk", "roc_auc", "mcc"],
        help="Metric to visualize in threshold method heatmap.",
    )
    parser.add_argument(
        "--show_ids",
        action="store_true",
        help="Show real sample/subject IDs in figure titles (default: "
        "anonymised #1, #2, ...). Never publish figures made with this flag.",
    )

    # ── rank–clf interaction ──
    parser.add_argument(
        "--cross_task_mode",
        type=str,
        default=None,
        help="Cross-task attention mode: none|transformer|mlp|gated|mean_pool",
    )
    parser.add_argument(
        "--agreement_weight",
        type=float,
        default=None,
        help="Rank–clf agreement loss weight (0 = disabled)",
    )
    parser.add_argument(
        "--rank_calibration",
        action="store_true",
        help="Enable rank-calibrated classification at eval time",
    )
    parser.add_argument(
        "--cal_sigma",
        type=float,
        default=None,
        help="Gaussian σ for rank-calibrated inference",
    )
    parser.add_argument(
        "--cal_alpha",
        type=float,
        default=None,
        help="Classification weight α in rank-calibrated fusion",
    )

    # ── ablation presets ──
    parser.add_argument(
        "--ablation",
        type=str,
        default="",
        help=f"Ablation preset: {sorted(ABLATION_PRESETS.keys())}",
    )
    parser.add_argument(
        "--list_ablations",
        action="store_true",
        help="List all available ablation presets and exit",
    )

    # ── fast numpy dataloader ──
    parser.add_argument(
        "--genodict_path",
        type=str,
        default="",
        help="Path to genodict pickle for fast numpy dataloader",
    )
    parser.add_argument(
        "--ivd_path",
        type=str,
        default="",
        help="Directory with .npy IVD volumes for fast numpy dataloader",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help=(
            "Root of the GENODISC data. Replaces the default relative prefix "
            f"'{DEFAULT_DATA_PREFIX}/' of GENODICT_PATH / IVD_PATH / JSON_PATH. "
            "Falls back to the GENODISC_ROOT environment variable."
        ),
    )

    # ── compatibility ──
    parser.add_argument(
        "--compat",
        action="store_true",
        help="Print checkpoint compatibility info and exit",
    )

    args = parser.parse_args()
    set_show_sample_ids(args.show_ids)

    # ── compatibility info ──────────────────────────────────────────
    if args.compat:
        print_compatibility_info()
        return

    # ── list ablation presets ────────────────────────────────────────
    if args.list_ablations:
        print(f"\n{'=' * 80}")
        print("Available ablation presets for --ablation <name>")
        print(f"{'=' * 80}")
        cats = {
            "Loss Component": [
                "clf_only", "rank_only", "triplet_only",
                "clf_rank", "clf_triplet", "rank_triplet", "all",
            ],
            "Ranking Loss": [
                "rank_deepsvm", "rank_ranknet", "rank_deeprelativeattributes", "rank_justnoticeabledifferences",
            ],
            "Classification Loss": [
                "clf_ce", "clf_emd", "clf_corn", "clf_coral",
                "clf_softce", "clf_clm", "clf_mae",
            ],
            "Ranking Head": [
                "head_mlp", "head_transformer", "head_least_squares",
                "head_kan", "head_linear",
            ],
            "Label Mode": ["labels_standard", "labels_ordinal"],
            "Task Weighting": [
                "tw_uniform", "tw_difficulty", "tw_uncertainty", "tw_dwa",
            ],
            "Contribution Validation": [
                "no_hard_mining", "no_severity_weight", "no_oversampling",
                "e2e_200ep", "e2e_400ep",
            ],
            "Cross-Task Attention": [
                "cross_task_transformer", "cross_task_mlp",
                "cross_task_gated", "cross_task_mean_pool",
            ],
        }
        for cat, presets in cats.items():
            print(f"\n  {cat}:")
            for p in presets:
                if p in ABLATION_PRESETS:
                    desc = ABLATION_PRESETS[p]["desc"]
                    print(f"    --ablation {p:25s}  # {desc}")
        print()
        return

    # ── Eval: auto-load training config from checkpoint dir ─────────
    # Lets `--mode eval --checkpoint <best.pt>` recover the right backbone /
    # ranking_loss / ranking_head without the user re-typing every flag.
    # Must run before the config is loaded below.
    if args.mode == "eval":
        _maybe_load_train_config(args)

    # ── Load config ──────────────────────────────────────────────────
    # Strip fast-dataloader keys (GENODICT_PATH, IVD_PATH) that
    # HybridConfig / RankingConfig don't know about, then attach
    # them manually after construction.
    _FAST_DL_KEYS = {"GENODICT_PATH", "IVD_PATH"}
    _fast_dl_kw: Dict[str, str] = {}

    if args.config:
        import yaml as _yaml
        with open(args.config, "r") as _f:
            _raw = _yaml.safe_load(_f) or {}
        # Path flags given on the command line replace the YAML values before
        # env-var expansion and before the config creates its output dirs.
        _cli_paths = {
            "GENODICT_PATH": args.genodict_path, "IVD_PATH": args.ivd_path,
            "SAVE_DIR": args.save_dir, "EVAL_DIR": args.eval_dir,
            "CHECKPOINT": args.checkpoint,
        }
        _cli_paths = {k: v for k, v in _cli_paths.items() if v}
        _raw = expand_env_paths({**_raw, **_cli_paths}, skip=_cli_paths.keys())
        for _k in _FAST_DL_KEYS:
            if _k in _raw:
                _fast_dl_kw[_k] = _raw.pop(_k)
        config = HybridConfig(**_raw)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = HybridConfig()
        logger.info("Using default HybridConfig")

    # Attach fast-dataloader paths (from YAML or CLI)
    for _k in _FAST_DL_KEYS:
        setattr(config, _k, _fast_dl_kw.get(_k, ""))
    if args.genodict_path:
        config.GENODICT_PATH = args.genodict_path
    if args.ivd_path:
        config.IVD_PATH = args.ivd_path

    # Relocate the default relative data layout (data/GENODISCv2/...) to
    # --data_root or $GENODISC_ROOT when given. Absolute paths and paths
    # outside the default prefix are left untouched.
    _data_root = os.path.expanduser(args.data_root or os.environ.get("GENODISC_ROOT", ""))
    if _data_root:
        for _k in ("GENODICT_PATH", "IVD_PATH", "JSON_PATH"):
            _v = getattr(config, _k, "") or ""
            _rel = _v.replace("\\", "/")
            if _rel.startswith("./"):
                _rel = _rel[2:]
            if _rel.startswith(DEFAULT_DATA_PREFIX + "/"):
                _new = os.path.join(_data_root, _rel[len(DEFAULT_DATA_PREFIX) + 1:])
                if _v.endswith("/") and not _new.endswith("/"):
                    _new += "/"
                setattr(config, _k, _new)
        logger.info(
            "Data root: %s (GENODICT_PATH=%s, IVD_PATH=%s)",
            _data_root, config.GENODICT_PATH, config.IVD_PATH,
        )

    # ── CLI overrides ────────────────────────────────────────────────
    if args.checkpoint:
        config.CHECKPOINT = args.checkpoint
    if args.backbone:
        config.MODEL_TYPE = args.backbone
    if args.ranking_loss:
        config.RANKING_LOSS = args.ranking_loss
    if args.ranking_head:
        config.RANKING_HEAD_TYPE = args.ranking_head
    if args.clf_head:
        config.HEAD_TYPE = args.clf_head
    if args.lr > 0:
        config.LR = args.lr
    if args.encoder_lr > 0:
        config.ENCODER_LR = args.encoder_lr
    if args.batch_size > 0:
        config.BATCH_SIZE = args.batch_size
    if args.epochs > 0:
        config.EPOCHS = args.epochs
    if args.seed > 0:
        config.SEED = args.seed
    if args.encoder_only:
        config.ENCODER_ONLY = True
    if args.rank_from_logits:
        config.RANK_FROM_LOGITS = True
    if getattr(args, "level_aware_pairs", False):
        config.LEVEL_AWARE_PAIRS = True
    if args.rank_from_concat_logits:
        config.RANK_FROM_CONCAT_LOGITS = True
        # Concat mode implies logit-based ranking (needs clf logits)
        config.RANK_FROM_LOGITS = False
    if args.save_dir:
        config.SAVE_DIR = args.save_dir
    if args.eval_dir:
        config.EVAL_DIR = args.eval_dir
    if args.no_test_filter:
        config.TEST_T2_S1_ONLY = False
    if args.threshold_metric:
        config.THRESHOLD_METRIC = args.threshold_metric
    if args.threshold_method:
        config.THRESHOLD_METHOD = args.threshold_method
    if hasattr(args, "binning_mode") and args.binning_mode:
        config.VAL_BINNING_MODE = args.binning_mode
    if args.label_mode == "standard":
        config.USE_RAW_LABELS = False
        config.TASKS = list(STANDARD_TASKS_13)
    elif args.label_mode == "ordinal" or args.raw_labels:
        config.USE_RAW_LABELS = True
        config.TASKS = list(ORDINAL_TASKS_13)
    if args.multi_contrast:
        config.USE_MULTI_CONTRAST = True
    if args.oversample:
        config.OVERSAMPLE_DIFFICULT = True
    if args.clf_loss_weight is not None:
        config.CLF_LOSS_WEIGHT = args.clf_loss_weight
    if args.ranking_loss_weight is not None:
        config.RANKING_LOSS_WEIGHT = args.ranking_loss_weight
    if args.use_triplet:
        config.USE_TRIPLET = True
    if args.triplet_weight is not None:
        config.TRIPLET_LOSS_WEIGHT = args.triplet_weight
    if args.cross_task_loss_weight is not None:
        config.CROSS_TASK_LOSS_WEIGHT = args.cross_task_loss_weight
    if args.task_weighting:
        config.TASK_WEIGHT_STRATEGY = args.task_weighting
    if args.clf_loss:
        config.TASK_LOSS_OVERRIDE = {"_all_": args.clf_loss}

    # ── rank–clf interaction overrides ────────────────────────────────
    if args.cross_task_mode is not None:
        config.CROSS_TASK_MODE = args.cross_task_mode
    if args.agreement_weight is not None:
        config.AGREEMENT_LOSS_WEIGHT = args.agreement_weight
    if args.rank_calibration:
        config.USE_RANK_CALIBRATION = True
    if args.cal_sigma is not None:
        config.CALIBRATION_SIGMA = args.cal_sigma
    if args.cal_alpha is not None:
        config.CALIBRATION_ALPHA = args.cal_alpha

    # ── Resolve tasks ────────────────────────────────────────────────
    tasks = resolve_tasks(config)
    config.TASKS = tasks

    # ── Apply mode shorthand ─────────────────────────────────────────
    if args.mode == "clf" and not args.ablation:
        if args.label_mode == "standard":
            args.ablation = "clf_only_standard"
        else:
            args.ablation = "clf_only"
    elif args.mode == "rank" and not args.ablation:
        args.ablation = "rank_only"

    # ── Apply ablation preset (after tasks resolved) ─────────────────
    ablation_desc = ""
    if args.ablation:
        ablation_desc = apply_ablation_preset(config, args.ablation)
        logger.info(f"Applied ablation preset: {args.ablation} -- {ablation_desc}")

    # ── Expand _all_ wildcard in TASK_LOSS_OVERRIDE ──────────────────
    tlo = getattr(config, "TASK_LOSS_OVERRIDE", None)
    if isinstance(tlo, dict) and "_all_" in tlo:
        loss_name = tlo["_all_"]
        config.TASK_LOSS_OVERRIDE = {t: loss_name for t in tasks}

    ranking_tasks = [
        t for t in tasks if TASK_DEFINITIONS.get(t, {}).get("ranking", True)
    ]
    logger.info(f"Tasks ({len(tasks)}): {tasks}")
    logger.info(f"Ranking tasks ({len(ranking_tasks)}): {ranking_tasks}")

    # ── Seed ─────────────────────────────────────────────────────────
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Output directories ───────────────────────────────────────────
    save_dir = Path(config.SAVE_DIR) / config.MODEL_TYPE
    eval_dir = Path(config.EVAL_DIR) / config.MODEL_TYPE
    log_dir = Path(config.LOGDIR) / config.MODEL_TYPE
    for d in [save_dir, eval_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Dump training config next to best.pt so future --mode eval can auto-recover
    # backbone / ranking_loss / ranking_head without re-typing every flag.
    if args.mode in ("train", "clf", "rank"):
        _dump_train_config(config, save_dir)

    # ── Dataloaders ──────────────────────────────────────────────────
    # The published trainer uses only the fast numpy pipeline: a genodict
    # pickle (GENODICT_PATH) plus pre-extracted per-IVD .npy volumes (IVD_PATH).
    if not (getattr(config, "GENODICT_PATH", "") and getattr(config, "IVD_PATH", "")):
        raise RuntimeError(
            "GENODICT_PATH and IVD_PATH must both be set (YAML config, "
            "--genodict_path / --ivd_path, or --data_root / $GENODISC_ROOT "
            "with the default data/GENODISCv2 layout)."
        )
    logger.info("Using FAST numpy dataloader (GenodiscOrdinalDataset)")
    loaders = get_dataloaders_fast(
        config,
        tasks,
        data_fraction=args.data_fraction,
        seed=seed,
        use_tta_test=args.use_tta,
    )
    logger.info(
        f"DataLoaders: Train={len(loaders['Train'].dataset)}, "
        f"Val={len(loaders['Val'].dataset)}, "
        f"Test={len(loaders['Test'].dataset)}"
    )

    # ── Resume: auto-discover checkpoint ─────────────────────────────
    _resume_ckpt = None  # will hold loaded checkpoint dict if resuming
    if args.resume:
        if args.resume.lower() == "auto":
            # Auto-discover latest checkpoint in save_dir
            periodic = sorted(
                save_dir.glob("ckpt_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
                reverse=True,
            )
            if periodic:
                args.resume = str(periodic[0])
                logger.info(f"Auto-resume: found latest periodic checkpoint {args.resume}")
            elif (save_dir / "best.pt").exists():
                args.resume = str(save_dir / "best.pt")
                logger.info(f"Auto-resume: found best.pt at {args.resume}")
            else:
                logger.warning("Auto-resume: no checkpoints found — starting fresh")
                args.resume = ""

        if args.resume and os.path.isfile(args.resume):
            if args.encoder_only:
                logger.warning("--resume overrides --encoder_only. Ignoring --encoder_only.")
                args.encoder_only = False
                config.ENCODER_ONLY = False
            _resume_ckpt = torch.load(args.resume, weights_only=False, map_location="cpu")
            logger.info(f"Loaded resume checkpoint: {args.resume}")

    # ── Model ────────────────────────────────────────────────────────
    if _resume_ckpt is not None:
        # Resume: build model without checkpoint, then load full state
        model, _, _ = build_model(config, tasks, skip_checkpoint=True)
        state = _resume_ckpt.get("model_weights", _resume_ckpt)
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered, strict=False)
        epoch_start = int(_resume_ckpt.get("epoch", _resume_ckpt.get("epoch_no", 0))) + 1
        best_metric = float(
            _resume_ckpt.get("best_metric", _resume_ckpt.get("acc", 0.0))
        )
        logger.info(
            f"RESUME: loaded {len(filtered)}/{len(state)} params "
            f"(epoch {epoch_start - 1}, best_metric={best_metric:.3f})"
        )
    else:
        model, epoch_start, best_metric = build_model(
            config,
            tasks,
            skip_checkpoint=(args.mode == "eval"),
        )
    logger.info(
        f"Model: {config.MODEL_TYPE} | "
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # ── Ranking loss ─────────────────────────────────────────────────
    ranking_weight = float(getattr(config, "RANKING_LOSS_WEIGHT", 1.0))
    loss_name = config.RANKING_LOSS
    if loss_name not in RANKING_LOSSES:
        raise ValueError(
            f"Unknown ranking loss: {loss_name!r}. "
            f"Available: {list(RANKING_LOSSES.keys())}"
        )
    # Build kwargs for the ranking loss
    rank_loss_kwargs = dict(
        margin=config.RANKING_MARGIN,
        C1=config.RANKING_C1,
        C2=config.RANKING_C2,
    )
    # SpineRank-specific: pass ablation flags and hyperparameters
    if loss_name == "SpineRank":
        if args.spinerank_margin_base is not None:
            rank_loss_kwargs["margin_base"] = args.spinerank_margin_base
        if args.spinerank_margin_scale is not None:
            rank_loss_kwargs["margin_scale"] = args.spinerank_margin_scale
        if args.spinerank_lambda_conc is not None:
            rank_loss_kwargs["lambda_conc"] = args.spinerank_lambda_conc
        if args.spinerank_c2 is not None:
            rank_loss_kwargs["C2"] = args.spinerank_c2
        rank_loss_kwargs["use_adaptive_margin"] = not args.spinerank_no_adaptive_margin
        rank_loss_kwargs["use_severity_weight"] = not args.spinerank_no_severity_weight
        rank_loss_kwargs["use_concordance"] = not args.spinerank_no_concordance
        rank_loss_kwargs["use_similarity"] = not args.spinerank_no_similarity

    ranking_loss_fn = RANKING_LOSSES[loss_name](**rank_loss_kwargs)
    logger.info(
        f"Ranking loss: {loss_name} "
        f"(pointwise={getattr(ranking_loss_fn, 'pointwise', False)})"
    )
    if loss_name == "SpineRank":
        logger.info(f"  SpineRank config: {ranking_loss_fn}")

    # ── Triplet loss ─────────────────────────────────────────────────
    use_triplet = getattr(config, "USE_TRIPLET", False)
    triplet_loss_fn = (
        TripletMarginOrdinalLoss(margin=getattr(config, "TRIPLET_MARGIN", 1.0))
        if use_triplet
        else None
    )

    # ── Class weights (for classification loss) ──────────────────────
    clf_weight = float(getattr(config, "CLF_LOSS_WEIGHT", 1.0))
    class_weights: Dict[str, Optional[torch.Tensor]] = {}
    if clf_weight > 0 and getattr(config, "USE_CLASS_WEIGHTS", True):
        if "_genodict_train" in loaders:
            class_weights = compute_class_weights_fast(
                loaders["_genodict_train"], tasks,
            )
        else:
            from spineranknet.training_utils import compute_class_weights
            class_weights = compute_class_weights(loaders["Train"].dataset, tasks)
        logger.info(f"Computed class weights for {len(class_weights)} tasks")

    # ── Task weighting (for multi-task clf loss) ─────────────────────
    tw_strategy = getattr(config, "TASK_WEIGHT_STRATEGY", "uniform")
    task_weighting = None
    if tw_strategy != "uniform" and clf_weight > 0:
        task_weighting = build_task_weighting(
            strategy=tw_strategy,
            tasks=tasks,
            task_definitions=TASK_DEFINITIONS,
            temperature=getattr(config, "DWA_TEMPERATURE", 2.0),
        )
        logger.info(f"Task weighting: {tw_strategy}")

    # ── Print config summary ─────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("  SpineRankNet -- Unified Baseline Training")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Mode:              {args.mode}")
    if ablation_desc:
        logger.info(f"  Ablation:          {args.ablation} ({ablation_desc})")
    logger.info(f"  Backbone:          {config.MODEL_TYPE}")
    logger.info(f"  CLF loss weight:   {clf_weight}")
    logger.info(f"  Rank loss weight:  {ranking_weight}")
    logger.info(f"  Triplet:           {use_triplet} (w={getattr(config, 'TRIPLET_LOSS_WEIGHT', 0.5)})")
    logger.info(f"  Ranking loss:      {loss_name}")
    logger.info(f"  Ranking head:      {config.RANKING_HEAD_TYPE}")
    logger.info(f"  Rank from logits:  {config.RANK_FROM_LOGITS}")
    logger.info(f"  Clf head:          {getattr(config, 'HEAD_TYPE', 'linear')}")
    logger.info(f"  Task weighting:    {tw_strategy}")
    logger.info(f"  Bounded:           {config.BOUNDED}")
    logger.info(f"  Encoder only:      {getattr(config, 'ENCODER_ONLY', False)}")
    logger.info(f"  LR:                {config.LR}")
    logger.info(f"  Encoder LR:        {config.ENCODER_LR}")
    logger.info(f"  Batch size:        {config.BATCH_SIZE}")
    logger.info(f"  Epochs:            {config.EPOCHS}")
    logger.info(f"  Seed:              {seed}")
    logger.info(f"  Data fraction:     {args.data_fraction}")
    logger.info(f"  Save dir:          {save_dir}")
    logger.info(f"  Eval dir:          {eval_dir}")
    tlo = getattr(config, "TASK_LOSS_OVERRIDE", None)
    if tlo:
        unique_losses = set(tlo.values()) if isinstance(tlo, dict) else set()
        logger.info(f"  Task loss override: {unique_losses}")
    logger.info(f"{'=' * 60}\n")

    # ════════════════════════════════════════════════════════════════════
    # EVAL ONLY
    # ════════════════════════════════════════════════════════════════════

    if args.mode == "eval":
        # Load checkpoint for eval
        if args.checkpoint and os.path.isfile(args.checkpoint):
            ckpt = torch.load(
                args.checkpoint, weights_only=False, map_location="cpu"
            )
            state = ckpt.get("model_weights", ckpt)
            state = _remap_legacy_keys(state)
            model_state = model.state_dict()
            filtered = {
                k: v
                for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = [
                k for k in state if k not in filtered
            ]
            model.load_state_dict(filtered, strict=False)
            logger.info(
                f"Loaded eval checkpoint: {args.checkpoint} "
                f"({len(filtered)}/{len(state)} params matched)"
            )
            if skipped:
                logger.warning(
                    f"  {len(skipped)} keys skipped (shape mismatch or missing): "
                    f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}"
                )
            # Log ranking-head loading status
            rank_loaded = [k for k in filtered if "ranking" in k]
            clf_loaded = [k for k in filtered if "heads.clf_" in k or "heads.clf." in k]
            enc_loaded = [k for k in filtered if "encoder" in k]
            logger.info(
                f"  Breakdown: encoder={len(enc_loaded)}, "
                f"clf_heads={len(clf_loaded)}, ranking_heads={len(rank_loaded)}"
            )
        else:
            logger.error(
                f"No checkpoint found for eval! "
                f"Path: {args.checkpoint!r} "
                f"(exists={os.path.isfile(args.checkpoint) if args.checkpoint else False})"
            )
            logger.error("Model will use random weights — results will be meaningless.")

        binarize_tasks_set = set(getattr(args, "binarize_tasks", "Pfirrmann CentralCanalStenosis Narrowing").split())
        comprehensive_evaluation(
            model, loaders, tasks, config, eval_dir,
            scores_cache=getattr(args, "scores_cache", ""),
            dataset_cache=getattr(args, "dataset_cache", ""),
            save_dataset_cache=getattr(args, "save_dataset_cache", False),
            binarize_tasks=binarize_tasks_set,
            show_threshold_dashes=getattr(args, "show_threshold_dashes", False),
            heatmap_metric=getattr(args, "heatmap_metric", "bal_acc"),
        )
        print_compatibility_info()
        return

    # ════════════════════════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════════════════════════

    # Optimizer: differential LR for encoder vs heads
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": config.ENCODER_LR},
            {"params": head_params, "lr": config.LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    # LR scheduler — built after all param groups are finalized
    sched_type = getattr(config, "LR_SCHEDULER_TYPE", "cosine").lower()
    if sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(config, "LR_SCHEDULER_STEP", 50),
            gamma=getattr(config, "LR_DECAY", 0.1),
        )
    elif sched_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=getattr(config, "LR_DECAY", 0.1),
            patience=getattr(config, "LR_PLATEAU_PATIENCE", 30),
            min_lr=getattr(config, "LR_PLATEAU_MIN", 1e-7),
            verbose=True,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )

    # Training history
    history = {
        "train_loss": [],
        "train_clf_loss": [],
        "train_rank_loss": [],
        "val_clf_acc": [],
        "val_rank_acc": [],
        "val_rank_qwk": [],
        "val_rank_spearman": [],
        "val_rank_roc_auc": [],
        "val_rank_mcc": [],
        "val_per_task_clf": [],
        "val_per_task_rank": [],
    }

    # ── Resume: restore optimizer, scheduler, history, counters ──
    _resume_epochs_no_improve = 0
    if _resume_ckpt is not None:
        # Optimizer state
        if "optimizer_state_dict" in _resume_ckpt:
            try:
                optimizer.load_state_dict(_resume_ckpt["optimizer_state_dict"])
                logger.info("  RESUME: restored optimizer state")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"  RESUME: optimizer state mismatch ({e}) — re-initialized")
        else:
            logger.warning("  RESUME: no optimizer state in checkpoint — re-initialized")

        # Scheduler state — handle epoch extension
        _ckpt_epoch = int(_resume_ckpt.get("epoch", 0))
        if config.EPOCHS > _ckpt_epoch:
            # Epochs extended — create fresh cosine for remaining epochs
            remaining = config.EPOCHS - _ckpt_epoch
            # Reset LRs to original config values for fresh schedule
            optimizer.param_groups[0]["lr"] = config.ENCODER_LR
            optimizer.param_groups[1]["lr"] = config.LR
            if sched_type == "cosine" or sched_type not in ("step", "plateau"):
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-6
                )
            logger.info(
                f"  RESUME: epochs extended ({_ckpt_epoch} → {config.EPOCHS}), "
                f"fresh cosine schedule T_max={remaining}"
            )
        elif "scheduler_state_dict" in _resume_ckpt:
            try:
                scheduler.load_state_dict(_resume_ckpt["scheduler_state_dict"])
                logger.info("  RESUME: restored scheduler state")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"  RESUME: scheduler state mismatch ({e}) — re-initialized")
                for _ in range(_ckpt_epoch):
                    scheduler.step()
                logger.info(f"  Manually stepped scheduler to epoch {_ckpt_epoch}")
        else:
            logger.warning("  RESUME: no scheduler state in checkpoint")
            for _ in range(_ckpt_epoch):
                scheduler.step()
            logger.info(f"  Manually stepped scheduler to epoch {_ckpt_epoch}")

        # History
        if "history" in _resume_ckpt:
            history = _resume_ckpt["history"]
            logger.info(
                f"  RESUME: restored training history "
                f"({len(history.get('train_loss', []))} epochs)"
            )

        # Early stopping counter
        _resume_epochs_no_improve = int(_resume_ckpt.get("epochs_no_improve", 0))
        logger.info(f"  RESUME: epochs_no_improve = {_resume_epochs_no_improve}")

        # Free memory
        del _resume_ckpt

    # ── model selection criterion ──
    # Supported: "bal_acc" (legacy), "qwk", "spearman",
    #            "composite_rank" = 0.5*QWK + 0.3*Spearman + 0.2*BalAcc
    selection_criterion = getattr(args, "selection_criterion", "bal_acc")
    patience = getattr(args, "patience", 30) or 30
    stop_on_loss = getattr(args, "stop_on_loss", False)
    loss_window = getattr(args, "loss_window", 20)
    logger.info(
        f"Model selection: criterion={selection_criterion}, patience={patience}"
        f"{', stop_on_loss (window=' + str(loss_window) + ')' if stop_on_loss else ''}"
    )

    best_val_metric = best_metric
    epochs_no_improve = _resume_epochs_no_improve

    for epoch in range(epoch_start, config.EPOCHS + 1):
        # ── train ──
        avg_total, avg_clf, avg_rank, task_losses = train_epoch(
            model,
            loaders["Train"],
            optimizer,
            tasks,
            config,
            class_weights,
            ranking_loss_fn,
            triplet_loss_fn,
            epoch,
            task_weighting=task_weighting,
        )

        history["train_loss"].append(avg_total)
        history["train_clf_loss"].append(avg_clf)
        history["train_rank_loss"].append(avg_rank)

        # ── validate ──
        binning_mode = getattr(config, "VAL_BINNING_MODE", "target_aligned")
        clf_acc, clf_per_task, rank_acc, rank_per_task, rank_extra = (
            validate_epoch(
                model, loaders["Val"], tasks, config, epoch, "Val",
                binning_mode=binning_mode,
            )
        )
        history["val_clf_acc"].append(clf_acc)
        history["val_rank_acc"].append(rank_acc)
        history["val_per_task_clf"].append(clf_per_task)
        history["val_per_task_rank"].append(rank_per_task)

        # Extract ranking extras for logging + selection
        mean_qwk = rank_extra.get("__mean__", {}).get("qwk", 0.0)
        mean_sp = rank_extra.get("__mean__", {}).get("spearman", 0.0)
        mean_kt = rank_extra.get("__mean__", {}).get("kendall", 0.0)
        mean_roc_auc = rank_extra.get("__mean__", {}).get("roc_auc", 0.0)
        mean_mcc = rank_extra.get("__mean__", {}).get("mcc", 0.0)
        history["val_rank_qwk"].append(mean_qwk)
        history["val_rank_spearman"].append(mean_sp)
        history["val_rank_roc_auc"].append(mean_roc_auc)
        history["val_rank_mcc"].append(mean_mcc)

        # ── combined metric for best-model selection ──
        if selection_criterion == "qwk":
            # Use mean QWK (scale to % for consistency with legacy)
            val_metric = mean_qwk * 100.0
        elif selection_criterion == "spearman":
            val_metric = mean_sp * 100.0
        elif selection_criterion == "roc_auc":
            # ROC-AUC already in [0, 1]; scale to %
            val_metric = mean_roc_auc * 100.0
        elif selection_criterion == "mcc":
            # MCC in [-1, 1]; map to [0, 100] so 0→50, 1→100
            val_metric = (mean_mcc + 1.0) * 50.0
        elif selection_criterion == "composite_rank":
            # Weighted combination of ranking-specific metrics
            # 0.5*QWK + 0.3*Spearman + 0.2*(rank_BalAcc/100)
            val_metric = (
                0.5 * max(mean_qwk, 0.0)
                + 0.3 * max(mean_sp, 0.0)
                + 0.2 * (rank_acc / 100.0)
            ) * 100.0
        else:
            # Legacy: "bal_acc" — use combined clf + rank balanced accuracy
            if clf_weight > 0 and ranking_weight > 0:
                val_metric = 0.5 * clf_acc + 0.5 * rank_acc
            elif clf_weight > 0:
                val_metric = clf_acc
            else:
                val_metric = rank_acc

        # Step scheduler (after validation so plateau has the metric)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric)
        else:
            scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Loss: {avg_total:.4f} (clf={avg_clf:.3f}, rank={avg_rank:.3f}) | "
            f"Val CLF: {clf_acc:.2f}% | Val RANK: {rank_acc:.2f}% | "
            f"QWK: {mean_qwk:.4f} | ρ: {mean_sp:.4f} | "
            f"AUC: {mean_roc_auc:.4f} | MCC: {mean_mcc:.4f} | "
            f"Sel({selection_criterion}): {val_metric:.2f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── save best ──
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            epochs_no_improve = 0
            ckpt_data = {
                "model_weights": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "acc": val_metric,
                "best_metric": best_val_metric,
                "epochs_no_improve": 0,  # just improved
                "history": history,
                "clf_acc": clf_acc,
                "rank_acc": rank_acc,
                "rank_qwk": mean_qwk,
                "rank_spearman": mean_sp,
                "selection_criterion": selection_criterion,
                "config": {
                    "MODEL_TYPE": config.MODEL_TYPE,
                    "RANKING_LOSS": config.RANKING_LOSS,
                    "RANKING_HEAD_TYPE": config.RANKING_HEAD_TYPE,
                    "RANK_FROM_LOGITS": config.RANK_FROM_LOGITS,
                    "BOUNDED": config.BOUNDED,
                    "CLF_LOSS_WEIGHT": clf_weight,
                    "RANKING_LOSS_WEIGHT": ranking_weight,
                    "LR": config.LR,
                    "SEED": config.SEED,
                    "mode": args.mode,
                    "EPOCHS": config.EPOCHS,
                },
            }
            best_path = save_dir / "best.pt"
            torch.save(ckpt_data, best_path)
            logger.info(f"  New best: {val_metric:.2f}% -> {best_path}")
        else:
            epochs_no_improve += 1

        # ── periodic checkpoint ──
        if epoch % 50 == 0:
            ckpt_data = {
                "model_weights": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "acc": val_metric,
                "best_metric": best_val_metric,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
            }
            torch.save(ckpt_data, save_dir / f"ckpt_{epoch}.pt")

        # ── early stopping (patience=0 is mapped to 30 above) ──
        if patience > 0 and epochs_no_improve >= patience:
            if stop_on_loss and _loss_still_decreasing(history, loss_window):
                # Loss is still decreasing — override metric-based stopping
                if epochs_no_improve == patience:  # log once at threshold
                    logger.info(
                        f"  Patience exhausted ({patience} ep) but training "
                        f"loss still decreasing — continuing training"
                    )
            else:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs"
                    f"{', loss also plateaued' if stop_on_loss else ''})"
                )
                break

    # ── save training curves ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(history["train_loss"], color=ETH_COLORS["petrol"], label="Total")
    axes[0, 0].plot(
        history["train_clf_loss"],
        color=ETH_COLORS["blue"],
        alpha=0.7,
        label="CLF",
    )
    axes[0, 0].plot(
        history["train_rank_loss"],
        color=ETH_COLORS["red"],
        alpha=0.7,
        label="Rank",
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        history["val_clf_acc"], color=ETH_COLORS["blue"], label="CLF BalAcc"
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("BalAcc (%)")
    axes[0, 1].set_title("Validation Classification BalAcc")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        history["val_rank_acc"], color=ETH_COLORS["red"], label="Rank BalAcc"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("BalAcc (%)")
    axes[1, 0].set_title("Validation Ranking BalAcc")
    axes[1, 0].grid(True, alpha=0.3)
    if history["val_rank_acc"]:
        best_rank = max(history["val_rank_acc"])
        axes[1, 0].axhline(
            y=best_rank,
            color=ETH_COLORS["red"],
            linestyle="--",
            alpha=0.5,
            label=f"Best: {best_rank:.2f}%",
        )
    axes[1, 0].legend()

    # Panel 4: Ranking QWK + Spearman ρ + ROC-AUC + MCC
    if history["val_rank_qwk"]:
        axes[1, 1].plot(
            history["val_rank_qwk"],
            color=ETH_COLORS["petrol"],
            label="QWK",
        )
    if history["val_rank_spearman"]:
        axes[1, 1].plot(
            history["val_rank_spearman"],
            color=ETH_COLORS["bronze"],
            label=r"Spearman $\rho$",
        )
    if history.get("val_rank_roc_auc"):
        axes[1, 1].plot(
            history["val_rank_roc_auc"],
            color=ETH_COLORS["blue"],
            alpha=0.8,
            label="ROC-AUC",
        )
    if history.get("val_rank_mcc"):
        # MCC is in [-1, 1]; plot raw value (most will be 0–1 range)
        axes[1, 1].plot(
            history["val_rank_mcc"],
            color=ETH_COLORS["purple"],
            alpha=0.8,
            label="MCC",
        )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Correlation / Agreement")
    axes[1, 1].set_title(f"Ranking Criteria (sel: {selection_criterion})")
    axes[1, 1].set_ylim(-0.1, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    if history["val_rank_qwk"]:
        best_qwk = max(history["val_rank_qwk"])
        axes[1, 1].axhline(
            y=best_qwk,
            color=ETH_COLORS["petrol"],
            linestyle="--",
            alpha=0.5,
            label=f"Best QWK: {best_qwk:.4f}",
        )
    axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(eval_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── per-task validation curves ───────────────────────────────────
    if history["val_per_task_rank"]:
        fig, ax = plt.subplots(figsize=(14, 8))
        for task in ranking_tasks:
            vals = [
                ep.get(task, np.nan) for ep in history["val_per_task_rank"]
            ]
            ax.plot(vals, label=task, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BalAcc (%)")
        ax.set_title("Per-Task Validation Ranking BalAcc")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            eval_dir / "per_task_val_rank_curves.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    if history["val_per_task_clf"]:
        fig, ax = plt.subplots(figsize=(14, 8))
        for task in tasks:
            vals = [
                ep.get(task, np.nan) for ep in history["val_per_task_clf"]
            ]
            ax.plot(vals, label=task, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BalAcc (%)")
        ax.set_title("Per-Task Validation Classification BalAcc")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            eval_dir / "per_task_val_clf_curves.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION WITH BEST CHECKPOINT
    # ════════════════════════════════════════════════════════════════════

    best_path = save_dir / "best.pt"
    if best_path.exists():
        logger.info(f"\nLoading best checkpoint for final evaluation: {best_path}")
        ckpt = torch.load(best_path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_weights"])

    binarize_tasks_set = set(getattr(args, "binarize_tasks", "Pfirrmann CentralCanalStenosis Narrowing").split())
    comprehensive_evaluation(
        model, loaders, tasks, config, eval_dir,
        scores_cache=getattr(args, "scores_cache", ""),
        dataset_cache=getattr(args, "dataset_cache", ""),
        save_dataset_cache=getattr(args, "save_dataset_cache", False),
        binarize_tasks=binarize_tasks_set,
        show_threshold_dashes=getattr(args, "show_threshold_dashes", False),
        heatmap_metric=getattr(args, "heatmap_metric", "bal_acc"),
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Best Val Metric: {best_val_metric:.2f}%")
    logger.info(f"  Weights:  {save_dir}")
    logger.info(f"  Eval:     {eval_dir}")
    logger.info(f"{'=' * 60}")

    # Print compatibility info at the end
    print_compatibility_info()


if __name__ == "__main__":
    main()
