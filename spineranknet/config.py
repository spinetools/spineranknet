#!/usr/bin/env python3
"""Shared task definitions and configuration dataclasses for SpineRankNet.

Central module providing the authoritative single source of truth for
clinical grading task metadata, task presets, training / ranking
configuration dataclasses (YAML-loadable), and the ETH Corporate Design
colour palette.

Contents
~~~~~~~~
* Clinical grading scales (``TASK_DEFINITIONS``) — one authoritative
  source shared by classification, ranking, and evaluation code.
* Task presets (``DEFAULT_TASKS_*``, ``GRANULAR_TASKS_*``,
  ``ORDINAL_TASKS_*``, ``STANDARD_TASKS_13``).
* :class:`GradingConfig` / :class:`RankingConfig` dataclasses
  (YAML-loadable through :meth:`~GradingConfig.from_yaml`).
* ETH Corporate Design colour palette and
  :func:`get_severity_colors` helper.

Notes
-----
Ordinal label ranges are the values emitted by
``GenodiscDataset._extract_labels``. The Pfirrmann grade is
stored 1–5 in the source JSON and shifted to 0–4 at extraction time.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import yaml


__all__ = [
    "INVALID_LABEL",
    "TASK_DEFINITIONS",
    "CLASSIFICATION_ONLY_TASKS",
    "STANDARD_TASKS_13",
    "ORDINAL_TASKS_13",
    "ORDINAL_TASKS_16",
    "DEFAULT_GRADING_TASKS",
    "DEFAULT_TASKS_8",
    "DEFAULT_TASKS_11",
    "DEFAULT_TASKS_13",
    "GRANULAR_TASKS_14",
    "GRANULAR_TASKS_16",
    "CLASS_NAMES_DICT",
    "ETH_COLORS",
    "ETH_SEVERITY_PALETTE",
    "get_severity_colors",
    "GradingConfig",
    "RankingConfig",
    "resolve_tasks",
    "PATH_KEYS",
    "expand_env_paths",
]


# ============================================================================
# INVALID LABEL SENTINEL
# ============================================================================

INVALID_LABEL: int = -100


# ============================================================================
# CLINICAL GRADING SCALES  (single source of truth)
# ============================================================================
#
#  Every task defines:
#    num_classes  – number of ordinal/nominal categories
#    binary       – True only for genuinely two-class tasks
#    default_loss – loss function when none is overridden
#    class_names  – human-readable category names (index-aligned)
#    model_head   – short tag used as the nn.ModuleDict key in MultiHeadClassifier
#
#  Ordinal ranges match the labels emitted by spineranknet.dataloaders.Genodisc._extract_labels.
#  Pfirrmann is stored 1–5 in the JSON and shifted to 0–4 at extraction time.

TASK_DEFINITIONS: Dict[str, Dict] = {
    # ════════════════════════════════════════════════════════════════════════
    # ORDINAL MULTI-CLASS  (full clinical grading scales)
    # ════════════════════════════════════════════════════════════════════════
    "Pfirrmann": {
        "num_classes": 5, "binary": False, "default_loss": "ce",
        "class_names": ["Grade I", "Grade II", "Grade III", "Grade IV", "Grade V"],
        "model_head": "clf_pf", "ranking_head": "rank_pf",
    },
    "Narrowing": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["None", "Mild", "Moderate", "Severe"],
        "model_head": "clf_nar", "ranking_head": "rank_nar",
    },
    "CentralCanalStenosis": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["None", "Mild", "Moderate", "Severe"],
        "model_head": "clf_ccs", "ranking_head": "rank_ccs",
    },
    "Spondylolisthesis": {
        "num_classes": 3, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Moderate", "Severe"],
        "model_head": "clf_spn", "ranking_head": "rank_spn",
    },
    "UpperEndplateDefect": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Slight", "Moderate", "Severe"],
        "model_head": "clf_ued", "ranking_head": "rank_ued",
    },
    "LowerEndplateDefect": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Slight", "Moderate", "Severe"],
        "model_head": "clf_led", "ranking_head": "rank_led",
    },
    "UpperModic": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["None", "Type I", "Type II", "Type III"],
        "model_head": "clf_umc", "ranking_head": "rank_umc",
    },
    "LowerModic": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["None", "Type I", "Type II", "Type III"],
        "model_head": "clf_lmc", "ranking_head": "rank_lmc",
    },
    "ForaminalStenosisLeft": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Absent", "Mild", "Moderate", "Severe"],
        "model_head": "clf_fsl", "ranking_head": "rank_fsl",
    },
    "ForaminalStenosisRight": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Absent", "Mild", "Moderate", "Severe"],
        "model_head": "clf_fsr", "ranking_head": "rank_fsr",
    },
    "Herniation": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Slight", "Moderate", "Large"],
        "model_head": "clf_hrn", "ranking_head": "rank_hrn",
    },
    "AnteriorBulging": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Slight", "Moderate", "Severe"],
        "model_head": "clf_anb", "ranking_head": "rank_anb",
    },
    "PosteriorBulging": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Slight", "Moderate", "Severe"],
        "model_head": "clf_pob", "ranking_head": "rank_pob",
    },
    "AnnularTears": {
        "num_classes": 3, "binary": False, "default_loss": "ce",
        "class_names": ["Absent", "Present", "Contiguous"],
        "model_head": "clf_ant", "ranking_head": "rank_ant",
    },
    "FacetJointArthropathyLeft": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Mild", "Moderate", "Severe"],
        "model_head": "clf_fjl", "ranking_head": "rank_fjl",
    },
    "FacetJointArthropathyRight": {
        "num_classes": 4, "binary": False, "default_loss": "ce",
        "class_names": ["Normal", "Mild", "Moderate", "Severe"],
        "model_head": "clf_fjr", "ranking_head": "rank_fjr",
    },

    # ════════════════════════════════════════════════════════════════════════
    # BINARY VARIANTS  (0 = normal / absent,  1 = any pathology present)
    #
    # Standard mode uses Pfirrmann / Narrowing / CCS as multi-class but
    # binarises every other task.  The "*Binary" tasks below provide the
    # 2-class definitions; _extract_labels populates them by thresholding
    # the ordinal label at > 0.
    # ════════════════════════════════════════════════════════════════════════
    # Modic change type is categorical (type I=oedema, type II=fat, type III=sclerosis)
    # — NOT an ordinal severity scale.  Classification only; no ranking head.
    "UpperMarrow": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_uma",
        "ranking": False,
    },
    "LowerMarrow": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_lma",
        "ranking": False,
    },
    "SpondylolisthesisBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Absent", "Present"],
        "model_head": "clf_spb", "ranking_head": "rank_spb",
    },
    "UpperEndplateDefectBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_ueb", "ranking_head": "rank_ueb",
    },
    "LowerEndplateDefectBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_leb", "ranking_head": "rank_leb",
    },
    "ForaminalStenosisLeftBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Absent", "Present"],
        "model_head": "clf_flb", "ranking_head": "rank_flb",
    },
    "ForaminalStenosisRightBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Absent", "Present"],
        "model_head": "clf_frb", "ranking_head": "rank_frb",
    },
    "HerniationBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_hrb", "ranking_head": "rank_hrb",
    },
    "AnteriorBulgingBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_abb", "ranking_head": "rank_abb",
    },
    "PosteriorBulgingBinary": {
        "num_classes": 2, "binary": True, "default_loss": "ce",
        "class_names": ["Normal", "Abnormal"],
        "model_head": "clf_pbb", "ranking_head": "rank_pbb",
    },

    # ── auxiliary (classification-only, no ranking) ─────────────────────────
    "IVDlevel": {
        "num_classes": 6, "binary": False, "default_loss": "ce",
        "class_names": ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"],
        "model_head": "clf_ivd",
        "ranking": False,  # anatomical location, not ordinal severity
    },
}

# Tasks that should NOT have ranking heads (classification-only).
# Check via TASK_DEFINITIONS[task].get("ranking", True).
CLASSIFICATION_ONLY_TASKS = frozenset(
    t for t, d in TASK_DEFINITIONS.items() if not d.get("ranking", True)
)


# ── task presets ─────────────────────────────────────────────────────────────

# Standard: Pfirrmann / Narrowing / CCS stay multi-class;
#           ALL other tasks are binarised (Normal vs Abnormal).
STANDARD_TASKS_13 = [
    "Pfirrmann", "Narrowing", "CentralCanalStenosis",
    "SpondylolisthesisBinary",
    "UpperEndplateDefectBinary", "LowerEndplateDefectBinary",
    "UpperMarrow", "LowerMarrow",
    "ForaminalStenosisLeftBinary", "ForaminalStenosisRightBinary",
    "HerniationBinary",
    "AnteriorBulgingBinary", "PosteriorBulgingBinary",
    "IVDlevel",
]

# Ordinal: ALL tasks use full multi-class ordinal grading scales.
ORDINAL_TASKS_13 = [
    "Pfirrmann", "Narrowing", "CentralCanalStenosis",
    "Spondylolisthesis",
    "UpperEndplateDefect", "LowerEndplateDefect",
    "UpperModic", "LowerModic",
    "ForaminalStenosisLeft", "ForaminalStenosisRight",
    "Herniation",
    "AnteriorBulging", "PosteriorBulging",
    "IVDlevel",
]
ORDINAL_TASKS_16 = ORDINAL_TASKS_13 + [
    "AnnularTears",
    "FacetJointArthropathyLeft", "FacetJointArthropathyRight",
]

# ── canonical task list (always ordinal) ─────────────────────────────────────
# All new code should use DEFAULT_GRADING_TASKS.  Binary task names
# (*Binary) are retained in TASK_DEFINITIONS only for backward-compatible
# checkpoint loading.
DEFAULT_GRADING_TASKS = list(ORDINAL_TASKS_13)

# ── legacy aliases (backward-compatible) ────────────────────────────────────
DEFAULT_TASKS_8 = [
    "Pfirrmann", "Narrowing", "CentralCanalStenosis", "Spondylolisthesis",
    "UpperEndplateDefect", "LowerEndplateDefect", "UpperMarrow", "LowerMarrow",
]
DEFAULT_TASKS_11 = DEFAULT_TASKS_8 + [
    "ForaminalStenosisLeft", "ForaminalStenosisRight", "Herniation",
]
DEFAULT_TASKS_13 = DEFAULT_TASKS_11 + ["AnteriorBulging", "PosteriorBulging"]

GRANULAR_TASKS_14 = [
    "Pfirrmann", "Narrowing", "CentralCanalStenosis", "Spondylolisthesis",
    "UpperEndplateDefect", "LowerEndplateDefect",
    "UpperModic", "LowerModic",
    "ForaminalStenosisLeft", "ForaminalStenosisRight",
    "Herniation", "AnteriorBulging", "PosteriorBulging",
    "AnnularTears",
]
GRANULAR_TASKS_16 = GRANULAR_TASKS_14 + [
    "FacetJointArthropathyLeft", "FacetJointArthropathyRight",
]

CLASS_NAMES_DICT: Dict[str, List[str]] = {
    task: td["class_names"] for task, td in TASK_DEFINITIONS.items()
}


# ============================================================================
# ETH CORPORATE DESIGN
# ============================================================================

ETH_COLORS = {
    "petrol": "#007A92", "petrol_light": "#99CAD5", "petrol_dark": "#00586A",
    "black": "#000000", "gray": "#6F6F6F", "white": "#FFFFFF",
    "blue": "#0069B4", "green": "#91C34A", "red": "#C1002A",
    "bronze": "#A98B3D", "purple": "#6B3FA0", "brown": "#956013",
}

# ── Severity-Based Color Palette ──────────────────────────────────────────
# Consistent across ALL score boxplots, density plots, ranked grids, etc.
# Rule: green = absent/none/normal (lowest grade), red = severe/present
#        (highest grade), intermediate grades use petrol, bronze, blue, purple.
#
# The mapping is by *ordinal position*, NOT by class name — so it works for
# any task regardless of how the grades are labelled.

ETH_SEVERITY_PALETTE = [
    "#91C34A",  # Grade 0 — ETH green   (None / Normal / Absent / Grade I)
    "#007A92",  # Grade 1 — ETH petrol  (Mild / Slight / Type I / Grade II)
    "#A98B3D",  # Grade 2 — ETH bronze  (Moderate / Type II / Grade III)
    "#0069B4",  # Grade 3 — ETH blue    (Severe / Large / Type III / Grade IV)
    "#6B3FA0",  # Grade 4 — ETH purple  (Very Severe / Grade V)
    "#C1002A",  # Grade 5 — ETH red     (extra, if >5 classes)
]


def get_severity_colors(num_classes: int) -> List[str]:
    """Return a list of *num_classes* ETH severity colours.

    Parameters
    ----------
    num_classes : int
        Number of ordinal classes to colour.

    Returns
    -------
    list of str
        Hex colour strings ordered from lowest (green) to highest (red)
        severity. Specific palettes per arity:

        * 2-class: green, red
        * 3-class: green, bronze, red
        * 4-class: green, petrol, bronze, red
        * 5-class: green, petrol, bronze, purple, red
        * 6+:      green, petrol, bronze, blue, purple, red, then cycle
    """
    if num_classes <= 1:
        return [ETH_SEVERITY_PALETTE[0]]
    if num_classes == 2:
        return [ETH_SEVERITY_PALETTE[0], ETH_COLORS["red"]]
    if num_classes == 3:
        return [ETH_SEVERITY_PALETTE[0], ETH_SEVERITY_PALETTE[2], ETH_COLORS["red"]]
    if num_classes == 4:
        return [ETH_SEVERITY_PALETTE[0], ETH_SEVERITY_PALETTE[1],
                ETH_SEVERITY_PALETTE[2], ETH_COLORS["red"]]
    if num_classes == 5:
        return [ETH_SEVERITY_PALETTE[0], ETH_SEVERITY_PALETTE[1],
                ETH_SEVERITY_PALETTE[2], ETH_SEVERITY_PALETTE[4], ETH_COLORS["red"]]
    # 6+: use full palette then cycle
    colors = list(ETH_SEVERITY_PALETTE[:num_classes])
    while len(colors) < num_classes:
        colors.append(ETH_SEVERITY_PALETTE[len(colors) % len(ETH_SEVERITY_PALETTE)])
    return colors


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

#: Config fields that hold file-system paths. ``$VAR`` / ``${VAR}`` and a
#: leading ``~`` in these fields are expanded when a YAML config is loaded, so
#: a config can refer to environment variables (e.g. ``${GENODISC_ROOT}``)
#: instead of machine-specific paths.
PATH_KEYS = frozenset({
    "GENODICT_PATH", "IVD_PATH", "JSON_PATH", "IMAGE_FOLDER", "CACHE_DIR",
    "SAVE_DIR", "SAVE_WEIGHTS_PATH", "EVAL_DIR", "LOGDIR", "CHECKPOINT",
})

#: ``$NAME`` or ``${NAME}``; any other ``$`` in a path is kept as written.
_ENV_REF = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")


def expand_env_paths(raw: Dict[str, Any], skip: Any = ()) -> Dict[str, Any]:
    """Expand environment variables and ``~`` in the path fields of a config.

    Parameters
    ----------
    raw : dict
        Mapping of config field names to values, as read from YAML.
    skip : iterable of str, optional
        Fields to leave as written, e.g. because the command line overrides
        them or the config class drops them.

    Returns
    -------
    dict
        A copy of *raw* in which every string value of a :data:`PATH_KEYS`
        field has ``$VAR`` / ``${VAR}`` and a leading ``~`` expanded.
        Any other ``$`` is kept as written.

    Raises
    ------
    ValueError
        If a path field refers to an environment variable that is unset or
        empty.
    """
    out = dict(raw)
    skip = set(skip)
    for key in [k for k in out if k in PATH_KEYS and k not in skip]:
        value = out[key]
        if not isinstance(value, str):
            continue
        names = {a or b for a, b in _ENV_REF.findall(value)}
        missing = sorted(n for n in names if not os.environ.get(n))
        if missing:
            raise ValueError(
                f"{key}={value!r}: environment variable not set or empty: "
                + ", ".join(missing)
            )
        expanded = _ENV_REF.sub(lambda m: os.environ[m.group(1) or m.group(2)], value)
        out[key] = os.path.expanduser(expanded)
    return out


def _config_from_yaml(cls: type, yaml_path: str) -> Any:
    """Load a dataclass instance from a YAML file, ignoring unknown keys.

    Parameters
    ----------
    cls : type
        Dataclass type to instantiate.
    yaml_path : str
        Path to a YAML file containing a flat mapping of field names to
        values. Keys that are not declared on *cls* are silently dropped.
        Environment variables in path fields are expanded
        (:func:`expand_env_paths`).

    Returns
    -------
    object
        An instance of ``cls`` populated from the YAML file.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    valid_keys = {fld.name for fld in fields(cls)}
    filtered = expand_env_paths({k: v for k, v in raw.items() if k in valid_keys})
    return cls(**filtered)


@dataclass
class GradingConfig:
    """Configuration for multi-task classification training.

    Holds every hyperparameter and path needed by the classification
    training entry point: data locations, geometry, dataset mode,
    augmentation, caching, task selection, model / head choices,
    optimisation, multi-task weighting, and output directories.

    Notes
    -----
    The dataclass is YAML-loadable through :meth:`from_yaml`; unknown
    keys are silently ignored. :meth:`__post_init__` coerces numeric
    fields that YAML may load as strings.
    """

    # ── data paths ──
    JSON_PATH: str = ""
    IMAGE_FOLDER: str = ""

    # ── geometry ──
    N_SLICES: int = 12
    HEIGHT: int = 128
    WIDTH: int = 256

    # ── dataset mode ──
    USE_MULTI_CONTRAST: bool = False
    CONTRASTS: List[str] = field(default_factory=lambda: ["T1", "T2"])
    RETURN_RAW: bool = False
    FILTER_SEQUENCES: Optional[List[str]] = None

    # ── augmentation ──
    AUGMENTATION_MODE: str = "standard"
    USE_CLAHE: bool = False

    # ── caching ──
    USE_CACHE: bool = True
    CACHE_DIR: str = ""
    CACHE_RATE: float = 1.0
    CACHE_NUM_WORKERS: int = 4

    # ── tasks ──
    TASKS: Optional[List[str]] = None
    USE_RAW_LABELS: bool = False
    INCLUDE_AUXILIARY_TASKS: bool = False
    TASK_LOSS_OVERRIDE: Optional[Dict[str, str]] = None

    # ── model ──
    MODEL_TYPE: str = "resnet34"
    HEAD_TYPE: str = "linear"
    HEAD_HIDDEN_DIM: int = 256
    FINETUNE: bool = False
    CHECKPOINT: str = ""

    # ── training ──
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    EPOCHS: int = 200
    OPTIM_TYPE: str = "AdamW"
    LR: float = 1e-3
    SGD_MOMENTUM: float = 0.9
    ADAM_BETAS: Tuple = (0.9, 0.999)
    WEIGHT_DECAY: float = 1e-4
    USE_LR_SCHEDULER: bool = True
    LR_SCHEDULER_TYPE: str = "cosine"  # cosine | step | plateau
    LR_SCHEDULER_STEP: int = 50
    LR_DECAY: float = 0.1
    LR_PLATEAU_PATIENCE: int = 30
    LR_PLATEAU_MIN: float = 1e-7
    USE_CLASS_WEIGHTS: bool = True

    # ── multi-task loss weighting ──
    TASK_WEIGHT_STRATEGY: str = "uniform"  # uniform | uncertainty | dwa | difficulty
    DWA_TEMPERATURE: float = 2.0

    # ── output ──
    SAVE_WEIGHTS_PATH: str = "./weights"
    EVAL_DIR: str = "./evaluation_results"
    LOGDIR: str = ".logs"

    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Coerce numeric fields and populate derived defaults.

        Ensures values loaded from YAML (which may arrive as strings or
        lists) are normalised to their declared types and that mutable
        defaults like ``TASK_LOSS_OVERRIDE`` and ``tags`` are populated.
        """
        self.BATCH_SIZE = int(self.BATCH_SIZE)
        self.NUM_WORKERS = int(self.NUM_WORKERS)
        self.EPOCHS = int(self.EPOCHS)
        self.N_SLICES = int(self.N_SLICES)
        self.HEIGHT = int(self.HEIGHT)
        self.WIDTH = int(self.WIDTH)
        self.LR = float(self.LR)
        self.WEIGHT_DECAY = float(self.WEIGHT_DECAY)
        self.HEAD_HIDDEN_DIM = int(self.HEAD_HIDDEN_DIM)

        if isinstance(self.ADAM_BETAS, list):
            self.ADAM_BETAS = tuple(float(x) for x in self.ADAM_BETAS)
        if isinstance(self.CONTRASTS, str):
            self.CONTRASTS = [c.strip() for c in self.CONTRASTS.split(",")]
        if self.TASK_LOSS_OVERRIDE is None:
            self.TASK_LOSS_OVERRIDE = {}

        if self.tags is None:
            self.tags = [self.MODEL_TYPE]
        if self.USE_MULTI_CONTRAST:
            self.tags.append("mc")
        if self.USE_RAW_LABELS:
            self.tags.append("ordinal")
        if self.FINETUNE:
            self.tags.append("finetune")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GradingConfig":
        """Instantiate a :class:`GradingConfig` from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to a YAML file with field-name keys. Unknown keys are
            silently dropped.

        Returns
        -------
        GradingConfig
        """
        return _config_from_yaml(cls, yaml_path)


@dataclass
class RankingConfig:
    """Configuration for ordinal ranking training.

    Extends the classification configuration vocabulary with ranking
    heads, cross-task attention, end-to-end multitask flags, ranking
    losses, rank–classification agreement / calibration, thresholding,
    OUST selective prediction, XAI, and validation binning.

    Notes
    -----
    The dataclass is YAML-loadable through :meth:`from_yaml`; unknown
    keys are silently ignored. :meth:`__post_init__` coerces numeric
    fields that YAML may load as strings and ensures the output
    directories exist.
    """

    # ── data paths (kept for YAML compatibility; not read by the trainer) ──
    JSON_PATH: str = ""
    IMAGE_FOLDER: str = ""

    # ── geometry ──
    N_SLICES: int = 12
    HEIGHT: int = 128
    WIDTH: int = 256

    # ── dataset mode ──
    FILTER_SEQUENCES: Optional[List[str]] = None
    USE_MULTI_CONTRAST: bool = False
    CONTRASTS: List[str] = field(default_factory=lambda: ["T1", "T2"])
    USE_CACHE: bool = True
    CACHE_DIR: str = ""
    CACHE_RATE: float = 1.0
    CACHE_NUM_WORKERS: int = 4
    AUGMENTATION_MODE: str = "standard"

    # ── backbone ──
    MODEL_TYPE: str = "resnet34"
    CHECKPOINT: str = ""
    HEAD_TYPE: str = "linear"
    HEAD_HIDDEN_DIM: int = 256

    # ── ranking heads ──
    RANKING_HEAD_TYPE: str = "linear"
    RANK_HIDDEN_DIM: int = 128
    RANK_DROPOUT: float = 0.3
    BOUNDED: bool = True
    RANK_FROM_LOGITS: bool = False

    # ── cross-task attention ──
    CROSS_TASK_MODE: str = "none"        # none | transformer | mlp | gated | mean_pool
    CROSS_TASK_D_MODEL: int = 64
    CROSS_TASK_NHEAD: int = 4
    CROSS_TASK_NUM_LAYERS: int = 2
    CROSS_TASK_DIM_FF: int = 128
    CROSS_TASK_DROPOUT: float = 0.1
    CROSS_TASK_INPUT: str = "score"      # score | logits | features (future)
    CROSS_TASK_STORE_ATTN: bool = False

    # ── end-to-end mode ──
    END_TO_END: bool = False
    CLF_LOSS_WEIGHT: float = 1.0
    USE_CLASS_WEIGHTS: bool = True
    OVERSAMPLE_DIFFICULT: bool = False

    # ── ranking training ──
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    EPOCHS: int = 100
    LR: float = 1e-4
    ENCODER_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    RANKING_LOSS: str = "DeepRankSVM"
    RANKING_MARGIN: float = 1.0
    RANKING_C1: float = 1.0
    RANKING_C2: float = 0.25
    MAX_PAIRS_PER_BATCH: int = 500
    HARD_NEGATIVE_RATIO: float = 0.7
    HARD_MARGIN: float = 1.0
    SKIP_EQUAL_PAIRS: bool = False   # only pairs with label difference (no same-grade pairs)
    LEVEL_AWARE_PAIRS: bool = False  # only form ranking pairs within same IVD level
    BOUNDING_MODE: str = "softplus"  # softplus | sigmoid | none

    # ── rank–classification agreement ──
    AGREEMENT_LOSS_WEIGHT: float = 0.0   # 0 = disabled (backward-compatible)
    AGREEMENT_DETACH_CLF: bool = True     # True: ranking follows clf; False: mutual

    # ── rank-calibrated classification (inference-time) ──
    USE_RANK_CALIBRATION: bool = False
    CALIBRATION_SIGMA: float = 1.0       # Gaussian width on [0, 10] scale
    CALIBRATION_ALPHA: float = 0.5       # clf weight in fusion (0.5 = equal)

    # ── thresholding ──
    THRESHOLD_METRIC: str = "balanced_accuracy"
    THRESHOLD_METHOD: str = "grid"               # grid | isotonic | gmm | youden
    THRESHOLD_GRID_STEPS: int = 200

    # ── ordinal uncertainty & selective prediction (OUST) ──
    SELECTIVE_PREDICTION: bool = False     # enable OUST analysis in eval
    ABSTENTION_THRESHOLD: float = 0.5     # confidence threshold for abstention
    ABSTENTION_MARGIN_BETA: float = 3.0   # sharpness of ranking margin sigmoid
    COMPUTE_OECE: bool = False            # compute Ordinal ECE in eval

    # ── XAI method ──
    XAI_METHOD: str = "ig"                # ig | gradcam

    # ── validation binning ──
    VAL_BINNING_MODE: str = "target_aligned"     # uniform | target_aligned | centroid

    # ── tasks ──
    TASKS: Optional[List[str]] = None
    USE_RAW_LABELS: bool = False
    TASK_LOSS_OVERRIDE: Optional[Dict[str, str]] = None

    # ── output ──
    SAVE_DIR: str = "weights/ranking"
    EVAL_DIR: str = "evaluation/ranking"
    LOGDIR: str = ".logs/ranking"

    SEED: int = 42

    def __post_init__(self) -> None:
        """Coerce numeric fields, normalise lists, and create output dirs.

        Ensures all numeric fields are cast to their declared Python
        types, splits comma-separated ``CONTRASTS`` strings into lists,
        creates ``SAVE_DIR``/``EVAL_DIR``/``LOGDIR``, and populates
        ``TASKS`` / ``TASK_LOSS_OVERRIDE`` defaults.
        """
        from pathlib import Path

        # ── coerce numeric types (YAML may load scientific notation as str) ──
        self.N_SLICES = int(self.N_SLICES)
        self.HEIGHT = int(self.HEIGHT)
        self.WIDTH = int(self.WIDTH)
        self.BATCH_SIZE = int(self.BATCH_SIZE)
        self.NUM_WORKERS = int(self.NUM_WORKERS)
        self.EPOCHS = int(self.EPOCHS)
        self.HEAD_HIDDEN_DIM = int(self.HEAD_HIDDEN_DIM)
        self.RANK_HIDDEN_DIM = int(self.RANK_HIDDEN_DIM)
        self.MAX_PAIRS_PER_BATCH = int(self.MAX_PAIRS_PER_BATCH)
        self.THRESHOLD_GRID_STEPS = int(self.THRESHOLD_GRID_STEPS)
        self.SEED = int(self.SEED)

        self.LR = float(self.LR)
        self.ENCODER_LR = float(self.ENCODER_LR)
        self.WEIGHT_DECAY = float(self.WEIGHT_DECAY)
        self.RANK_DROPOUT = float(self.RANK_DROPOUT)
        self.CLF_LOSS_WEIGHT = float(self.CLF_LOSS_WEIGHT)
        self.RANKING_MARGIN = float(self.RANKING_MARGIN)
        self.RANKING_C1 = float(self.RANKING_C1)
        self.RANKING_C2 = float(self.RANKING_C2)
        self.HARD_NEGATIVE_RATIO = float(self.HARD_NEGATIVE_RATIO)
        self.HARD_MARGIN = float(self.HARD_MARGIN)
        self.CACHE_RATE = float(self.CACHE_RATE)
        self.CACHE_NUM_WORKERS = int(self.CACHE_NUM_WORKERS)

        # ── agreement / calibration coercion ──
        self.AGREEMENT_LOSS_WEIGHT = float(self.AGREEMENT_LOSS_WEIGHT)
        self.CALIBRATION_SIGMA = float(self.CALIBRATION_SIGMA)
        self.CALIBRATION_ALPHA = float(self.CALIBRATION_ALPHA)

        # ── cross-task attention coercion ──
        self.CROSS_TASK_D_MODEL = int(self.CROSS_TASK_D_MODEL)
        self.CROSS_TASK_NHEAD = int(self.CROSS_TASK_NHEAD)
        self.CROSS_TASK_NUM_LAYERS = int(self.CROSS_TASK_NUM_LAYERS)
        self.CROSS_TASK_DIM_FF = int(self.CROSS_TASK_DIM_FF)
        self.CROSS_TASK_DROPOUT = float(self.CROSS_TASK_DROPOUT)

        # ── dataset mode coercion ──
        if isinstance(self.CONTRASTS, str):
            self.CONTRASTS = [c.strip() for c in self.CONTRASTS.split(",")]

        # ── output directories ──
        for d in [self.SAVE_DIR, self.EVAL_DIR, self.LOGDIR]:
            Path(d).mkdir(parents=True, exist_ok=True)
        if self.TASKS is None:
            self.TASKS = list(DEFAULT_TASKS_8)
        if self.TASK_LOSS_OVERRIDE is None:
            self.TASK_LOSS_OVERRIDE = {}

    @classmethod
    def from_yaml(cls, path: str) -> "RankingConfig":
        """Instantiate a :class:`RankingConfig` from a YAML file.

        Parameters
        ----------
        path : str
            Path to a YAML file with field-name keys. Unknown keys are
            silently dropped.

        Returns
        -------
        RankingConfig
        """
        return _config_from_yaml(cls, path)


# ============================================================================
# TASK HELPERS
# ============================================================================

def resolve_tasks(
    config: Any,
    ordinal_default: List[str] = ORDINAL_TASKS_13,
    standard_default: List[str] = STANDARD_TASKS_13,
) -> List[str]:
    """Return the task list from *config* (:class:`GradingConfig` or
    :class:`RankingConfig`).

    Parameters
    ----------
    config : GradingConfig or RankingConfig
        Configuration object exposing an optional ``TASKS`` attribute.
    ordinal_default : list of str, optional
        Default task list when ``config.TASKS`` is ``None``. Defaults
        to :data:`ORDINAL_TASKS_13` — the canonical multi-class ordinal
        grading scale used for all new experiments.
    standard_default : list of str, optional
        Legacy parameter (with ``*Binary`` task names) retained for
        backward compatibility; no longer used by default.

    Returns
    -------
    list of str
        Resolved task names. ``"IVDlevel"`` is appended when
        ``config.INCLUDE_AUXILIARY_TASKS`` is true and the task is not
        already present.

    Notes
    -----
    Resolution order:

    1. Explicit ``TASKS`` list in *config* → filtered against
       :data:`TASK_DEFINITIONS` and used as-is.
    2. Otherwise → *ordinal_default*.
    """
    tasks_attr = getattr(config, "TASKS", None)
    if tasks_attr is not None:
        tasks = [t for t in tasks_attr if t in TASK_DEFINITIONS]
    else:
        tasks = list(ordinal_default)

    if getattr(config, "INCLUDE_AUXILIARY_TASKS", False) and "IVDlevel" not in tasks:
        tasks.append("IVDlevel")
    return tasks
