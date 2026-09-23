"""Label decoding and figure style used by ``scripts/eda_dataset.py``.

Self-contained helpers so that the EDA script only needs numpy + matplotlib
(no torch). Contents:

* ``extract_labels`` / ``INVALID_LABEL`` / ``IVD_LEVELS``: decode one Genodisc
  ``score`` dict into 0-based ordinal labels. Pfirrmann is stored 1-5 in the
  JSON and becomes 0-4; Modic type is derived from the ``{Upper,Lower}Modic{1,2,3}``
  flags (highest type wins); values outside ``TASK_DEFINITIONS[task]["num_classes"]``
  are set to ``INVALID_LABEL``.
* ETH Zurich corporate colours, ``FIG_FULL_W`` (LNCS text width) and
  ``apply_eth_style`` (matplotlib rcParams used for the EDA figures).
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from spineranknet.config import INVALID_LABEL, TASK_DEFINITIONS

__all__ = [
    "INVALID_LABEL", "IVD_LEVELS", "extract_labels",
    "ETH_BLUE", "ETH_PETROL", "ETH_GREEN", "ETH_BRONZE", "ETH_ORANGE", "ETH_RED",
    "ETH_PURPLE", "ETH_GREY", "ETH_LGREY", "ETH_BLACK", "FIG_FULL_W", "apply_eth_style",
]


# ── labels ───────────────────────────────────────────────────────────────────
IVD_LEVELS = ("T12L1", "L1L2", "L2L3", "L3L4", "L4L5", "L5S1")

# Ordinal grading tasks read directly from the JSON (0-based, no remapping).
_DIRECT_TASKS = (
    "Narrowing", "CentralCanalStenosis", "Spondylolisthesis",
    "UpperEndplateDefect", "LowerEndplateDefect",
    "ForaminalStenosisLeft", "ForaminalStenosisRight",
    "Herniation", "AnteriorBulging", "PosteriorBulging",
)


def normalize_label(value: Any, invalid: int = INVALID_LABEL) -> int:
    """Coerce a raw JSON value to a non-negative int label, else ``invalid``."""
    if value is None or isinstance(value, (list, tuple, dict)):
        return invalid
    if isinstance(value, (np.generic, np.ndarray)):
        try:
            value = value.item()
        except Exception:
            return invalid
    if isinstance(value, bool):
        return int(value)
    try:
        v = int(value)
    except (TypeError, ValueError):
        return invalid
    return v if v >= 0 else invalid


def derive_modic_type(score: Dict[str, Any], prefix: str,
                      invalid: int = INVALID_LABEL) -> int:
    """0 = no Modic change, 1/2/3 = Type I/II/III (highest type present wins);
    ``invalid`` when all three component flags are missing."""
    values = {}
    for cls in (1, 2, 3):
        v = normalize_label(score.get(f"{prefix}Modic{cls}", invalid), invalid=invalid)
        if v != invalid:
            values[cls] = v
    if not values:
        return invalid
    present = [cls for cls, v in values.items() if v > 0]
    return max(present) if present else 0


def extract_labels(score: Dict[str, Any], sequence: str = "T2", *,
                   invalid: int = INVALID_LABEL) -> Dict[str, int]:
    """Map a Genodisc ``score`` dict to 0-based labels for the 11 paper tasks
    plus ``UpperModic`` / ``LowerModic``. Pfirrmann and Narrowing are set to
    ``invalid`` for T1-weighted sequences."""
    pfi = normalize_label(score.get("Pfirrmann", invalid), invalid=invalid)
    labels = {"Pfirrmann": (pfi - 1) if (pfi != invalid and pfi >= 1) else pfi}
    for task in _DIRECT_TASKS:
        labels[task] = normalize_label(score.get(task, invalid), invalid=invalid)
    labels["UpperModic"] = derive_modic_type(score, "Upper", invalid)
    labels["LowerModic"] = derive_modic_type(score, "Lower", invalid)

    if sequence.upper().startswith("T1"):
        labels["Pfirrmann"] = invalid
        labels["Narrowing"] = invalid

    for task, val in labels.items():
        if val != invalid and not 0 <= val < TASK_DEFINITIONS[task]["num_classes"]:
            labels[task] = invalid
    return labels


# ── figure style ─────────────────────────────────────────────────────────────
# ETH Zurich corporate colours
# https://ethz.ch/en/the-eth-zurich/corporate-identity/corporate-design/colour.html
ETH_BLUE = "#215CAF"
ETH_PETROL = "#007894"
ETH_GREEN = "#627313"
ETH_BRONZE = "#8E6713"
ETH_ORANGE = "#D4780A"
ETH_RED = "#B7352D"
ETH_PURPLE = "#A7117A"
ETH_GREY = "#6F6F6F"
ETH_LGREY = "#B8B8B8"
ETH_BLACK = "#000000"
PALETTE = [ETH_BLUE, ETH_PETROL, ETH_GREEN, ETH_BRONZE, ETH_RED, ETH_PURPLE, ETH_GREY]

FIG_FULL_W = 6.97   # LNCS full text width (inches)


def apply_eth_style(*, fontsize: int = 9) -> None:
    """Set matplotlib rcParams: sans-serif, no top/right spines, dotted grid,
    TrueType (Type-42) fonts in PDF, 300 dpi output."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "figure.titlesize": fontsize + 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.70,
        "axes.edgecolor": ETH_GREY,
        "axes.labelcolor": ETH_BLACK,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.grid": True,
        "grid.color": ETH_LGREY,
        "grid.alpha": 0.45,
        "grid.linewidth": 0.40,
        "grid.linestyle": ":",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.70,
        "ytick.major.width": 0.70,
        "xtick.color": ETH_GREY,
        "ytick.color": ETH_GREY,
        "lines.linewidth": 1.80,
        "lines.markersize": 4.50,
        "lines.markeredgewidth": 0.0,
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.pad_inches": 0.04,
    })
