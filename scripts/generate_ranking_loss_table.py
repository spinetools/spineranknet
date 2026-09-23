#!/usr/bin/env python3
"""Generate LaTeX tables comparing ranking losses across all 13 spinal pathologies.

Reads hybrid_metrics.json (or evaluation_metrics.json) from each experiment
directory and produces a multi-row, multi-column LaTeX table with:
  - Rows grouped by ranking loss (best head per loss by mean QWK)
  - Columns: 13 tasks + Mean
  - Sub-rows: BAcc, QWK, AUC, MAE (configurable)
  - Optional: include classification baseline from standalone clf results
  - Optional: include classification-side metrics from ranking experiments

Expected layout (as written by ``scripts/run_all_experiments.py``)::

    <results_dir>/<backbone>/<loss>_<head>/eval/<backbone>/hybrid_metrics.json

Usage:
    # Default: results/ranking, best head per loss
    python scripts/generate_ranking_loss_table.py

    # Another results directory
    python scripts/generate_ranking_loss_table.py --results_dir my_runs/ranking

    # Include a classification baseline row from a per-task CSV
    python scripts/generate_ranking_loss_table.py --include_clf_baseline \\
        --clf_baseline_path path/to/clf_metrics.csv

    # Specific losses only
    python scripts/generate_ranking_loss_table.py --losses deepranksvm mse ranknet

    # Output to file
    python scripts/generate_ranking_loss_table.py -o paper/tables/ranking_loss_table.tex

    # Use evaluation_metrics.json (has both clf + ranking sections)
    python scripts/generate_ranking_loss_table.py --source evaluation_metrics

    # Show classification-side metrics alongside ranking
    python scripts/generate_ranking_loss_table.py --show_clf_side

    # Custom metrics
    python scripts/generate_ranking_loss_table.py --metrics qwk mae spearman_rho
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

TASK_ORDER = [
    "Pfirrmann", "Narrowing", "CentralCanalStenosis",
    "Spondylolisthesis",
    "UpperEndplateDefect", "LowerEndplateDefect",
    "ForaminalStenosisLeft", "ForaminalStenosisRight",
    "Herniation",
    "AnteriorBulging", "PosteriorBulging",
    "UpperModic", "LowerModic",
]

TASK_SHORT = {
    "Pfirrmann":              "Pfi",
    "Narrowing":              "Nar",
    "CentralCanalStenosis":   "CCS",
    "Spondylolisthesis":      "Spo",
    "UpperEndplateDefect":    "UEP",
    "LowerEndplateDefect":    "LEP",
    "ForaminalStenosisLeft":  "FoL",
    "ForaminalStenosisRight": "FoR",
    "Herniation":             "Hrn",
    "AnteriorBulging":        "AnB",
    "PosteriorBulging":       "PoB",
    "UpperModic":             "UMr",
    "LowerModic":             "LMr",
}

# LaTeX-friendly display names
TASK_LATEX = {
    "Pfirrmann":              r"\textbf{Pfi}",
    "Narrowing":              r"\textbf{Nar}",
    "CentralCanalStenosis":   r"\textbf{CCS}",
    "Spondylolisthesis":      r"\textbf{Spo}",
    "UpperEndplateDefect":    r"\textbf{UEP}",
    "LowerEndplateDefect":    r"\textbf{LEP}",
    "ForaminalStenosisLeft":  r"\textbf{FoL}",
    "ForaminalStenosisRight": r"\textbf{FoR}",
    "Herniation":             r"\textbf{Hrn}",
    "AnteriorBulging":        r"\textbf{AnB}",
    "PosteriorBulging":       r"\textbf{PoB}",
    "UpperModic":             r"\textbf{UMr$^*$}",
    "LowerModic":             r"\textbf{LMr$^*$}",
}

LOSS_DISPLAY = {
    "deepranksvm":              "DeepRankSVM",
    "mse":                      "MSE",
    "mae":                      "MAE",
    "ranknet":                  "RankNet",
    "deeprelativeattributes":   "DRA",
    "justnoticeabledifferences": "JND",
    "relativeattributes":       "RA",
    "tripletordinal":           "TripletOrd",
    "spinerank":                "SpineRank",
    # Old names (backward compat)
    "dra":                      "DRA",
    "jnd":                      "JND",
    "parikh2011":               "RA",
}

# Canonical ordering for losses in the table
LOSS_ORDER = [
    "mse", "mae", "ranknet", "deepranksvm",
    "relativeattributes", "deeprelativeattributes",
    "justnoticeabledifferences", "tripletordinal", "spinerank",
]

ALL_HEADS = ["linear", "mlp", "transformer", "kan", "least_squares"]

# Metrics configuration
METRIC_DISPLAY = {
    "balanced_accuracy": "BA",
    "qwk":              "QW",
    "roc_auc":          "AUC",
    "mae":              "MAE",
    "mcc":              "MCC",
    "spearman_rho":     r"$\rho$",
    "kendall_tau":      r"$\tau$",
    "binary_bal_acc":   "Bin.BA",
    "binary_f1":        "Bin.F1",
}

# Default metrics to show in the table
DEFAULT_METRICS = ["balanced_accuracy", "qwk", "spearman_rho", "mae"]

# Higher-is-better or lower-is-better
HIGHER_IS_BETTER = {
    "balanced_accuracy": True,
    "qwk": True,
    "roc_auc": True,
    "mcc": True,
    "mae": False,
    "spearman_rho": True,
    "kendall_tau": True,
    "binary_bal_acc": True,
    "binary_f1": True,
}

# Selection metric for best head
BEST_HEAD_METRIC = "qwk"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hybrid_metrics(json_path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Load hybrid_metrics.json → {task: {metric: value}}."""
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def load_evaluation_metrics(json_path: Path, section: str = "ranking"
                            ) -> Optional[Dict[str, Dict[str, float]]]:
    """Load evaluation_metrics.json → {task: {metric: value}} from given section.

    Parameters
    ----------
    json_path : Path
        Path to evaluation_metrics.json.
    section : str
        ``"classification"`` or ``"ranking"``.
    """
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get(section)


def load_clf_baseline(csv_path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Load a classification baseline from a per-task metrics CSV.

    Expected columns: ``Task``, ``Grade`` and any of ``Bal.Acc``, ``QWK``,
    ``ROC_AUC``, ``MAE``, ``Spearman``. Only rows with ``Grade == "All"``
    are used. Returns {task_name: {metric: value}} with normalized task names.
    """
    if not csv_path.exists():
        return None

    import csv
    results = {}
    # Map CSV task names to our canonical names
    task_map = {
        "Pfirrmann": "Pfirrmann",
        "Narrowing": "Narrowing",
        "CentralCanalStenosis": "CentralCanalStenosis",
        "SpondylolisthesisBinary": "Spondylolisthesis",
        "UpperEndplateDefect": "UpperEndplateDefect",
        "LowerEndplateDefect": "LowerEndplateDefect",
        "UpperMarrow": "UpperModic",
        "LowerMarrow": "LowerModic",
        "ForaminalStenosisLeft": "ForaminalStenosisLeft",
        "ForaminalStenosisRight": "ForaminalStenosisRight",
        "Herniation": "Herniation",
        "AnteriorBulging": "AnteriorBulging",
        "PosteriorBulging": "PosteriorBulging",
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Grade"] != "All":
                continue
            csv_task = row["Task"]
            task = task_map.get(csv_task, csv_task)
            if task not in TASK_ORDER:
                continue

            results[task] = {}
            # Map CSV column names → our metric names
            if row.get("Bal.Acc"):
                results[task]["balanced_accuracy"] = float(row["Bal.Acc"])
            if row.get("QWK"):
                results[task]["qwk"] = float(row["QWK"])
            if row.get("ROC_AUC"):
                results[task]["roc_auc"] = float(row["ROC_AUC"])
            if row.get("MAE"):
                results[task]["mae"] = float(row["MAE"])
            if row.get("Spearman"):
                results[task]["spearman_rho"] = float(row["Spearman"])

    return results if results else None


def discover_experiments(results_dir: Path, backbone: str = "resnet18"
                         ) -> Dict[str, Dict[str, Path]]:
    """Discover all loss×head experiments under results_dir/backbone/.

    Returns
    -------
    experiments : dict
        ``{loss_name: {head_name: path_to_eval_dir}}``
    """
    base = results_dir / backbone
    if not base.exists():
        return {}

    experiments = {}
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        # Skip non-experiment dirs
        if name in ("comparison", "comparison_clf_rank"):
            continue

        # Parse loss_head from directory name
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ALL_HEADS:
            loss, head = parts
        else:
            # Try longer head names or single-word experiments
            found = False
            for h in ALL_HEADS:
                suffix = f"_{h}"
                if name.endswith(suffix):
                    loss = name[:-len(suffix)]
                    head = h
                    found = True
                    break
            if not found:
                continue

        eval_dir = exp_dir / "eval" / backbone
        if not eval_dir.exists():
            continue

        if loss not in experiments:
            experiments[loss] = {}
        experiments[loss][head] = eval_dir

    return experiments


def select_best_head(loss_heads: Dict[str, Path],
                     metric: str = BEST_HEAD_METRIC,
                     source: str = "hybrid_metrics"
                     ) -> Tuple[Optional[str], Optional[Dict]]:
    """Select the best head for a loss by mean of selection metric across tasks.

    Returns (head_name, task_metrics) or (None, None) if nothing loadable.
    """
    best_head = None
    best_score = -float("inf")
    best_data = None

    for head, eval_dir in loss_heads.items():
        if source == "hybrid_metrics":
            data = load_hybrid_metrics(eval_dir / "hybrid_metrics.json")
        else:
            data = load_evaluation_metrics(
                eval_dir / "evaluation_metrics.json", section="ranking")
        if data is None:
            continue

        # Compute mean metric across valid tasks
        values = []
        for task in TASK_ORDER:
            if task in data and metric in data[task]:
                v = data[task][metric]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    values.append(v)

        if not values:
            continue

        mean_val = np.mean(values)
        # For MAE, lower is better
        score = -mean_val if not HIGHER_IS_BETTER.get(metric, True) else mean_val

        if score > best_score:
            best_score = score
            best_head = head
            best_data = data

    return best_head, best_data


def average_across_heads(loss_heads: Dict[str, Path],
                         source: str = "hybrid_metrics"
                         ) -> Tuple[Optional[str], Optional[Dict]]:
    """Average metrics across all available heads for a loss.

    Returns ("avg(N)", averaged_task_metrics) or (None, None).
    """
    all_head_data: List[Dict] = []
    head_names: List[str] = []

    for head, eval_dir in loss_heads.items():
        if source == "hybrid_metrics":
            data = load_hybrid_metrics(eval_dir / "hybrid_metrics.json")
        else:
            data = load_evaluation_metrics(
                eval_dir / "evaluation_metrics.json", section="ranking")
        if data is not None:
            all_head_data.append(data)
            head_names.append(head)

    if not all_head_data:
        return None, None

    if len(all_head_data) == 1:
        return head_names[0], all_head_data[0]

    # Average across heads per task per metric
    averaged: Dict[str, Dict[str, float]] = {}
    all_tasks = set()
    all_metrics_keys = set()
    for data in all_head_data:
        for task, mdict in data.items():
            all_tasks.add(task)
            all_metrics_keys.update(mdict.keys())

    for task in all_tasks:
        averaged[task] = {}
        for metric_key in all_metrics_keys:
            values = []
            for data in all_head_data:
                v = data.get(task, {}).get(metric_key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    values.append(v)
            if values:
                averaged[task][metric_key] = float(np.mean(values))

    label = f"avg({len(all_head_data)}:{'+'.join(sorted(head_names))})"
    return label, averaged


# =============================================================================
# TABLE FORMATTING
# =============================================================================

def fmt_val(val: Optional[float], metric: str, bold: bool = False,
            decimals: int = 3) -> str:
    """Format a metric value for LaTeX."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"

    s = f"{val:.{decimals}f}"

    if bold:
        return rf"\textbf{{{s}}}"
    return s


def compute_mean(task_data: Dict[str, Dict[str, float]],
                 metric: str) -> Optional[float]:
    """Compute mean of a metric across all 13 tasks."""
    values = []
    for task in TASK_ORDER:
        if task in task_data and metric in task_data[task]:
            v = task_data[task][metric]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(v)
    if not values:
        return None
    return float(np.mean(values))


def find_best_per_task(all_data: Dict[str, Dict[str, Dict[str, float]]],
                       metrics: List[str]
                       ) -> Dict[str, Dict[str, str]]:
    """Find the best loss for each (task, metric) combination.

    Returns {metric: {task: loss_name}} for bolding.
    """
    best = {}
    for metric in metrics:
        hib = HIGHER_IS_BETTER.get(metric, True)
        best[metric] = {}
        for task in TASK_ORDER + ["Mean"]:
            best_val = None
            best_loss = None
            for loss_name, task_data in all_data.items():
                if task == "Mean":
                    v = compute_mean(task_data, metric)
                else:
                    v = task_data.get(task, {}).get(metric)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if best_val is None:
                    best_val = v
                    best_loss = loss_name
                elif (hib and v > best_val) or (not hib and v < best_val):
                    best_val = v
                    best_loss = loss_name
            if best_loss is not None:
                best[metric][task] = best_loss
    return best


def generate_latex_table(
    all_data: Dict[str, Dict[str, Dict[str, float]]],
    loss_order: List[str],
    metrics: List[str],
    head_info: Dict[str, str],
    include_clf: Optional[Dict[str, Dict[str, float]]] = None,
    clf_side_data: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    bold_best: bool = True,
    caption: str = ("Per-task metrics results across all loss objectives test set "
                    "for a ResNet 3D 18 encoder; "
                    r"Balance accuracy (BA) / QWK / $\rho$ / MAE$\downarrow$ per loss;"),
    label: str = "tab:per_task_all_metrics",
    footnote: str = "",
) -> str:
    r"""Generate a full per-task LaTeX table matching MICCAI format.

    Output matches the multirow/resizebox style with task-group column
    spacing, ``\textbf`` bolding of per-task best, and a ``Met.`` column.

    Parameters
    ----------
    all_data : dict
        ``{loss_name: {task: {metric: value}}}`` — ranking-side metrics.
    loss_order : list
        Order of losses in the table.
    metrics : list
        Which metrics to show as sub-rows (e.g. BA, QW, ρ, MAE).
    head_info : dict
        ``{loss_name: head_name}`` — which head was selected.
    include_clf : dict, optional
        Classification baseline ``{task: {metric: value}}``.
    clf_side_data : dict, optional
        ``{loss_name: {task: {metric: value}}}`` — clf-side metrics.
    bold_best : bool
        Whether to bold the best value per task per metric.
    footnote : str
        Optional footnote text added below the table.
    """
    # Find best values for bolding
    if bold_best:
        comparison_data = dict(all_data)
        if include_clf is not None:
            comparison_data["__clf_baseline__"] = include_clf
        best_map = find_best_per_task(comparison_data, metrics)
    else:
        best_map = {m: {} for m in metrics}

    # ── Column specification with task-group spacing ──
    # Loss | Met. | Pfi Nar CCS Spo | UEP LEP FoL FoR | Hrn AnB PoB | UMr LMr | Mean
    col_spec = (
        r"@{}ll"
        r"    cccc"          # Pfi Nar CCS Spo
        r"    @{\hskip 5pt}cccc"   # UEP LEP FoL FoR
        r"    @{\hskip 5pt}ccc"    # Hrn AnB PoB
        r"    @{\hskip 5pt}cc"     # UMr LMr
        r"    @{\hskip 5pt}c@{}"   # Mean
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.92}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # ── Header row ──
    header_parts = [r"\textbf{Loss}", r"\textbf{Met.}"]
    for task in TASK_ORDER:
        header_parts.append(TASK_LATEX[task])
    header_parts.append(r"\textbf{Mean}")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # ── Helper to emit rows for one loss ──
    def _emit_loss_rows(loss_key: str, display_name: str, task_data: Dict,
                        is_last: bool) -> None:
        n_metrics = len(metrics)
        lines.append(rf"\multirow{{{n_metrics}}}{{*}}{{{display_name}}}")
        for m_idx, metric in enumerate(metrics):
            m_disp = METRIC_DISPLAY.get(metric, metric)
            row_parts = [f"  & {m_disp}"]
            for task in TASK_ORDER:
                v = task_data.get(task, {}).get(metric)
                is_best = (best_map.get(metric, {}).get(task) == loss_key)
                row_parts.append(fmt_val(v, metric, bold=is_best))
            # Mean column
            mean_v = compute_mean(task_data, metric)
            is_best = (best_map.get(metric, {}).get("Mean") == loss_key)
            row_parts.append(fmt_val(mean_v, metric, bold=is_best))
            lines.append(" & ".join(row_parts) + r" \\")
        if not is_last:
            lines.append(r"\midrule")

    # ── Classification baseline ──
    if include_clf is not None:
        _emit_loss_rows("__clf_baseline__", r"\textbf{Clf.\,Baseline}",
                        include_clf, is_last=False)

    # ── Each ranking loss ──
    active_losses = [l for l in loss_order if l in all_data]
    for loss_idx, loss in enumerate(active_losses):
        task_data = all_data[loss]
        display = LOSS_DISPLAY.get(loss, loss.title())
        is_last = (loss_idx == len(active_losses) - 1)
        _emit_loss_rows(loss, display, task_data, is_last)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")  # close \resizebox
    # ── Footnote ──
    if not footnote:
        footnote = (
            r"$^*$~Marrow changes follow Modic "
            r"Types 0-III, which lacks a clinically severity ordering. "
            r"They are treated as 4-class nominal classification and excluded "
            r"from ordinal-metric comparisons."
        )
    lines.append(r"\smallskip")
    lines.append(r"\begin{minipage}{\textwidth}")
    lines.append(r"\scriptsize")
    lines.append(footnote)
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_compact_table(
    all_data: Dict[str, Dict[str, Dict[str, float]]],
    loss_order: List[str],
    metrics: List[str],
    head_info: Dict[str, str],
    include_clf: Optional[Dict[str, Dict[str, float]]] = None,
    bold_best: bool = True,
    caption: str = "Mean ranking performance across ordinal ranking losses.",
    label: str = "tab:ranking_losses_compact",
) -> str:
    """Generate a compact table: one row per loss, columns = metrics (mean only).

    Useful for a summary overview when 13 per-task columns are too wide.
    """
    lines = []
    n_metrics = len(metrics)
    col_spec = "l" + "l" + "c" * n_metrics

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = [r"\textbf{Loss}", r"\textbf{Head}"]
    for m in metrics:
        header.append(rf"\textbf{{{METRIC_DISPLAY.get(m, m)}}}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Find best per metric (mean) for bolding
    best_per_metric = {}
    if bold_best:
        comparison = dict(all_data)
        if include_clf is not None:
            comparison["__clf__"] = include_clf
        for m in metrics:
            hib = HIGHER_IS_BETTER.get(m, True)
            best_val = None
            best_loss = None
            for loss_name, td in comparison.items():
                v = compute_mean(td, m)
                if v is None:
                    continue
                if best_val is None or (hib and v > best_val) or (not hib and v < best_val):
                    best_val = v
                    best_loss = loss_name
            best_per_metric[m] = best_loss

    # Clf baseline row
    if include_clf is not None:
        row = [r"\textbf{Clf. Baseline}", "--"]
        for m in metrics:
            v = compute_mean(include_clf, m)
            is_best = (best_per_metric.get(m) == "__clf__")
            row.append(fmt_val(v, m, bold=is_best))
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    # Each loss
    for loss in loss_order:
        if loss not in all_data:
            continue
        td = all_data[loss]
        display = LOSS_DISPLAY.get(loss, loss.title())
        head = head_info.get(loss, "?")

        row = [display, head.replace("_", " ").title()]
        for m in metrics:
            v = compute_mean(td, m)
            is_best = (best_per_metric.get(m) == loss)
            row.append(fmt_val(v, m, bold=is_best))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_csv_summary(
    all_data: Dict[str, Dict[str, Dict[str, float]]],
    loss_order: List[str],
    metrics: List[str],
    head_info: Dict[str, str],
) -> str:
    """Generate a CSV summary for easy ingestion by other tools."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["loss", "head"]
    for task in TASK_ORDER:
        for m in metrics:
            header.append(f"{TASK_SHORT[task]}_{m}")
    for m in metrics:
        header.append(f"Mean_{m}")
    writer.writerow(header)

    for loss in loss_order:
        if loss not in all_data:
            continue
        td = all_data[loss]
        head = head_info.get(loss, "?")
        row = [LOSS_DISPLAY.get(loss, loss), head]
        for task in TASK_ORDER:
            for m in metrics:
                v = td.get(task, {}).get(m)
                row.append(f"{v:.4f}" if v is not None else "")
        for m in metrics:
            v = compute_mean(td, m)
            row.append(f"{v:.4f}" if v is not None else "")
        writer.writerow(row)

    return output.getvalue()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables comparing ranking losses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="results/ranking",
        help="Path to results directory (default: results/ranking)",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18",
        help="Backbone name (default: resnet18)",
    )
    parser.add_argument(
        "--losses", nargs="+", type=str, default=None,
        help="Specific losses to include (default: all found)",
    )
    parser.add_argument(
        "--heads", nargs="+", type=str, default=None,
        help="Restrict to specific heads (default: all found, select best)",
    )
    parser.add_argument(
        "--fixed_head", type=str, default=None,
        help="Use a fixed head for all losses instead of best-per-loss "
             "(e.g., 'linear')",
    )
    parser.add_argument(
        "--metrics", nargs="+", type=str, default=DEFAULT_METRICS,
        help=f"Metrics to display (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--source", type=str, default="hybrid_metrics",
        choices=["hybrid_metrics", "evaluation_metrics"],
        help="Metrics source file (default: hybrid_metrics)",
    )
    parser.add_argument(
        "--selection_metric", type=str, default=BEST_HEAD_METRIC,
        help=f"Metric for selecting best head (default: {BEST_HEAD_METRIC})",
    )
    parser.add_argument(
        "--include_clf_baseline", action="store_true",
        help="Include standalone classification baseline as first row",
    )
    parser.add_argument(
        "--clf_baseline_path", type=str, default="",
        help="Per-task classification baseline CSV (columns: Task, Grade, "
             "Bal.Acc, QWK, ROC_AUC, MAE, Spearman; only Grade=='All' rows "
             "are used). Required with --include_clf_baseline.",
    )
    parser.add_argument(
        "--show_clf_side", action="store_true",
        help="Show classification-side metrics alongside ranking metrics "
             "(requires evaluation_metrics.json)",
    )
    parser.add_argument(
        "--average_heads", action="store_true",
        help="Average metrics across all available heads per loss "
             "(instead of selecting the best head)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Generate compact table (mean only, one row per loss)",
    )
    parser.add_argument(
        "--no_bold", action="store_true",
        help="Disable bolding of best values",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (default: stdout). "
             "If ends with .csv, generates CSV instead of LaTeX.",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output CSV summary instead of LaTeX",
    )
    parser.add_argument(
        "--caption", type=str,
        default=("Per-task metrics results across all loss objectives test set "
                 "for a ResNet 3D 18 encoder; "
                 r"Balance accuracy (BA) / QWK / $\rho$ / MAE$\downarrow$ per loss;"),
    )
    parser.add_argument(
        "--label", type=str, default="tab:per_task_all_metrics",
    )
    parser.add_argument(
        "--footnote", type=str, default="",
        help="Custom footnote text below the table (default: Modic note)",
    )

    args = parser.parse_args()
    if args.include_clf_baseline and not args.clf_baseline_path:
        parser.error("--include_clf_baseline requires --clf_baseline_path")

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    # Discover experiments
    experiments = discover_experiments(results_dir, args.backbone)
    if not experiments:
        print(f"ERROR: No experiments found in {results_dir / args.backbone}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(experiments)} losses: {sorted(experiments.keys())}",
          file=sys.stderr)
    for loss, heads in sorted(experiments.items()):
        print(f"  {loss}: {sorted(heads.keys())}", file=sys.stderr)

    # Filter losses if requested
    if args.losses:
        filtered = {}
        for loss in args.losses:
            loss_lower = loss.lower()
            if loss_lower in experiments:
                filtered[loss_lower] = experiments[loss_lower]
            else:
                print(f"WARNING: Loss '{loss}' not found in results",
                      file=sys.stderr)
        experiments = filtered

    # Filter heads if requested
    if args.heads:
        for loss in experiments:
            experiments[loss] = {
                h: p for h, p in experiments[loss].items()
                if h in args.heads
            }

    # Select head strategy and load data
    all_data = {}
    head_info = {}
    clf_side_data = {} if args.show_clf_side else None

    for loss, heads in sorted(experiments.items()):
        if args.fixed_head:
            if args.fixed_head in heads:
                selected_heads = {args.fixed_head: heads[args.fixed_head]}
            else:
                print(f"WARNING: Fixed head '{args.fixed_head}' not found "
                      f"for loss '{loss}', skipping", file=sys.stderr)
                continue
        else:
            selected_heads = heads

        if args.average_heads:
            head_label, avg_data = average_across_heads(
                selected_heads, source=args.source,
            )
            if head_label is None or avg_data is None:
                print(f"WARNING: No valid data for loss '{loss}'",
                      file=sys.stderr)
                continue
            all_data[loss] = avg_data
            head_info[loss] = head_label
            print(f"  {loss}: averaged over {head_label}",
                  file=sys.stderr)
        else:
            best_head, best_data = select_best_head(
                selected_heads,
                metric=args.selection_metric,
                source=args.source,
            )
            if best_head is None or best_data is None:
                print(f"WARNING: No valid data for loss '{loss}'",
                      file=sys.stderr)
                continue

            all_data[loss] = best_data
            head_info[loss] = best_head
            print(f"  {loss}: selected head='{best_head}' "
                  f"(mean {args.selection_metric}="
                  f"{compute_mean(best_data, args.selection_metric):.4f})",
                  file=sys.stderr)

        # Load classification-side if requested
        if args.show_clf_side and not args.average_heads:
            eval_dir = selected_heads.get(head_info.get(loss, ""), None)
            if eval_dir is not None:
                clf_data = load_evaluation_metrics(
                    eval_dir / "evaluation_metrics.json",
                    section="classification",
                )
                if clf_data:
                    clf_side_data[loss] = clf_data

    if not all_data:
        print("ERROR: No valid experiment data loaded", file=sys.stderr)
        sys.exit(1)

    # Determine loss order
    loss_order = [l for l in LOSS_ORDER if l in all_data]
    # Add any losses not in LOSS_ORDER at the end
    for l in sorted(all_data.keys()):
        if l not in loss_order:
            loss_order.append(l)

    # Load classification baseline
    clf_baseline = None
    if args.include_clf_baseline:
        clf_path = Path(args.clf_baseline_path)
        if not clf_path.is_absolute():
            clf_path = project_root / clf_path
        clf_baseline = load_clf_baseline(clf_path)
        if clf_baseline is None:
            print(f"WARNING: Could not load clf baseline from {clf_path}",
                  file=sys.stderr)
        else:
            print(f"Loaded classification baseline ({len(clf_baseline)} tasks)",
                  file=sys.stderr)

    # Generate output
    if args.csv or (args.output and args.output.endswith(".csv")):
        output = generate_csv_summary(all_data, loss_order, args.metrics,
                                       head_info)
    elif args.compact:
        output = generate_compact_table(
            all_data, loss_order, args.metrics, head_info,
            include_clf=clf_baseline,
            bold_best=not args.no_bold,
            caption=args.caption,
            label=args.label,
        )
    else:
        output = generate_latex_table(
            all_data, loss_order, args.metrics, head_info,
            include_clf=clf_baseline,
            clf_side_data=clf_side_data if args.show_clf_side else None,
            bold_best=not args.no_bold,
            caption=args.caption,
            label=args.label,
            footnote=args.footnote,
        )

    # Output
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output)
        print(f"\nOutput written to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
