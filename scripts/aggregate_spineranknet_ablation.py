#!/usr/bin/env python3
"""Aggregate SpineRankNet ablation results into concise paper tables.

Produces two ablation tables suitable for a methods / results section:

  Table 1 — Ranking Head Ablation
      Rows   : MLP, Linear, Transformer, KAN   (Least-Sq. if evaluated)
      Columns: BalAcc, QWK, Spearman ρ, Kendall τ, MAE
      Source : eval_grid/ (falls back to eval/) per head

  Table 2 — Threshold Strategy Ablation
      Rows   : Grid Search, Isotonic, GMM, Youden
      Columns: same metrics, for MLP only by default (override with --thresh_heads)
      Source : eval_grid|isotonic|gmm|youden/ per head

  Note: UpperModic and LowerModic are excluded from all computations
  (11 tasks total).

Optional Table 3 (--per_task):
      Per-task BalAcc + QWK × head, grouped by pathology category.

Outputs
-------
  <out_dir>/head_ablation.csv / .tex
  <out_dir>/threshold_ablation.csv / .tex
  <out_dir>/threshold_per_head.csv
  <out_dir>/per_task_breakdown.csv   (with --per_task)

Usage
-----
  # Default paths (run from project root)
  python scripts/aggregate_spineranknet_ablation.py

  # Custom paths
  python scripts/aggregate_spineranknet_ablation.py \\
      --results_dir results/ranking \\
      --backbone resnet18 \\
      --eval_subdir eval_grid \\
      --out_dir paper/tables/ablation

  # Include per-task breakdown, skip LaTeX
  python scripts/aggregate_spineranknet_ablation.py --per_task --no_latex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Task definitions ──────────────────────────────────────────────────────────

TASK_GROUPS: Dict[str, List[str]] = {
    "Disc Degen.":   ["Pfirrmann", "Narrowing"],
    "Stenosis":      ["CentralCanalStenosis", "ForaminalStenosisLeft", "ForaminalStenosisRight"],
    "Endplate":      ["UpperEndplateDefect", "LowerEndplateDefect"],
    "Disc Morph.":   ["Herniation", "AnteriorBulging", "PosteriorBulging"],
    "Alignment":     ["Spondylolisthesis"],
}
# Modic changes (UpperModic, LowerModic) are excluded from all computations
# because their highly imbalanced distribution and limited imaging contrast
# make them unsuitable for ranking-based evaluation.
ALL_TASKS: List[str] = [t for tasks in TASK_GROUPS.values() for t in tasks]  # 11

TASK_SHORT: Dict[str, str] = {
    "Pfirrmann":            "Pfi",
    "Narrowing":            "Nar",
    "CentralCanalStenosis": "CCS",
    "ForaminalStenosisLeft":  "FSL",
    "ForaminalStenosisRight": "FSR",
    "UpperEndplateDefect":  "UED",
    "LowerEndplateDefect":  "LED",
    "Herniation":           "Her",
    "AnteriorBulging":      "ABu",
    "PosteriorBulging":     "PBu",
    "Spondylolisthesis":    "Spo",
}

# ── Display / ordering constants ──────────────────────────────────────────────

HEAD_ORDER: List[str] = ["mlp", "linear", "transformer", "kan", "least_squares"]
HEAD_DISPLAY: Dict[str, str] = {
    "mlp":           "MLP",
    "linear":        "Linear",
    "transformer":   "Transformer",
    "kan":           "KAN",
    "least_squares": "Least-Sq.",
}

METHOD_DISPLAY: Dict[str, str] = {
    "eval_grid":     "Grid Search",
    "eval_isotonic": "Isotonic",
    "eval_gmm":      "GMM",
    "eval_youden":   "Youden",
}
METHOD_ORDER: List[str] = ["eval_grid", "eval_isotonic", "eval_gmm", "eval_youden"]

# Primary metrics shown in ablation tables
METRICS: List[str] = [
    "balanced_accuracy",
    "qwk",
    "spearman_rho",
    "kendall_tau",
    "mae",
]
METRIC_HEADER: Dict[str, str] = {
    "balanced_accuracy": "BalAcc",
    "qwk":               "QWK",
    "spearman_rho":      "Spearman",
    "kendall_tau":       "Kendall τ",
    "mae":               "MAE",
}
# Metrics where higher value = better (used for bold-best highlighting)
HIGHER_BETTER: set = {"balanced_accuracy", "qwk", "spearman_rho", "kendall_tau",
                      "c_index", "adj_acc"}

# ── Prediction-level metrics (C-index, SER) ───────────────────────────────────
# These require per-sample CSVs, not just hybrid_metrics.json.
# C-index  = P(score_i > score_j | grade_i > grade_j)  — threshold-independent
# SER      = P(|pred − true| ≥ 2)                       — threshold-dependent
# Adj. Acc = P(|pred − true| ≤ 1) = 1 − SER

# ── Data loading ──────────────────────────────────────────────────────────────


def load_json(path: Path) -> Optional[Dict]:
    """Return parsed JSON or None if the file does not exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def avg_metrics(metrics_dict: Dict, tasks: List[str] = ALL_TASKS) -> Dict[str, float]:
    """Average every metric in METRICS across available tasks."""
    out: Dict[str, float] = {}
    for m in METRICS:
        vals = [
            metrics_dict[t][m]
            for t in tasks
            if t in metrics_dict and m in metrics_dict[t]
        ]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    out["n_tasks"] = sum(1 for t in tasks if t in metrics_dict)
    return out


def _concordance_index(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score_i > score_j | grade_i > grade_j) — pairwise concordance index.

    Ranges from 0.5 (random ordering) to 1.0 (perfect monotone ranking).
    Ties in scores are counted as 0.5 (half-concordant).
    """
    conc = tot = 0.0
    for g in np.unique(labels):
        lo = scores[labels == g]
        for g2 in np.unique(labels[labels > g]):
            hi = scores[labels == g2]
            conc += (hi[:, None] > lo[None, :]).sum()
            conc += 0.5 * (hi[:, None] == lo[None, :]).sum()
            tot  += hi.size * lo.size
    return conc / tot if tot > 0 else 0.5


def _load_pred_csv(pred_dir: Path, task: str) -> Optional[pd.DataFrame]:
    """Load rank_<task>_predictions.csv; return None if missing."""
    p = pred_dir / f"rank_{task}_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df[df["true_label"] >= 0].copy()


def compute_pred_metrics(
    eval_subdir_path: Path,
    backbone: str,
    tasks: List[str] = ALL_TASKS,
) -> Dict[str, float]:
    """Compute C-index and SER from per-sample ranking prediction CSVs.

    Looks for CSVs at: <eval_subdir_path>/<backbone>/ranking/predictions/

    Returns dict with keys: c_index, ser, adj_acc  (means over available tasks)
    """
    pred_dir = eval_subdir_path / backbone / "ranking" / "predictions"
    if not pred_dir.exists():
        return {}

    ci_vals, ser_vals = [], []
    for task in tasks:
        df = _load_pred_csv(pred_dir, task)
        if df is None or "raw_score" not in df.columns:
            continue
        scores = df["raw_score"].values.astype(float)
        true_l = df["true_label"].values.astype(int)
        pred_l = df["predicted_label"].values.astype(int)
        err    = np.abs(pred_l - true_l)
        ci_vals.append(_concordance_index(scores, true_l))
        ser_vals.append(float((err >= 2).mean()))

    if not ci_vals:
        return {}
    return {
        "c_index": float(np.mean(ci_vals)),
        "ser":     float(np.mean(ser_vals)),
        "adj_acc": float(1.0 - np.mean(ser_vals)),
    }


def load_head_metrics(
    results_dir: Path,
    backbone: str,
    preferred_subdir: str = "eval_grid",
    fallback_subdirs: Tuple[str, ...] = ("eval",),
) -> Dict[str, Dict]:
    """Load per-head summary metrics from the best available eval directory."""
    out: Dict[str, Dict] = {}
    for head in HEAD_ORDER:
        exp_dir = results_dir / backbone / f"spinerank_{head}"
        for subdir in (preferred_subdir, *fallback_subdirs):
            json_path = exp_dir / subdir / backbone / "hybrid_metrics.json"
            data = load_json(json_path)
            if data is not None:
                summary = avg_metrics(data)
                # Augment with prediction-level metrics (C-index, SER)
                pred_stats = compute_pred_metrics(exp_dir / subdir, backbone)
                summary.update(pred_stats)
                out[head] = {
                    "summary":  summary,
                    "per_task": {
                        t: {m: data[t].get(m) for m in METRICS}
                        for t in ALL_TASKS
                        if t in data
                    },
                    "source": subdir,
                }
                break
        else:
            pass
    return out


def load_threshold_metrics(
    results_dir: Path,
    backbone: str,
    heads: List[str],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load per-(head, method) summary metrics for threshold ablation."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for head in heads:
        exp_dir = results_dir / backbone / f"spinerank_{head}"
        for subdir, label in METHOD_DISPLAY.items():
            json_path = exp_dir / subdir / backbone / "hybrid_metrics.json"
            data = load_json(json_path)
            if data is not None:
                out[(head, label)] = avg_metrics(data)
    return out


# ── Table builders ────────────────────────────────────────────────────────────


def build_head_table(head_data: Dict) -> pd.DataFrame:
    """One row per head; columns: Head, Source, metric columns, C-index, SER, Adj.Acc, N Tasks."""
    rows = []
    for head in HEAD_ORDER:
        if head not in head_data:
            continue
        d = head_data[head]
        row: Dict = {
            "Head":   HEAD_DISPLAY[head],
            "Source": d["source"],
        }
        for m in METRICS:
            row[METRIC_HEADER[m]] = d["summary"][m]
        # Prediction-level metrics (may be absent if no per-sample CSVs found)
        row["C-index"]  = d["summary"].get("c_index",  float("nan"))
        row["SER (%)"]  = d["summary"].get("ser",      float("nan")) * 100 \
                          if not np.isnan(d["summary"].get("ser", float("nan"))) \
                          else float("nan")
        row["Adj.Acc"] = d["summary"].get("adj_acc",  float("nan"))
        row["# Tasks"] = int(d["summary"]["n_tasks"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_aggregate_table(
    thresh_data: Dict[Tuple[str, str], Dict[str, float]],
) -> pd.DataFrame:
    """Threshold ablation aggregated across heads — one row per method."""
    # Group by method label
    method_groups: Dict[str, List[Dict[str, float]]] = {}
    for (_, method), vals in thresh_data.items():
        method_groups.setdefault(method, []).append(vals)

    rows = []
    for method in METHOD_DISPLAY.values():
        if method not in method_groups:
            continue
        group = method_groups[method]
        row: Dict = {"Threshold": method}
        for m in METRICS:
            vs = [d[m] for d in group if not np.isnan(d.get(m, float("nan")))]
            row[METRIC_HEADER[m]] = float(np.mean(vs)) if vs else float("nan")
        row["N Heads"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_per_head_table(
    thresh_data: Dict[Tuple[str, str], Dict[str, float]],
) -> pd.DataFrame:
    """Full Head × Threshold breakdown — one row per combination."""
    method_order = list(METHOD_DISPLAY.values())

    def sort_key(pair):
        head, method = pair
        hi = HEAD_ORDER.index(head) if head in HEAD_ORDER else 99
        mi = method_order.index(method) if method in method_order else 99
        return (hi, mi)

    rows = []
    for (head, method) in sorted(thresh_data.keys(), key=sort_key):
        vals = thresh_data[(head, method)]
        row: Dict = {
            "Head":      HEAD_DISPLAY.get(head, head),
            "Threshold": method,
        }
        for m in METRICS:
            row[METRIC_HEADER[m]] = vals[m]
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_task_table(head_data: Dict) -> pd.DataFrame:
    """Per-task BalAcc and QWK for each evaluated head, grouped by pathology."""
    available_heads = [h for h in HEAD_ORDER if h in head_data]
    rows = []
    for group_name, tasks in TASK_GROUPS.items():
        for task in tasks:
            row: Dict = {
                "Group": group_name,
                "Task":  TASK_SHORT.get(task, task),
            }
            for head in available_heads:
                pt = head_data[head]["per_task"].get(task, {})
                label = HEAD_DISPLAY[head]
                row[f"{label} BalAcc"] = pt.get("balanced_accuracy")
                row[f"{label} QWK"]    = pt.get("qwk")
            rows.append(row)

    df = pd.DataFrame(rows)

    # Append group-average rows
    group_rows = []
    for group_name, tasks in TASK_GROUPS.items():
        task_shorts = [TASK_SHORT.get(t, t) for t in tasks]
        sub = df[df["Task"].isin(task_shorts)]
        row = {"Group": group_name, "Task": "(avg)"}
        for head in available_heads:
            label = HEAD_DISPLAY[head]
            for metric in ["BalAcc", "QWK"]:
                col = f"{label} {metric}"
                if col in sub.columns:
                    row[col] = sub[col].mean()
        group_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    return pd.concat([df, group_df], ignore_index=True)


# ── Formatting helpers ────────────────────────────────────────────────────────


def fmt_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Pretty-print a DataFrame with tabulate if available, else fallback."""
    try:
        from tabulate import tabulate  # type: ignore
        return tabulate(
            df, headers="keys", tablefmt="rounded_grid",
            floatfmt=float_fmt, showindex=False, missingval="—",
        )
    except ImportError:
        return df.to_string(
            index=False,
            float_format=lambda x: f"{x:{float_fmt}}" if not np.isnan(x) else "—",
        )


def _bold_best(series: pd.Series, higher: bool) -> List[str]:
    """Format numeric series as strings, bolding the best value in LaTeX."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return ["--"] * len(series)
    best = valid.max() if higher else valid.min()
    out = []
    for v in numeric:
        if np.isnan(v):
            out.append("--")
        elif abs(v - best) < 1e-9:
            out.append(f"\\textbf{{{v:.4f}}}")
        else:
            out.append(f"{v:.4f}")
    return out


def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    metric_cols: List[str],
) -> str:
    """Render a DataFrame as a paper-ready booktabs LaTeX table."""
    sub = df.copy()
    for col in metric_cols:
        if col not in sub.columns:
            continue
        higher = any(k in col for k in ["BalAcc", "QWK", "Spearman", "Kendall",
                                         "C-index", "Adj.Acc"])
        sub[col] = _bold_best(sub[col], higher)

    n_cols = len(sub.columns)
    col_spec = "l" + "r" * (n_cols - 1)
    header = " & ".join(
        f"\\textbf{{{c.replace('↑','').replace('↓','').strip()}}}"
        for c in sub.columns
    )

    lines = [
        "% Auto-generated by scripts/aggregate_spineranknet_ablation.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    for _, row in sub.iterrows():
        lines.append("    " + " & ".join(str(v) if v is not None else "--" for v in row.values) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Quick-summary helper ──────────────────────────────────────────────────────


def print_quick_summary(head_table: pd.DataFrame, thresh_agg: Optional[pd.DataFrame]) -> None:
    w = 60
    print(f"\n{'━' * w}")
    print("  QUICK SUMMARY — Best head per metric")
    print(f"{'━' * w}")
    # Standard metrics from hybrid_metrics.json
    check_cols = [(METRIC_HEADER[m], m in HIGHER_BETTER) for m in METRICS]
    # Prediction-level metrics (may not be present if CSVs not available)
    check_cols += [("C-index", True), ("SER (%)", False), ("Adj.Acc", True)]
    for col, higher in check_cols:
        if col not in head_table.columns:
            continue
        sub = head_table[["Head", col]].dropna(subset=[col])
        if sub.empty:
            continue
        best_idx = sub[col].idxmax() if higher else sub[col].idxmin()
        direction = "↑" if higher else "↓"
        print(f"  {col:<12} {direction}  {sub.loc[best_idx, 'Head']:<14} ({sub.loc[best_idx, col]:.4f})")

    if thresh_agg is not None and not thresh_agg.empty:
        print(f"\n{'━' * w}")
        print("  QUICK SUMMARY — Best threshold per metric")
        print(f"{'━' * w}")
        for m in METRICS:
            col = METRIC_HEADER[m]
            if col not in thresh_agg.columns:
                continue
            sub = thresh_agg[["Threshold", col]].dropna(subset=[col])
            if sub.empty:
                continue
            higher = m in HIGHER_BETTER
            best_idx = sub[col].idxmax() if higher else sub[col].idxmin()
            direction = "↑" if higher else "↓"
            print(f"  {col:<12} {direction}  {sub.loc[best_idx, 'Threshold']:<14} ({sub.loc[best_idx, col]:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results_dir", type=Path, default=Path("results/ranking"),
        help="Root results directory, should contain <backbone>/ subdirs "
             "(default: results/ranking)",
    )
    ap.add_argument(
        "--backbone", default="resnet18",
        help="Backbone sub-directory name (default: resnet18)",
    )
    ap.add_argument(
        "--eval_subdir", default="eval_grid",
        help="Preferred eval subdirectory for head ablation table; "
             "falls back to eval/ if not found (default: eval_grid)",
    )
    ap.add_argument(
        "--thresh_heads", nargs="+",
        default=["mlp"],
        metavar="HEAD",
        help="Heads to include in threshold ablation table "
             "(default: mlp)",
    )
    ap.add_argument(
        "--out_dir", type=Path,
        default=Path("results/ranking/ablation_tables"),
        help="Directory for CSV and LaTeX outputs",
    )
    ap.add_argument(
        "--per_task", action="store_true",
        help="Also print and save per-task breakdown (Table 3)",
    )
    ap.add_argument(
        "--no_latex", action="store_true",
        help="Skip LaTeX .tex file output",
    )
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    backbone: str = args.backbone
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    W = 65
    print(f"\n{'═' * W}")
    print("  SpineRankNet — Ablation Results Aggregation")
    print(f"{'═' * W}")
    print(f"  Results dir : {results_dir}")
    print(f"  Backbone    : {backbone}")
    print(f"  Head eval   : {args.eval_subdir}  (fallback: eval/)")
    print(f"  Thresh heads: {', '.join(args.thresh_heads)}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'═' * W}\n")

    # ── TABLE 1: Ranking Head Ablation ────────────────────────────────────
    print("Loading head metrics …")
    head_data = load_head_metrics(
        results_dir=results_dir,
        backbone=backbone,
        preferred_subdir=args.eval_subdir,
    )

    if not head_data:
        print(
            f"ERROR: No metrics found under {results_dir / backbone}.\n"
            "       Check --results_dir and --backbone arguments."
        )
        sys.exit(1)

    for head, d in head_data.items():
        flag = "✓" if d["source"] == args.eval_subdir else f"⚠ fallback ({d['source']})"
        print(f"  {HEAD_DISPLAY[head]:<14}  {flag}  ({d['summary']['n_tasks']} tasks)")

    head_table = build_head_table(head_data)

    print(f"\n{'━' * W}")
    print(f"  TABLE 1 — Ranking Head Ablation  (mean over {len(ALL_TASKS)} tasks)")
    print(f"{'━' * W}")
    print(fmt_table(head_table))

    csv1 = out_dir / "head_ablation.csv"
    head_table.to_csv(csv1, index=False, float_format="%.6f")
    print(f"\n  Saved → {csv1}")

    if not args.no_latex:
        tex1 = out_dir / "head_ablation.tex"
        tex1.write_text(
            to_latex_table(
                df=head_table,
                caption=(
                    "Ablation study on the ranking head architecture. "
                    "All metrics are averaged over 11 spinal grading tasks "
                    "(Modic changes excluded). "
                    "Bold denotes the best value per column."
                ),
                label="tab:spinerank_head_ablation",
                metric_cols=list(METRIC_HEADER.values()) + ["C-index", "Adj.Acc"],
            )
        )
        print(f"  Saved → {tex1}")

    # ── TABLE 2: Threshold Strategy Ablation ─────────────────────────────
    print(f"\nLoading threshold metrics for: {args.thresh_heads} …")
    thresh_data = load_threshold_metrics(
        results_dir=results_dir,
        backbone=backbone,
        heads=args.thresh_heads,
    )

    thresh_agg: Optional[pd.DataFrame] = None
    if thresh_data:
        found_combos = list(thresh_data.keys())
        print(f"  Found {len(found_combos)} (head, method) combinations:")
        for head, method in found_combos:
            print(f"    {HEAD_DISPLAY.get(head, head):<14} × {method}")

        thresh_agg = build_threshold_aggregate_table(thresh_data)
        thresh_per_head = build_threshold_per_head_table(thresh_data)

        heads_label = " / ".join(HEAD_DISPLAY.get(h, h) for h in args.thresh_heads)
        print(f"\n{'━' * W}")
        print("  TABLE 2 — Threshold Strategy Ablation")
        print(f"  Averaged over heads: {heads_label}")
        print(f"{'━' * W}")
        print(fmt_table(thresh_agg))

        print(f"\n{'━' * W}")
        print("  TABLE 2b — Full Head × Threshold Breakdown")
        print(f"{'━' * W}")
        print(fmt_table(thresh_per_head))

        csv2 = out_dir / "threshold_ablation.csv"
        csv2b = out_dir / "threshold_per_head.csv"
        thresh_agg.to_csv(csv2, index=False, float_format="%.6f")
        thresh_per_head.to_csv(csv2b, index=False, float_format="%.6f")
        print(f"\n  Saved → {csv2}")
        print(f"  Saved → {csv2b}")

        if not args.no_latex:
            tex2 = out_dir / "threshold_ablation.tex"
            tex2.write_text(
                to_latex_table(
                    df=thresh_agg,
                    caption=(
                        f"Ablation study on threshold calibration strategy "
                        f"(heads: {heads_label}). "
                        "Metrics are averaged over all 13 tasks and all included heads. "
                        "Bold denotes the best value per column."
                    ),
                    label="tab:spinerank_threshold_ablation",
                    metric_cols=list(METRIC_HEADER.values()),
                )
            )
            print(f"  Saved → {tex2}")
    else:
        print(
            "  [WARN] No threshold method data found for any of the requested heads.\n"
            "         Run eval_grid / eval_isotonic / eval_gmm / eval_youden first."
        )

    # ── TABLE 3 (optional): Per-Task Breakdown ────────────────────────────
    if args.per_task:
        per_task_df = build_per_task_table(head_data)
        print(f"\n{'━' * W}")
        print("  TABLE 3 — Per-Task Breakdown  (BalAcc  |  QWK  per head)")
        print(f"{'━' * W}")
        print(fmt_table(per_task_df, float_fmt=".3f"))

        csv3 = out_dir / "per_task_breakdown.csv"
        per_task_df.to_csv(csv3, index=False, float_format="%.4f")
        print(f"\n  Saved → {csv3}")

    # ── Quick Summary ─────────────────────────────────────────────────────
    print_quick_summary(head_table, thresh_agg)

    # ── Final ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  Done.  All tables saved to:")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
