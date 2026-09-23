#!/usr/bin/env python3
"""Lightweight results explorer.

Walks every ``hybrid_metrics.json`` under ``--root`` and prints a compact
summary (per-seed and per-config) so you can inspect multi-seed / multi-
config results without opening the multi-GB visualisation caches.

Three output modes (each ~10 KB):
  * ``--mode text``    one-screen table per (config, seed)        [default]
  * ``--mode csv``     one row per (config, seed, task) -> stdout (pipe to .csv)
  * ``--mode markdown`` GitHub-ready summary; copy into a notebook / PR

To create a downloadable bundle of *only* the small artefacts (JSONs +
per-task PNGs, no viz_cache), pass ``--bundle <out.tar.gz>``. The bundle
typically lands well under 50 MB even for dozens of seeds.

Usage examples
--------------
::

    # Quick text summary of every run under results/
    python scripts/explore_results.py --root results

    # Restrict to one experiment tree
    python scripts/explore_results.py --root results/ranking

    # Build a small archive of the JSON / PNG artefacts
    python scripts/explore_results.py --root results --bundle results_summary.tar.gz

    # Show per-task best/worst rows in markdown
    python scripts/explore_results.py --root results --mode markdown
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

METRIC_COLS: List[Tuple[str, str]] = [
    ("balanced_accuracy", "BalAcc"),
    ("qwk",               "QWK"),
    ("roc_auc",           "ROC-AUC"),
    ("mcc",               "MCC"),
    ("mae",               "MAE"),
    ("concordance_index", "C-idx"),
]


def discover_runs(root: Path) -> Dict[Tuple[str, str], Path]:
    """Return ``{(config, seed): hybrid_metrics_path}``.

    ``config`` is everything between *root* and the ``seed_<N>`` directory,
    so runs under e.g.
    ``results/ranking/resnet18/spinerank_mlp/seed_42/eval/resnet18/hybrid_metrics.json``
    bucket as ``config = spineranknet/resnet18/spinerank_mlp``.
    """
    out: Dict[Tuple[str, str], Path] = {}
    for p in root.rglob("hybrid_metrics.json"):
        parts = p.parts
        # Find the LAST `seed_<N>` segment whose suffix is numeric (so
        # parent directories like ``seed_sweep/`` are skipped).
        seed_idx = None
        for k in range(len(parts) - 1, -1, -1):
            s = parts[k]
            if s.startswith("seed_") and s.split("_", 1)[1].lstrip("-").isdigit():
                seed_idx = k; break
        if seed_idx is None:
            continue
        seed_str = parts[seed_idx].split("_", 1)[1]
        # config = everything between root and seed_<N>
        rel_to_root = Path(*parts[len(root.parts):seed_idx]) \
            if seed_idx > len(root.parts) else Path(".")
        out[(str(rel_to_root), seed_str)] = p
    return dict(sorted(out.items()))


def load_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARN  could not read {path}: {e}", file=sys.stderr)
        return {}


def macro_mean(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col, _ in METRIC_COLS:
        vals = [v[col] for v in metrics.values() if col in v and v[col] is not None]
        if vals:
            out[col] = float(np.mean(vals))
    return out


def summarise_text(
    runs: Dict[Tuple[str, str], Path],
    selected_metrics: List[Tuple[str, str]],
) -> None:
    # group by config
    by_cfg: Dict[str, List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
    for (cfg, seed), p in runs.items():
        macro = macro_mean(load_metrics(p))
        if macro:
            by_cfg[cfg].append((seed, macro))

    if not by_cfg:
        print("No hybrid_metrics.json found.")
        return

    for cfg, seed_rows in by_cfg.items():
        print()
        print(f"=== {cfg or '.'}  ({len(seed_rows)} seed{'' if len(seed_rows)==1 else 's'}) ===")
        header = f"{'seed':<8}" + " ".join(f"{name:>9}" for _, name in selected_metrics)
        print(header)
        print("-" * len(header))
        for seed, macro in sorted(seed_rows, key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            row = f"{seed:<8}"
            for col, _ in selected_metrics:
                v = macro.get(col)
                row += f" {v:>9.4f}" if v is not None else f" {'-':>9}"
            print(row)
        # macro across seeds
        if len(seed_rows) > 1:
            print("-" * len(header))
            row_mean = f"{'mean':<8}"
            row_std  = f"{'std':<8}"
            for col, _ in selected_metrics:
                vals = [macro.get(col) for _, macro in seed_rows if macro.get(col) is not None]
                if vals:
                    row_mean += f" {np.mean(vals):>9.4f}"
                    row_std  += f" {np.std(vals, ddof=1) if len(vals) > 1 else 0.0:>9.4f}"
                else:
                    row_mean += f" {'-':>9}"
                    row_std  += f" {'-':>9}"
            print(row_mean)
            print(row_std)


def write_csv(
    runs: Dict[Tuple[str, str], Path],
    selected_metrics: List[Tuple[str, str]],
) -> None:
    """One row per (config, seed, task)."""
    fields = ["config", "seed", "task", "n_samples"] + [c for c, _ in selected_metrics]
    w = csv.DictWriter(sys.stdout, fieldnames=fields)
    w.writeheader()
    for (cfg, seed), p in runs.items():
        for task, m in load_metrics(p).items():
            row = {
                "config": cfg, "seed": seed, "task": task,
                "n_samples": m.get("n_samples", ""),
            }
            for col, _ in selected_metrics:
                v = m.get(col)
                row[col] = f"{v:.6f}" if isinstance(v, (int, float)) else ""
            w.writerow(row)


def write_markdown(
    runs: Dict[Tuple[str, str], Path],
    selected_metrics: List[Tuple[str, str]],
) -> None:
    by_cfg: Dict[str, List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
    for (cfg, seed), p in runs.items():
        macro = macro_mean(load_metrics(p))
        if macro:
            by_cfg[cfg].append((seed, macro))

    print(f"# Results summary  ({sum(len(v) for v in by_cfg.values())} runs across {len(by_cfg)} config(s))\n")
    for cfg, seed_rows in by_cfg.items():
        print(f"## `{cfg or '.'}` — {len(seed_rows)} seed(s)\n")
        head = "| seed | " + " | ".join(name for _, name in selected_metrics) + " |"
        sep  = "|" + ("---|" * (len(selected_metrics) + 1))
        print(head); print(sep)
        for seed, macro in sorted(seed_rows, key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            cells = [f"`{seed}`"] + [
                f"{macro.get(col, float('nan')):.4f}" if macro.get(col) is not None else "-"
                for col, _ in selected_metrics
            ]
            print("| " + " | ".join(cells) + " |")
        if len(seed_rows) > 1:
            mean_cells = ["**mean ± std**"]
            for col, _ in selected_metrics:
                vals = [m.get(col) for _, m in seed_rows if m.get(col) is not None]
                if vals:
                    mu = np.mean(vals); sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    mean_cells.append(f"**{mu:.3f} ± {sd:.3f}**")
                else:
                    mean_cells.append("-")
            print("| " + " | ".join(mean_cells) + " |")
        print()


# ----------------------------------------------------------------------
# Tarball-bundle option (small files only)
# ----------------------------------------------------------------------

BUNDLE_PATTERNS = (
    "hybrid_metrics.json",
    "evaluation_metrics.json",
    "*_metrics.json",
    "*_cm.png",                 # confusion matrices
    "*_score_boxplot.png",      # score-vs-grade boxplots
    "*_density_hist.png",       # score distributions
    "*_summary.csv",
    "*_summary.tex",
    "*_per_pathology.tex",
    # Run configs (train_config.yaml) are deliberately not bundled: they
    # contain local data/output paths.
)


def bundle(root: Path, out_path: Path) -> None:
    """Tar+gzip the small artefacts only; skip viz_cache, weights, big PNGs."""
    SKIP_SUBSTRINGS = ("viz_cache", "ranked_image_grid", "patient_level_grid")
    EXTS_SKIP = (".pt", ".pkl", ".pkl.gz")

    matched = []
    for pat in BUNDLE_PATTERNS:
        matched.extend(root.rglob(pat))
    # de-dupe while preserving order
    seen = set(); uniq: List[Path] = []
    for p in matched:
        if p in seen:
            continue
        if any(s in str(p) for s in SKIP_SUBSTRINGS):
            continue
        if any(p.name.endswith(e) for e in EXTS_SKIP):
            continue
        seen.add(p); uniq.append(p)

    print(f"bundling {len(uniq)} files -> {out_path}")
    total_bytes = 0
    with tarfile.open(out_path, "w:gz") as tar:
        for p in uniq:
            arcname = str(p.relative_to(root))
            tar.add(p, arcname=arcname)
            total_bytes += p.stat().st_size
    final_mb = out_path.stat().st_size / (1024 * 1024)
    raw_mb = total_bytes / (1024 * 1024)
    print(f"wrote {out_path}  ({final_mb:.1f} MB compressed, {raw_mb:.1f} MB raw)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results"),
                    help="Root directory to scan (default: results/).")
    ap.add_argument("--mode", choices=("text", "csv", "markdown"),
                    default="text",
                    help="Output format for the summary.")
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="Subset of metrics to show. Default: all available.")
    ap.add_argument("--bundle", type=Path, default=None,
                    help="If set, additionally write a small tar.gz of only "
                         "the JSON/PNG/CSV/YAML files (no checkpoints, no "
                         "viz_cache).")
    args = ap.parse_args()

    if not args.root.exists():
        sys.exit(f"--root {args.root} does not exist")

    runs = discover_runs(args.root)
    if not runs:
        sys.exit(f"No hybrid_metrics.json found under {args.root}")

    selected = METRIC_COLS
    if args.metrics:
        keep = set(args.metrics)
        selected = [(c, n) for c, n in METRIC_COLS if c in keep or n.lower() in keep]
        if not selected:
            selected = METRIC_COLS

    if args.mode == "text":
        summarise_text(runs, selected)
    elif args.mode == "csv":
        write_csv(runs, selected)
    elif args.mode == "markdown":
        write_markdown(runs, selected)

    if args.bundle:
        bundle(args.root, args.bundle)


if __name__ == "__main__":
    main()
