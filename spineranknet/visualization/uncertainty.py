#!/usr/bin/env python3
"""
SpineRankNet — TTA Uncertainty Quantification
=============================================

Computes and visualises uncertainty metrics derived from Test-Time
Augmentation (TTA) per-view softmax predictions.

Metrics
-------
- **Predictive entropy**  H[p̄] = −Σ p̄_c log p̄_c
- **Mean softmax std**    average σ of softmax probs across views per class
- **Disagreement rate**   fraction of views whose argmax ≠ majority vote
- **Mutual information**  H[p̄] − (1/V) Σ_v H[p_v]  (epistemic uncertainty)

Usage::

    from spineranknet.visualization.uncertainty import (
        save_tta_predictions_csv,
        compute_and_plot_uncertainty,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# UNCERTAINTY METRICS
# ════════════════════════════════════════════════════════════════════════════

def _entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """Shannon entropy along *axis*.  ``probs`` must already be normalised."""
    return -np.sum(probs * np.log(probs + eps), axis=axis)


def compute_sample_uncertainty(per_view_probs: np.ndarray) -> dict:
    """Compute uncertainty metrics for a single sample.

    Parameters
    ----------
    per_view_probs : ndarray (V, K)
        Softmax probabilities for each of the V TTA views.

    Returns
    -------
    dict with keys: predictive_entropy, mean_softmax_std, disagreement_rate,
    mutual_information, mean_pred, predicted_label.
    """
    mean_prob = per_view_probs.mean(axis=0)                       # (K,)
    pred_entropy = float(_entropy(mean_prob))

    # Per-view entropies → expected entropy under each augmentation
    view_entropies = _entropy(per_view_probs, axis=1)             # (V,)
    mutual_info = pred_entropy - float(view_entropies.mean())

    # Mean standard deviation of softmax across views
    mean_std = float(per_view_probs.std(axis=0).mean())

    # Disagreement: fraction of views whose argmax ≠ majority vote
    view_preds = per_view_probs.argmax(axis=1)                    # (V,)
    counts = np.bincount(view_preds, minlength=per_view_probs.shape[1])
    majority_vote = int(counts.argmax())
    disagreement = float((view_preds != majority_vote).mean())

    return {
        "predictive_entropy": pred_entropy,
        "mutual_information": mutual_info,
        "mean_softmax_std": mean_std,
        "disagreement_rate": disagreement,
        "mean_prob": mean_prob,
        "predicted_label": int(mean_prob.argmax()),
    }


# ════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ════════════════════════════════════════════════════════════════════════════

def save_tta_predictions_csv(
    per_view_probs: List[np.ndarray],
    sample_ids: List[str],
    true_labels: List[int],
    class_names: List[str],
    task: str,
    out_dir: Union[str, Path],
) -> Path:
    """Save per-view TTA predictions to CSV (one row per view per sample).

    Parameters
    ----------
    per_view_probs : list of ndarray (V, K)
        One array per test sample.
    sample_ids : list of str
    true_labels : list of int
    class_names : list of str
    task : str
    out_dir : path-like

    Returns
    -------
    Path to the saved CSV file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid, y_true, pv in zip(sample_ids, true_labels, per_view_probs):
        n_views = pv.shape[0]
        for v in range(n_views):
            row = {
                "sample_id": sid,
                "true_label": y_true,
                "view_idx": v,
                "predicted_label": int(pv[v].argmax()),
            }
            for c, cn in enumerate(class_names):
                row[f"prob_{cn}"] = float(pv[v, c])
            row["view_entropy"] = float(_entropy(pv[v]))
            rows.append(row)

    csv_path = out_dir / f"{task}_tta_predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"  TTA predictions CSV ({len(rows)} rows) → {csv_path}")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def compute_and_plot_uncertainty(
    per_view_probs: List[np.ndarray],
    true_labels: List[int],
    class_names: List[str],
    task: str,
    out_dir: Union[str, Path],
) -> Optional[Path]:
    """Compute uncertainty metrics and generate ETH-styled plots.

    Parameters
    ----------
    per_view_probs : list of ndarray (V, K)
    true_labels : list of int
    class_names : list of str
    task : str
    out_dir : path-like

    Returns
    -------
    Path to the uncertainty output directory, or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping uncertainty plots")
        return None

    from spineranknet.visualization.evaluation import ETH_COLORS, ETH_CATEGORICAL

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── compute per-sample metrics ────────────────────────────────────
    records = []
    for sid_idx, (pv, y_true) in enumerate(zip(per_view_probs, true_labels)):
        m = compute_sample_uncertainty(pv)
        m["true_label"] = y_true
        m["correct"] = int(m["predicted_label"] == y_true)
        records.append(m)

    df = pd.DataFrame(records)
    if df.empty:
        return None

    # ── save summary CSV ──────────────────────────────────────────────
    summary = {
        "task": task,
        "n_samples": len(df),
        "mean_entropy": float(df["predictive_entropy"].mean()),
        "std_entropy": float(df["predictive_entropy"].std()),
        "mean_mutual_info": float(df["mutual_information"].mean()),
        "mean_disagreement": float(df["disagreement_rate"].mean()),
        "mean_softmax_std": float(df["mean_softmax_std"].mean()),
        "accuracy": float(df["correct"].mean()),
    }
    summary_path = out_dir / "uncertainty_summary.csv"
    # Append if file exists (multi-task)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, pd.DataFrame([summary])], ignore_index=True)
        combined.to_csv(summary_path, index=False)
    else:
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # ── Plot 1: Entropy distribution ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    correct_mask = df["correct"] == 1
    if correct_mask.any():
        ax.hist(
            df.loc[correct_mask, "predictive_entropy"],
            bins=30, alpha=0.7, label="Correct",
            color=ETH_COLORS["green"], edgecolor="white", linewidth=0.5,
        )
    if (~correct_mask).any():
        ax.hist(
            df.loc[~correct_mask, "predictive_entropy"],
            bins=30, alpha=0.7, label="Incorrect",
            color=ETH_COLORS["red"], edgecolor="white", linewidth=0.5,
        )
    ax.set_xlabel("Predictive Entropy  H[p̄]")
    ax.set_ylabel("Count")
    ax.set_title(f"TTA Entropy Distribution — {task}", color=ETH_COLORS["petrol"])
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
    fig.tight_layout()
    fig.savefig(out_dir / f"entropy_distribution_{task}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Entropy vs accuracy ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(df))
    ax.scatter(
        df["predictive_entropy"],
        df["correct"] + jitter,
        s=15, alpha=0.5,
        c=[ETH_COLORS["green"] if c else ETH_COLORS["red"] for c in df["correct"]],
        edgecolors="none",
    )
    # Running mean (binned)
    n_bins = min(20, max(5, len(df) // 10))
    bins = np.linspace(df["predictive_entropy"].min(), df["predictive_entropy"].max(), n_bins + 1)
    bin_centres, bin_accs = [], []
    for i in range(n_bins):
        mask = (df["predictive_entropy"] >= bins[i]) & (df["predictive_entropy"] < bins[i + 1])
        if mask.sum() >= 3:
            bin_centres.append((bins[i] + bins[i + 1]) / 2)
            bin_accs.append(df.loc[mask, "correct"].mean())
    if bin_centres:
        ax.plot(bin_centres, bin_accs, "-o", color=ETH_COLORS["petrol"],
                linewidth=2, markersize=5, label="Binned accuracy")
        ax.legend(framealpha=0.9)
    ax.set_xlabel("Predictive Entropy  H[p̄]")
    ax.set_ylabel("Correct (0/1)")
    ax.set_title(f"Entropy vs Accuracy — {task}", color=ETH_COLORS["petrol"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])
    ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
    fig.tight_layout()
    fig.savefig(out_dir / f"entropy_vs_accuracy_{task}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Disagreement rate distribution ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    if correct_mask.any():
        ax.hist(
            df.loc[correct_mask, "disagreement_rate"],
            bins=30, alpha=0.7, label="Correct",
            color=ETH_COLORS["green"], edgecolor="white", linewidth=0.5,
        )
    if (~correct_mask).any():
        ax.hist(
            df.loc[~correct_mask, "disagreement_rate"],
            bins=30, alpha=0.7, label="Incorrect",
            color=ETH_COLORS["red"], edgecolor="white", linewidth=0.5,
        )
    ax.set_xlabel("TTA Disagreement Rate")
    ax.set_ylabel("Count")
    ax.set_title(f"TTA Disagreement Distribution — {task}", color=ETH_COLORS["petrol"])
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
    fig.tight_layout()
    fig.savefig(out_dir / f"disagreement_rate_{task}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Uncertainty plots for {task} → {out_dir}")
    return out_dir


# ════════════════════════════════════════════════════════════════════════════
# RANKING-SCORE TTA UNCERTAINTY  (σ_TTA)
# ════════════════════════════════════════════════════════════════════════════

def compute_rank_sigma(per_view_scores: np.ndarray) -> float:
    """Compute σ_TTA for a single sample.

    σ_TTA(t) = sqrt( 1/N_TTA · Σ_v (s_v(t) − s̄(t))² )
             = np.std(per_view_scores)

    Parameters
    ----------
    per_view_scores : ndarray (N_views,)
        Raw ranking scores across N_TTA augmentation views.

    Returns
    -------
    float  — σ_TTA for this sample.
    """
    return float(np.std(per_view_scores))


def _threshold_proximity(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Minimum absolute distance from each score to the nearest threshold.

    Parameters
    ----------
    scores : (N,)
    thresholds : (K-1,) — boundary values between adjacent grades.

    Returns
    -------
    ndarray (N,)  — min |s - thr_k| over all boundaries k.
    """
    if len(thresholds) == 0:
        return np.ones(len(scores)) * np.inf
    # (N, K-1) matrix, take row-wise min
    dists = np.abs(scores[:, None] - thresholds[None, :])
    return dists.min(axis=1)


def plot_rank_sigma_overview(
    sigma_tta: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    task: str,
    out_dir: Path,
    flag_pct: float = 90.0,
) -> Path:
    """Three-panel uncertainty overview for a single grading task.

    Panel 1 — σ_TTA distribution (correct vs incorrect)
    Panel 2 — σ_TTA vs threshold proximity (scatter, flag region)
    Panel 3 — Mean σ_TTA per grade (bar chart)

    Parameters
    ----------
    sigma_tta : (N,)
    scores : (N,)
    labels : (N,)
    thresholds : (K-1,)
    task : str
    out_dir : Path
    flag_pct : float
        Samples whose σ_TTA exceeds this percentile AND whose threshold
        proximity is below the median are flagged for expert review.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping σ_TTA plots")
        return out_dir

    try:
        from spineranknet.visualization.evaluation import ETH_COLORS
    except Exception:
        ETH_COLORS = {
            "blue": "#0069B4", "red": "#C1002A", "petrol": "#007A92",
            "green": "#91C34A", "gray": "#B3B3B3", "purple": "#6B3FA0",
        }

    proximity = _threshold_proximity(scores, thresholds)  # (N,)
    sigma_thr = np.percentile(sigma_tta, flag_pct)
    prox_thr = np.median(proximity)
    flagged = (sigma_tta >= sigma_thr) & (proximity <= prox_thr)

    # Predict grade using thresholds
    preds = np.searchsorted(np.sort(thresholds), scores)
    preds = np.clip(preds, 0, labels.max() if len(labels) else 0)
    correct = (preds == labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: σ_TTA distribution (correct vs incorrect) ────────────
    ax = axes[0]
    for is_correct, color, lbl in [
        (True,  ETH_COLORS["green"], "Correct"),
        (False, ETH_COLORS["red"],   "Incorrect"),
    ]:
        subset = sigma_tta[correct == is_correct]
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.7, color=color, label=lbl,
                    edgecolor="white", linewidth=0.5)
    ax.axvline(sigma_thr, color=ETH_COLORS["purple"], linestyle="--", linewidth=1.5,
               label=f"p{flag_pct:.0f} = {sigma_thr:.3f}")
    ax.set_xlabel(r"$\sigma_\mathrm{TTA}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"σ_TTA Distribution\n{task}", fontsize=12,
                 color=ETH_COLORS["petrol"], fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: σ_TTA vs threshold proximity ─────────────────────────
    ax = axes[1]
    colors_pt = [ETH_COLORS["red"] if f else ETH_COLORS["blue"] for f in flagged]
    ax.scatter(proximity, sigma_tta, c=colors_pt, s=12, alpha=0.5, linewidths=0)
    ax.axhline(sigma_thr, color=ETH_COLORS["purple"], linestyle="--", linewidth=1.5,
               label=f"σ flag threshold (p{flag_pct:.0f})")
    ax.axvline(prox_thr, color=ETH_COLORS["petrol"], linestyle=":", linewidth=1.5,
               label="Median proximity")
    # Shade flagged region
    ax.fill_betweenx([sigma_thr, sigma_tta.max() * 1.05],
                     0, prox_thr, alpha=0.08, color=ETH_COLORS["red"],
                     label=f"Flagged ({flagged.sum()} / {len(flagged)})")
    ax.set_xlabel("Min distance to threshold  |s − thr|", fontsize=12)
    ax.set_ylabel(r"$\sigma_\mathrm{TTA}$", fontsize=12)
    ax.set_title(f"σ_TTA vs Threshold Proximity\n{task}", fontsize=12,
                 color=ETH_COLORS["petrol"], fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, color=ETH_COLORS["gray"])
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: Mean σ_TTA per grade ─────────────────────────────────
    ax = axes[2]
    unique_grades = np.unique(labels)
    grade_means = [sigma_tta[labels == g].mean() for g in unique_grades]
    grade_stds  = [sigma_tta[labels == g].std()  for g in unique_grades]
    grade_ns    = [(labels == g).sum() for g in unique_grades]

    bar_colors = [ETH_COLORS["blue"]] * len(unique_grades)
    bars = ax.bar(unique_grades, grade_means, color=bar_colors, width=0.65,
                  zorder=3, yerr=grade_stds, capsize=4,
                  error_kw={"ecolor": ETH_COLORS["gray"], "linewidth": 1.5})
    for bar, g, n, mn in zip(bars, unique_grades, grade_ns, grade_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(grade_stds) * 0.1,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="#555555")
    ax.set_xlabel("True Grade", fontsize=12)
    ax.set_ylabel(r"Mean $\sigma_\mathrm{TTA}$", fontsize=12)
    ax.set_xticks(unique_grades)
    ax.set_title(f"Mean σ_TTA per Grade\n{task}", fontsize=12,
                 color=ETH_COLORS["petrol"], fontweight="bold")
    ax.grid(axis="y", alpha=0.25, color=ETH_COLORS["gray"], zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Ranking TTA Uncertainty — {task}", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"rank_sigma_overview_{task}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_and_plot_rank_uncertainty(
    rank_results: Dict[str, dict],
    tasks: List[str],
    thresholds: Dict[str, np.ndarray],
    out_dir: Union[str, Path],
    flag_pct: float = 90.0,
) -> Optional[Path]:
    """Full σ_TTA pipeline: CSV export + overview plots for all tasks.

    Parameters
    ----------
    rank_results : dict
        ``{task: {"scores": [...], "labels": [...], "sigma_tta": [...], ...}}``
    tasks : list of str
    thresholds : dict  ``{task: np.ndarray}``
    out_dir : path-like
    flag_pct : float
        Samples above this σ_TTA percentile and near a threshold are flagged.

    Returns
    -------
    Path to the output directory, or None if nothing was computed.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    any_plotted = False

    for task in tasks:
        rd = rank_results.get(task, {})
        sigma_list = rd.get("sigma_tta", [])
        scores_list = rd.get("scores", [])
        labels_list = rd.get("labels", [])

        if not sigma_list:
            continue
        if len(sigma_list) != len(scores_list):
            logger.warning(
                f"  [{task}] sigma_tta length ({len(sigma_list)}) != "
                f"scores length ({len(scores_list)}) — skipping"
            )
            continue

        sigma = np.array(sigma_list, dtype=np.float32)
        scores = np.array(scores_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)
        thr = np.array(thresholds.get(task, []), dtype=np.float32)
        proximity = _threshold_proximity(scores, thr)

        sigma_thr = float(np.percentile(sigma, flag_pct))
        prox_thr = float(np.median(proximity))
        flagged = (sigma >= sigma_thr) & (proximity <= prox_thr)
        flag_rate = float(flagged.mean())

        logger.info(
            f"  {task:30s}  mean_σ={sigma.mean():.4f}  "
            f"std_σ={sigma.std():.4f}  flagged={flagged.sum()}/{len(sigma)} "
            f"({flag_rate * 100:.1f}%)"
        )

        # ── Per-sample CSV ────────────────────────────────────────────
        csv_path = out_dir / f"{task}_rank_sigma.csv"
        ids = rd.get("ids", [""] * len(sigma))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "true_label", "score", "sigma_tta",
                "threshold_proximity", "flagged",
            ])
            for sid, y, s, sg, pr, fl in zip(
                ids, labels, scores, sigma, proximity, flagged
            ):
                writer.writerow([sid, int(y), f"{s:.5f}", f"{sg:.5f}",
                                 f"{pr:.5f}", int(fl)])

        # ── Summary stats ─────────────────────────────────────────────
        summary_rows.append({
            "task": task,
            "n_samples": len(sigma),
            "mean_sigma_tta": float(sigma.mean()),
            "std_sigma_tta": float(sigma.std()),
            "median_sigma_tta": float(np.median(sigma)),
            "p90_sigma_tta": sigma_thr,
            "flag_rate_pct": flag_rate * 100,
            "n_flagged": int(flagged.sum()),
        })

        # ── Per-task overview plot ─────────────────────────────────────
        try:
            plot_rank_sigma_overview(
                sigma, scores, labels, thr, task, out_dir, flag_pct=flag_pct
            )
            any_plotted = True
        except Exception as plot_err:
            logger.warning(f"  [{task}] plot failed: {plot_err}")

    # ── Cross-task summary CSV ─────────────────────────────────────────
    if summary_rows:
        import csv as csv_mod
        summary_path = out_dir / "rank_sigma_summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"\n  σ_TTA summary → {summary_path}")

    # ── Cross-task σ_TTA bar chart ─────────────────────────────────────
    if summary_rows and any_plotted:
        try:
            import matplotlib.pyplot as plt
            from spineranknet.visualization.evaluation import ETH_COLORS
        except Exception:
            return out_dir if any_plotted else None

        fig, ax = plt.subplots(figsize=(max(8, len(summary_rows) * 1.1), 5))
        task_names = [r["task"] for r in summary_rows]
        means = [r["mean_sigma_tta"] for r in summary_rows]
        stds = [r["std_sigma_tta"] for r in summary_rows]
        flag_rates = [r["flag_rate_pct"] for r in summary_rows]

        x = np.arange(len(task_names))
        bars = ax.bar(x, means, color=ETH_COLORS["blue"], width=0.65, zorder=3,
                      yerr=stds, capsize=4,
                      error_kw={"ecolor": ETH_COLORS["gray"], "linewidth": 1.5})
        # Annotate flag rate on each bar
        for bar, fr in zip(bars, flag_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{fr:.0f}%↑",
                ha="center", va="bottom", fontsize=8, color=ETH_COLORS["red"],
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(r"Mean $\sigma_\mathrm{TTA}$ (± std)", fontsize=12)
        ax.set_title(
            r"Ranking TTA Uncertainty $\sigma_\mathrm{TTA}$ per Task"
            "\n(% = flag rate: high σ near threshold)",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(out_dir / "rank_sigma_all_tasks.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Cross-task σ_TTA chart → {out_dir / 'rank_sigma_all_tasks.png'}")

    return out_dir if any_plotted else None
