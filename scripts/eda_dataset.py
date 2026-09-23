#!/usr/bin/env python
"""Basic exploratory data analysis (EDA) of the Genodisc grading dataset.

Companion to the MICCAI 2026 paper "Be Indiscrete: The Benefits of Learning
Continuous Spine Degeneration Severity Scores". Reads only the label JSON,
the demographics/scan CSVs and a few dozen T2w IVD volumes, so it runs on a
CPU in about one minute.

Outputs (written to --out-dir, default ``results/eda/``)
    fig_grade_distribution.{png,pdf}  grade prevalence per task (100 % stacked bars)
    fig_grade_by_level.{png,pdf}      grade x IVD-level counts per task (heatmaps)
    fig_dataset_overview.{png,pdf}    subjects per centre/split, age/sex, T1w/T2w availability
    prevalence_table.{csv,md}         per task: n, % normal, % most severe grade
    level_grade_counts.csv            long table (task, level, grade, split counts)
    fig_severity_grid.{png,pdf}       one T2w example per (severity, task); rows are
                                      Normal/Mild/Moderate/Severe, one IVD level per column
    severity_grid_manifest.csv        which IVD volume is shown in every grid tile
                                      (contains subject IDs and scan dates: keep
                                      it private, do not commit it)

Conventions
    * Labels are decoded with ``extract_labels`` in ``scripts/_eda_helpers.py``
      (the 0-based grading scales of ``spineranknet.config.TASK_DEFINITIONS``).
      Pfirrmann 1-5 becomes 0-4.
    * An IVD is counted when it has a ``T2_S1`` series (N = 12,078). Counts pool
      Train + Val + Test unless a column says otherwise.
    * Grid tiles are the mid-sagittal slice of the paper's network input: the
      central 128 x 256 crop of the 192 x 320 x 15 IVD volume (slice index 7),
      windowed to the 1st-99th intensity percentile. Anterior is on the left.
    * The grid has four severity rows. Pfirrmann shows Gr. I / III / IV / V
      (Gr. II, a non-degenerated disc, is skipped). Spondylolisthesis has no
      mild grade, so its Mild and Moderate rows show the same moderate IVD.
      Columns use the full task names in alphabetical order, with the IVD level
      under each column. Grade badges appear only where they add information
      (the Pfirrmann grade, and the repeated moderate Spondylolisthesis tile).

Data (default root: $GENODISC_ROOT, else ``data/GENODISCv2``)
    genodisc_grading.json                  labels (split -> IVD key -> score)
    IVDs-numpy/<key>_T2_S1.npy             IVD volumes (192 x 320 x 15)
    genodisc-pathologies-demographics.csv  age / sex per session
    genodisc.csv                           scan table (acquisition centre)
    Every path can also be set with a flag (--json, --image-folder, ...).

Usage (from the repository root)
    export GENODISC_ROOT=/path/to/GENODISCv2
    python scripts/eda_dataset.py
    python scripts/eda_dataset.py --out-dir results/eda_seed1 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]          # scripts/ and the repo root
from _eda_helpers import (  # noqa: E402
    ETH_BLUE, ETH_BRONZE, ETH_GREEN, ETH_GREY, ETH_LGREY, ETH_ORANGE,
    ETH_PETROL, ETH_PURPLE, ETH_RED, FIG_FULL_W, INVALID_LABEL, IVD_LEVELS,
    apply_eth_style, extract_labels)

DATA = Path(os.environ.get("GENODISC_ROOT", "data/GENODISCv2"))

# ── tasks ────────────────────────────────────────────────────────────────────
# The 11 grading tasks reported in the paper (training TASKS minus Modic/IVDlevel).
PAPER_TASKS = [
    "Pfirrmann", "Narrowing", "CentralCanalStenosis", "Spondylolisthesis",
    "UpperEndplateDefect", "LowerEndplateDefect", "ForaminalStenosisLeft",
    "ForaminalStenosisRight", "Herniation", "AnteriorBulging", "PosteriorBulging",
]
# Trained as auxiliary heads but not reported in the paper (nominal Modic type).
EXTRA_TASKS = ["UpperModic", "LowerModic"]
ALL_TASKS = PAPER_TASKS + EXTRA_TASKS

LONG_NAME = {
    "Pfirrmann": "Pfirrmann grade", "Narrowing": "Disc narrowing",
    "CentralCanalStenosis": "Central canal stenosis",
    "Spondylolisthesis": "Spondylolisthesis",
    "UpperEndplateDefect": "Upper endplate defect",
    "LowerEndplateDefect": "Lower endplate defect",
    "ForaminalStenosisLeft": "Foraminal stenosis (L)",
    "ForaminalStenosisRight": "Foraminal stenosis (R)",
    "Herniation": "Herniation", "AnteriorBulging": "Anterior bulging",
    "PosteriorBulging": "Posterior bulging",
    "UpperModic": "Upper Modic type†", "LowerModic": "Lower Modic type†",
}
# Full task names for the severity-grid headers, wrapped to fit one tile.
GRID_NAME = {
    "AnteriorBulging": "Anterior\nBulging",
    "CentralCanalStenosis": "Central Canal\nStenosis",
    "Narrowing": "Disc\nNarrowing",
    "ForaminalStenosisLeft": "Foraminal\nStenosis Left",
    "ForaminalStenosisRight": "Foraminal\nStenosis Right",
    "Herniation": "Herniation",
    "LowerEndplateDefect": "Lower Endplate\nDefect",
    "Pfirrmann": "Pfirrmann\nGrade",
    "PosteriorBulging": "Posterior\nBulging",
    "Spondylolisthesis": "Spondylolisthesis",
    "UpperEndplateDefect": "Upper Endplate\nDefect",
}
# Grade names as printed on the paper's badges.
GRADE_NAMES = {
    "Pfirrmann": ["Gr. I", "Gr. II", "Gr. III", "Gr. IV", "Gr. V"],
    "Spondylolisthesis": ["None", "Mod.", "Sev."],
    "UpperModic": ["None", "Type I", "Type II", "Type III"],
    "LowerModic": ["None", "Type I", "Type II", "Type III"],
}
DEFAULT_GRADES = ["None", "Mild", "Mod.", "Sev."]
# Severity position of every grade: 0 normal, 1 mild, 2 moderate, 3 severe.
# Spondylolisthesis has no mild grade; Pfirrmann keeps its own five grades.
SEVERITY_POS = {"Spondylolisthesis": [0, 2, 3], "Pfirrmann": [0, 1, 2, 3, 4]}
POS_COLOR = {0: ETH_GREEN, 1: ETH_PETROL, 2: ETH_BRONZE, 3: ETH_RED}
# Severity grid: the grade index shown in each row (Normal, Mild, Moderate, Severe).
GRID_ROWS = ["Normal", "Mild", "Moderate", "Severe"]
GRID_GRADES = {"Pfirrmann": [0, 2, 3, 4],          # Gr. I, III, IV, V (skip Gr. II)
               "Spondylolisthesis": [0, 1, 1, 2]}  # no mild grade: repeat moderate
PFIRRMANN_COLOR = [ETH_GREEN, ETH_PETROL, ETH_BRONZE, ETH_PURPLE, ETH_RED]
# Correlated findings that are not counted as "other pathology" when picking
# a clean example for a task (e.g. a Pfirrmann V disc is expected to be narrow).
PARTNERS = [
    {"Pfirrmann", "Narrowing"},
    {"UpperEndplateDefect", "LowerEndplateDefect"},
    {"ForaminalStenosisLeft", "ForaminalStenosisRight"},
    {"Herniation", "PosteriorBulging", "CentralCanalStenosis"},
]
SPLITS = ["Train", "Val", "Test"]
SPLIT_COLOR = {"Train": ETH_BLUE, "Val": ETH_ORANGE, "Test": ETH_PURPLE}
TILE_H, TILE_W = 128, 256        # paper input crop (H x W); 12 of 15 slices are used


def grade_names(task):
    return GRADE_NAMES.get(task, DEFAULT_GRADES)


def grid_grades(task):
    """Grade index shown in each severity row of the grid."""
    return GRID_GRADES.get(task, [0, 1, 2, 3])


def grade_colors(task):
    if task == "Pfirrmann":
        return PFIRRMANN_COLOR
    return [POS_COLOR[p] for p in SEVERITY_POS.get(task, [0, 1, 2, 3])]


def level_tag(level, sep="-"):
    """'L4L5' -> 'L4-L5' (or 'L4/L5' with sep='/')."""
    return level.replace("L", sep + "L", 1) if level.startswith("T") else \
        level[:2] + sep + level[2:]


def save(fig, out_dir, name, dpi=300):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# ── data ─────────────────────────────────────────────────────────────────────
def load_ivd_table(json_path):
    """One row per IVD entry: key, subject, level, split, T1/T2 flags, T2 labels."""
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for split in SPLITS:
        for key, rec in data.get(split, {}).items():
            pid, date, level = key.split("_")
            seqs = rec.get("sequence", [])
            row = {"key": key, "subject": pid, "date": date, "level": level,
                   "split": split, "has_T1": "T1_S1" in seqs, "has_T2": "T2_S1" in seqs}
            labels = extract_labels(rec["score"], "T2")
            row.update({t: labels[t] for t in ALL_TASKS})
            rows.append(row)
    return pd.DataFrame(rows)


def load_centres(scans_csv):
    """subject -> acquisition centre (first path component of scan_path)."""
    g = pd.read_csv(scans_csv, usecols=["id", "scan_path"])
    g["centre"] = g["scan_path"].astype(str).str.extract(r"^\['([^/]+)/")[0]
    return g.dropna(subset=["centre"]).drop_duplicates("id").set_index("id")["centre"]


def load_demographics(demo_csv, sessions):
    """One row per subject (earliest session present in the label JSON)."""
    dm = pd.read_csv(demo_csv, usecols=["id", "date", "age", "sex"])
    dm = dm.drop_duplicates(["id", "date"])
    dm = dm[(dm["id"] + "_" + dm["date"]).isin(sessions)]
    dm = dm.sort_values(["id", "date"]).drop_duplicates("id")
    dm["age"] = pd.to_numeric(dm["age"], errors="coerce").where(lambda a: a >= 0)
    dm["sex"] = dm["sex"].where(dm["sex"].isin(["F", "M"]), "Unknown")
    return dm


def count_grades(t2, task):
    k = len(grade_names(task))
    v = t2[task].to_numpy()
    return np.array([(v == g).sum() for g in range(k)])


# ── (a) grade distribution ───────────────────────────────────────────────────
def fig_grade_distribution(t2, out_dir):
    tasks = ALL_TASKS
    fig, ax = plt.subplots(figsize=(FIG_FULL_W, 4.6))
    ypos = {t: i + (0.6 if t in EXTRA_TASKS else 0) for i, t in enumerate(tasks)}
    for t in tasks:
        cnt = count_grades(t2, t)
        pct = 100 * cnt / cnt.sum()
        left = 0.0
        for g, (p, c) in enumerate(zip(pct, grade_colors(t))):
            ax.barh(ypos[t], p, left=left, color=c, height=0.72, edgecolor="white",
                    linewidth=0.5)
            if p >= 6:
                lab = f"{p:.0f}%"
                if t == "Pfirrmann":
                    lab = f"{grade_names(t)[g].split()[-1]}  {p:.0f}%"
                ax.text(left + p / 2, ypos[t], lab, ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
            left += p
        ax.text(101.5, ypos[t], f"{cnt.sum():,}", va="center", ha="left",
                fontsize=6.5, color=ETH_GREY)
    ax.text(101.5, -0.95, "N", va="center", ha="left", fontsize=6.5,
            color=ETH_GREY, fontweight="bold")
    ax.set_yticks([ypos[t] for t in tasks], [LONG_NAME[t] for t in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("IVDs with a T2w series (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=ETH_GREEN, label="Normal / None (Pf. I)"),
               Patch(color=ETH_PETROL, label="Mild (Pf. II)"),
               Patch(color=ETH_BRONZE, label="Moderate (Pf. III)"),
               Patch(color=ETH_PURPLE, label="Pf. IV"),
               Patch(color=ETH_RED, label="Severe (Pf. V)")]
    ax.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              fontsize=6.8, handlelength=1.2, columnspacing=1.0)
    fig.text(0.01, 0.005,
             "Pooled Train+Val+Test, T2w labels.  Spondylolisthesis has no mild grade.  "
             "† Modic colours = None / Type I / II / III (nominal); trained but "
             "not reported in the paper.", fontsize=6, color=ETH_GREY)
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    save(fig, out_dir, "fig_grade_distribution")


# ── (b) grade x level ────────────────────────────────────────────────────────
def fig_grade_by_level(t2, out_dir):
    tasks = ALL_TASKS
    ncol = 4
    nrow = int(np.ceil((len(tasks) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_FULL_W, 1.55 * nrow + 0.3))
    cmap = LinearSegmentedColormap.from_list("eth_petrol", ["#FFFFFF", ETH_PETROL, "#00384A"])
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=100)
    for ax, t in zip(axes.flat, tasks):
        names = grade_names(t)
        mat = np.array([[((t2["level"] == lv) & (t2[t] == g)).sum() for lv in IVD_LEVELS]
                        for g in range(len(names))])
        pct = 100 * mat / np.maximum(mat.sum(0, keepdims=True), 1)
        ax.imshow(pct, cmap=cmap, norm=norm, aspect="auto")
        for g in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                n = mat[g, j]
                ax.text(j, g, f"{n}" if n else "–", ha="center", va="center",
                        fontsize=5.3, color="white" if pct[g, j] > 40 else
                        ("#9A9A9A" if n == 0 else "black"))
        ax.set_xticks(range(len(IVD_LEVELS)),
                      [level_tag(lv).replace("-", "\n") for lv in IVD_LEVELS], fontsize=5.5)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_title(LONG_NAME[t], fontsize=7, fontweight="bold", pad=3)
        ax.grid(False)
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    cax = axes.flat[len(tasks)].inset_axes([0.08, 0.55, 0.84, 0.1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 5, 25, 50, 100])
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)
    cb.ax.grid(False)
    cb.set_label("% of IVDs at that level", fontsize=6)
    axes.flat[len(tasks)].text(
        0.5, 0.18, "numbers = IVD counts (T2w, all splits)\n– = no example",
        transform=axes.flat[len(tasks)].transAxes, ha="center", va="center",
        fontsize=5.8, color=ETH_GREY)
    fig.tight_layout(h_pad=1.1, w_pad=0.6)
    save(fig, out_dir, "fig_grade_by_level")


def level_grade_counts(t2):
    rows = []
    for t in ALL_TASKS:
        for lv in IVD_LEVELS:
            for g, name in enumerate(grade_names(t)):
                m = (t2["level"] == lv) & (t2[t] == g)
                row = {"task": t, "level": lv, "grade": g, "grade_name": name,
                       "n_all": int(m.sum())}
                row.update({f"n_{s.lower()}": int((m & (t2["split"] == s)).sum())
                            for s in SPLITS})
                rows.append(row)
    return pd.DataFrame(rows)


# ── (c) dataset overview ─────────────────────────────────────────────────────
def fig_dataset_overview(df, centres, demo, out_dir):
    subj = df.drop_duplicates("subject")[["subject", "split"]].copy()
    subj["centre"] = subj["subject"].map(centres).fillna("Unknown")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_FULL_W, 2.3),
                             gridspec_kw={"width_ratios": [1.15, 1.1, 1.0]})

    # A: subjects per centre, stacked by split
    ax = axes[0]
    tab = subj.groupby(["centre", "split"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=SPLITS, fill_value=0)
    tab = tab.loc[tab.sum(1).sort_values().index]
    left = np.zeros(len(tab))
    for s in SPLITS:
        ax.barh(tab.index, tab[s], left=left, color=SPLIT_COLOR[s], height=0.7, label=s)
        left += tab[s].to_numpy()
    for i, tot in enumerate(left):
        ax.text(tot + 15, i, f"{int(tot)}", va="center", fontsize=6, color=ETH_GREY)
    ax.set_xlim(0, left.max() * 1.18)
    ax.set_xlabel("Subjects")
    ax.set_title(f"A  Subjects per centre (n={len(subj):,})", loc="left", fontsize=7.5,
                 fontweight="bold")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(axis="y", visible=False)

    # B: age by sex
    ax = axes[1]
    bins = np.arange(10, 95, 5)
    known = demo.dropna(subset=["age"])
    parts = [(s, c) for s, c in (("F", ETH_PURPLE), ("M", ETH_BLUE), ("Unknown", ETH_LGREY))
             if (known["sex"] == s).any()]
    ax.hist([known.loc[known["sex"] == s, "age"] for s, _ in parts], bins=bins,
            stacked=True, color=[c for _, c in parts],
            label=[f"{s} (n={(demo['sex'] == s).sum()})" for s, _ in parts])
    ax.set_xlabel("Age at scan (years)")
    ax.set_ylabel("Subjects")
    n_missing = demo["age"].isna().sum()
    ax.set_title(f"B  Age / sex (age missing: {n_missing})", loc="left", fontsize=7.5,
                 fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
    ax.legend(fontsize=6, loc="upper left")

    # C: IVD volumes per split by sequence availability
    ax = axes[2]
    cats = [("T1w + T2w", df.has_T1 & df.has_T2, ETH_PETROL),
            ("T2w only", ~df.has_T1 & df.has_T2, ETH_GREEN),
            ("T1w only", df.has_T1 & ~df.has_T2, ETH_BRONZE)]
    bottom = np.zeros(len(SPLITS))
    for name, mask, col in cats:
        vals = np.array([(mask & (df.split == s)).sum() for s in SPLITS])
        ax.bar(SPLITS, vals, bottom=bottom, color=col, width=0.65,
               label=f"{name} ({int(mask.sum()):,})")
        bottom += vals
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 120, f"{int(tot):,}", ha="center", fontsize=6, color=ETH_GREY)
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_ylabel("IVD volumes")
    ax.set_title("C  IVDs per split", loc="left", fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=5.8, loc="center right", bbox_to_anchor=(1.0, 0.6))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, out_dir, "fig_dataset_overview")
    return subj


# ── (d) prevalence table ─────────────────────────────────────────────────────
def prevalence_table(t2, out_dir):
    rows = []
    for t in ALL_TASKS:
        cnt = count_grades(t2, t)
        n = int(cnt.sum())
        names = grade_names(t)
        rows.append({
            "task": t, "in_paper": t in PAPER_TASKS, "K": len(names),
            "grades": " / ".join(names), "n": n,
            "counts": " / ".join(f"{c}" for c in cnt),
            "pct_normal": round(100 * cnt[0] / n, 1),
            "most_severe_grade": names[-1],
            "n_most_severe": int(cnt[-1]),
            "pct_most_severe": round(100 * cnt[-1] / n, 1),
            "pct_abnormal": round(100 * (n - cnt[0]) / n, 1),
            "n_invalid": int((t2[t] == INVALID_LABEL).sum()),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "prevalence_table.csv", index=False)

    paper = tab[tab.in_paper & (tab.task != "Pfirrmann")]
    lines = [
        "# Genodisc grade prevalence (T2w IVDs, Train+Val+Test pooled)",
        "",
        "| Task | In paper | Grades | n | Counts per grade | % normal | % most severe |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for r in tab.itertuples():
        lines.append(f"| {LONG_NAME[r.task]} | {'yes' if r.in_paper else 'no'} | "
                     f"{r.grades} | {r.n:,} | {r.counts} | {r.pct_normal:.1f} | "
                     f"{r.pct_most_severe:.1f} ({r.most_severe_grade}) |")
    ccs = tab.set_index("task").loc["CentralCanalStenosis"]
    pf = tab.set_index("task").loc["Pfirrmann"]
    pf_cnt = np.array([int(c) for c in pf["counts"].split(" / ")])
    lines += [
        "",
        "Pfirrmann has no 'normal' grade: Gr. I is shown as % normal; "
        f"Gr. I+II = {100 * pf_cnt[:2].sum() / pf_cnt.sum():.1f}%.",
        "",
        "## Check against the paper text",
        "",
        "Paper: *60-96% normal, severe 2-10% (severe stenosis 2%)*.",
        "",
        f"- % normal over the 10 non-Pfirrmann paper tasks: "
        f"{paper.pct_normal.min():.1f}-{paper.pct_normal.max():.1f}% "
        f"(min {LONG_NAME[paper.loc[paper.pct_normal.idxmin(), 'task']]}, "
        f"max {LONG_NAME[paper.loc[paper.pct_normal.idxmax(), 'task']]}).",
        f"- % most severe grade over the same tasks: "
        f"{paper.pct_most_severe.min():.1f}-{paper.pct_most_severe.max():.1f}% "
        f"(min {LONG_NAME[paper.loc[paper.pct_most_severe.idxmin(), 'task']]}, "
        f"max {LONG_NAME[paper.loc[paper.pct_most_severe.idxmax(), 'task']]}).",
        f"- Severe central canal stenosis: {ccs.pct_most_severe:.1f}% "
        f"({ccs.n_most_severe} of {ccs.n:,}).",
        "- Tasks whose most severe grade is below 2%: "
        + ", ".join(f"{LONG_NAME[r.task]} {r.pct_most_severe:.1f}%"
                    for r in paper.itertuples() if r.pct_most_severe < 2) + ".",
    ]
    (out_dir / "prevalence_table.md").write_text("\n".join(lines) + "\n")
    print("  wrote prevalence_table.csv/.md")
    return tab


# ── (e) severity grid ────────────────────────────────────────────────────────
class VolumeReader:
    """Loads the T2w IVD volume (H=192, W=320, slices=15) for an IVD key."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def path(self, key):
        return self.folder / f"{key}_T2_S1.npy"

    def tile(self, key):
        p = self.path(key)
        if not p.exists():
            return None
        vol = np.load(p)
        r0 = max(0, (vol.shape[0] - TILE_H) // 2)
        c0 = max(0, (vol.shape[1] - TILE_W) // 2)
        im = vol[r0:r0 + TILE_H, c0:c0 + TILE_W, vol.shape[2] // 2].astype(np.float32)
        lo, hi = np.percentile(im, [1, 99])
        if not np.isfinite(im).all() or hi - lo < 1e-3 or (im == 0).mean() > 0.2:
            return None        # empty / padded / corrupt slice
        return np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)


def sharpness(im):
    """Mean absolute second difference of a [0, 1] tile. Low values flag
    blurry (low-resolution, upsampled) scans; typical tiles score 0.02-0.05."""
    return float(np.abs(np.diff(im, 2, axis=0)).mean() + np.abs(np.diff(im, 2, axis=1)).mean())


def other_pathology(t2, task):
    """Number of moderate-or-worse findings in other (non-partner) paper tasks."""
    skip = {task}.union(*[g for g in PARTNERS if task in g])
    pen = np.zeros(len(t2), dtype=int)
    for t in PAPER_TASKS:
        if t in skip:
            continue
        thr = 3 if t == "Pfirrmann" else (1 if t == "Spondylolisthesis" else 2)
        pen += (t2[t].to_numpy() >= thr).astype(int)
    return pen


def choose_levels(t2, tasks, min_candidates=3):
    """Pick one IVD level per task where every grade has an example, spreading
    the levels across rows: prefer levels with >= min_candidates examples of the
    rarest grade, then least-used levels, then levels different from the
    neighbouring rows, then the level with the most examples of the rarest grade.
    The most constrained tasks (fewest feasible levels) are assigned first."""
    rarest = {}
    for t in tasks:
        grades = sorted(set(grid_grades(t)))
        rarest[t] = {lv: min(((t2.level == lv) & (t2[t] == g)).sum() for g in grades)
                     for lv in IVD_LEVELS}
    feasible = {t: [lv for lv in IVD_LEVELS if rarest[t][lv] >= 1] for t in tasks}
    usage, chosen = Counter(), {}
    order = sorted(tasks, key=lambda t: (len(feasible[t]), tasks.index(t)))
    for t in order:
        i = tasks.index(t)
        neighbours = {chosen.get(tasks[j]) for j in (i - 1, i + 1) if 0 <= j < len(tasks)}
        cands = feasible[t] or IVD_LEVELS
        chosen[t] = min(cands, key=lambda lv: (rarest[t][lv] < min_candidates, usage[lv],
                                               lv in neighbours, -rarest[t][lv],
                                               IVD_LEVELS.index(lv)))
        usage[chosen[t]] += 1
    return chosen, rarest


def pick_examples(t2, tasks, levels, reader, seed, min_sharpness=0.018):
    """One tile per (task, grid grade) at the task's level. Candidates are shuffled
    with the seed and ordered by (other moderate+ pathology, subject already
    shown in another row). The first candidate that is sharp enough and whose
    subject is not yet in the row is taken; the two conditions are relaxed only
    when no candidate meets them."""
    rng = np.random.default_rng(seed)
    shown, picks, cache = set(), [], {}

    def load(i):
        key = t2.key.iat[i]
        if key not in cache:
            cache[key] = reader.tile(key)
        return cache[key]

    for t in tasks:
        pen = other_pathology(t2, t)
        used_row = set()
        for g in sorted(set(grid_grades(t))):
            gname = grade_names(t)[g]
            rows = ";".join(GRID_ROWS[r] for r, gg in enumerate(grid_grades(t)) if gg == g)
            idx = np.flatnonzero(((t2.level == levels[t]) & (t2[t] == g)).to_numpy())
            idx = idx[rng.permutation(len(idx))]
            order = sorted(idx, key=lambda i: (pen[i], t2.subject.iat[i] in shown))
            pick = None
            for new_subject, min_sharp in ((True, min_sharpness), (True, 0), (False, 0)):
                for i in order:
                    if new_subject and t2.subject.iat[i] in used_row:
                        continue
                    im = load(i)
                    if im is not None and sharpness(im) >= min_sharp:
                        pick = (i, im)
                        break
                if pick:
                    break
            if pick is None:
                picks.append({"task": t, "level": levels[t], "grade": g, "grade_name": gname,
                              "grid_rows": rows, "n_candidates": len(idx), "image": None})
                continue
            i, im = pick
            used_row.add(t2.subject.iat[i])
            shown.add(t2.subject.iat[i])
            picks.append({
                "task": t, "level": levels[t], "grade": g, "grade_name": gname,
                "grid_rows": rows, "key": t2.key.iat[i], "subject": t2.subject.iat[i], "date": t2.date.iat[i],
                "split": t2.split.iat[i], "sequence": "T2_S1",
                "file": reader.path(t2.key.iat[i]).name,
                "slice_index": 7, "crop_hw": f"{TILE_H}x{TILE_W}",
                "other_moderate_plus": int(pen[i]), "sharpness": round(sharpness(im), 4),
                "n_candidates": len(idx), "image": im,
            })
    return picks


def badge(ax, x, y, text, edge, va):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=6.6,
            color="#1A1A1A", zorder=5,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.55", facecolor="#E4E4E4",
                      edgecolor=edge, linewidth=1.1, alpha=0.93))


def fig_severity_grid(picks, tasks, out_dir):
    """Wide grid: rows are Normal/Mild/Moderate/Severe, columns are tasks in
    alphabetical order of their full name (one IVD level per column, printed
    below). A grade badge is drawn only where the row name does not already say
    the grade: every Pfirrmann tile, and the moderate Spondylolisthesis IVD that
    is repeated in the Mild row (badge in the moderate colour)."""
    tasks = sorted(tasks, key=lambda t: GRID_NAME[t].replace("\n", " "))
    nrow, ncol = len(GRID_ROWS), len(tasks)
    tile_w = 1.3                                     # inches per tile
    fig, axes = plt.subplots(nrow, ncol, figsize=(tile_w * ncol + 0.3,
                                                  tile_w * TILE_H / TILE_W * nrow * 1.04 + 0.35),
                             squeeze=False)
    by_cell = {(p["task"], p["grade"]): p for p in picks}
    for c, t in enumerate(tasks):
        grades = grid_grades(t)
        for r, g in enumerate(grades):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            p = by_cell.get((t, g))
            if p is None or p["image"] is None:
                ax.set_facecolor("#F3F3F3")
                ax.text(0.5, 0.5, f"no {grade_names(t)[g]} example",
                        transform=ax.transAxes, ha="center", va="center", fontsize=6,
                        color="#8A8A8A")
                continue
            ax.imshow(p["image"], cmap="gray", vmin=0, vmax=1, aspect="equal",
                      interpolation="bilinear")
            # colour = severity of the grade shown: the repeated Spondylolisthesis
            # tile in the Mild row keeps the moderate colour
            pos = SEVERITY_POS[t][g] if t == "Spondylolisthesis" else r
            if t == "Pfirrmann" or pos != r:
                badge(ax, 0.035, 0.05, p["grade_name"], POS_COLOR[pos], "bottom")
        axes[0][c].text(0.5, 1.05, GRID_NAME[t], transform=axes[0][c].transAxes,
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        linespacing=1.05)
        level = by_cell.get((t, grades[0]), {}).get("level", "")
        axes[-1][c].text(0.5, -0.05, level_tag(level, "/"), transform=axes[-1][c].transAxes,
                         ha="center", va="top", fontsize=7.5, color="#444444")
    for r, name in enumerate(GRID_ROWS):
        axes[r][0].text(-0.06, 0.5, name, transform=axes[r][0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=8, fontweight="bold",
                        color=POS_COLOR[r])
    fig.subplots_adjust(left=0.02, right=0.998, top=0.9, bottom=0.004,
                        wspace=0.03, hspace=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_severity_grid.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print("  wrote fig_severity_grid.png/.pdf")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=str(DATA / "genodisc_grading.json"),
                    help="Genodisc grading JSON (split -> IVD key -> score)")
    ap.add_argument("--image-folder", default=str(DATA / "IVDs-numpy"),
                    help="folder with the T2w IVD volumes <key>_T2_S1.npy")
    ap.add_argument("--demographics", default=str(DATA / "genodisc-pathologies-demographics.csv"),
                    help="demographics CSV (columns id, date, age, sex)")
    ap.add_argument("--scans-csv", default=str(DATA / "genodisc.csv"),
                    help="scan table; acquisition centre = first component of scan_path")
    ap.add_argument("--out-dir", default="results/eda", help="output folder")
    ap.add_argument("--seed", type=int, default=42, help="seed for picking grid examples")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_eth_style(fontsize=8)
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "standard"

    print("loading labels ...")
    df = load_ivd_table(args.json)
    t2 = df[df.has_T2].reset_index(drop=True)
    print(f"  {len(df):,} IVD entries, {len(t2):,} with T2_S1, "
          f"{df.subject.nunique():,} subjects")

    print("distributions ...")
    fig_grade_distribution(t2, out_dir)
    fig_grade_by_level(t2, out_dir)
    level_grade_counts(t2).to_csv(out_dir / "level_grade_counts.csv", index=False)
    prevalence_table(t2, out_dir)
    centres = load_centres(args.scans_csv)
    demo = load_demographics(args.demographics, set(df.subject + "_" + df.date))
    fig_dataset_overview(df, centres, demo, out_dir)

    print("severity grid ...")
    levels, rarest = choose_levels(t2, PAPER_TASKS)
    for t in PAPER_TASKS:
        print(f"  {t:24s} {levels[t]:6s} (fewest examples of any grade there: "
              f"{rarest[t][levels[t]]})")
    picks = pick_examples(t2, PAPER_TASKS, levels, VolumeReader(args.image_folder), args.seed)
    fig_severity_grid(picks, PAPER_TASKS, out_dir)
    man = pd.DataFrame([{k: v for k, v in p.items() if k != "image"} for p in picks])
    man.to_csv(out_dir / "severity_grid_manifest.csv", index=False)
    print(f"  wrote severity_grid_manifest.csv ({len(man)} tiles)")


if __name__ == "__main__":
    main()
