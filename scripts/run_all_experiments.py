#!/usr/bin/env python3
"""Reproducibility driver for the SpineRankNet paper experiments.

Enumerates (and optionally executes) the training / evaluation commands
behind the paper, all targeting
``python -m spineranknet.baseline.train_ranking_mse``:

* **phase1**             — classification / ordinal-regression baselines
                           trained from scratch (``--mode clf``):
                           CE, CORN, MAE_l (Table 1). The CE run is also the
                           Phase-1 checkpoint that all ranking runs start from.
* **paper**              — SpineRankNet: SpineRank loss + MLP ranking head on
                           the classification logits (``--rank_from_logits``),
                           fine-tuned from the Phase-1 CE checkpoint.
* **ranking_baselines**  — the same Phase-2 protocol with the RankNet and
                           DeepRankSVM losses (Table 1).
* **head_ablation**      — SpineRank loss with mlp / linear / kan / transformer
                           ranking heads.
* **threshold_ablation** — re-evaluates the SpineRankNet (mlp) checkpoint with
                           grid / isotonic / youden / gmm score thresholds.

Output layout (under ``--results-root``, default ``results``)::

    phase1/<clf_loss>/{weights,eval}/...
    ranking/<backbone>/<loss>_<head>/{weights,eval,eval_<threshold>}/...

``ranking/`` is directly readable by ``scripts/generate_ranking_loss_table.py
--results_dir <root>/ranking`` and ``scripts/aggregate_spineranknet_ablation.py
--results_dir <root>/ranking``.

Examples
--------
List the enumerated runs::

    python scripts/run_all_experiments.py --list

Print the commands (default, nothing is executed)::

    python scripts/run_all_experiments.py --group phase1 paper

Execute locally (needs the GENODISC data and a CUDA GPU)::

    GENODISC_ROOT=/path/to/GENODISCv2 \\
        python scripts/run_all_experiments.py --group phase1 paper --run

Ranking runs start from ``<results-root>/phase1/ce/weights/<backbone>/best.pt``
unless ``--clf-checkpoint`` points to another Phase-1 checkpoint.

This driver only builds command strings for ``--list`` / dry-run and does not
import torch.
"""
from __future__ import annotations

import argparse
import dataclasses
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Repository root = parent of this scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Trainer entry point ────────────────────────────────────────────────────
TRAINER_MODULE = "spineranknet.baseline.train_ranking_mse"

DEFAULT_CONFIG = "config/ranking_logit_heads.yaml"
DEFAULT_BACKBONE = "resnet18"

# ── Paper SpineRank hyperparameters ────────────────────────────────────────
MARGIN_BASE = 0.5
MARGIN_SCALE = 1.5
LAMBDA_SIM = 0.25

# ── Training schedule ──────────────────────────────────────────────────────
PHASE1_EPOCHS = 300
PHASE2_EPOCHS = 200
DEFAULT_BATCH = 32
DEFAULT_SEED = 42

#: Best-checkpoint selection criterion used for the paper runs.
DEFAULT_SELECTION_CRITERION = "bal_acc"

# Phase-1 classification / ordinal-regression losses (Table 1).
PHASE1_LOSSES = ["ce", "corn", "mae"]
# Phase-2 ranking losses compared in Table 1 (besides SpineRank).
RANKING_BASELINES = ["RankNet", "DeepRankSVM"]
HEAD_ABLATION = ["mlp", "linear", "kan", "transformer"]
THRESHOLD_METHODS = ["grid", "isotonic", "youden", "gmm"]

GROUPS = [
    "phase1",
    "paper",
    "ranking_baselines",
    "head_ablation",
    "threshold_ablation",
]

_BEST_CKPT = "__BEST_CKPT__"


@dataclasses.dataclass
class Experiment:
    """A single reproducible run: an optional train command, then eval.

    Attributes
    ----------
    group : str
        Experiment family (one of :data:`GROUPS`).
    name : str
        Unique experiment tag.
    train_cmd : list of str or None
        ``python -m ...`` argv for training (``None`` for eval-only runs).
    eval_cmd : list of str
        ``python -m ...`` argv for evaluation. ``__BEST_CKPT__`` is replaced
        by the resolved checkpoint at run time.
    exp_dir : Path
        Results directory holding ``weights/`` and ``eval*/``.
    seed : int
        Random seed for the run.
    backbone : str
        Encoder backbone identifier.
    """

    group: str
    name: str
    train_cmd: Optional[List[str]]
    eval_cmd: List[str]
    exp_dir: Path
    seed: int
    backbone: str


def _python() -> str:
    """Return the interpreter used to launch the trainer subprocesses."""
    return sys.executable or "python"


def _spinerank_flags(args: argparse.Namespace) -> List[str]:
    return [
        "--spinerank_margin_base", str(args.margin_base),
        "--spinerank_margin_scale", str(args.margin_scale),
        "--spinerank_c2", str(args.lambda_sim),
    ]


def _phase1_cmds(args: argparse.Namespace, clf_loss: str, exp_dir: Path):
    train = [
        _python(), "-m", TRAINER_MODULE,
        "--config", args.config,
        "--mode", "clf",
        "--backbone", args.backbone,
        "--clf_loss", clf_loss,
        "--selection_criterion", args.selection_criterion,
        "--epochs", str(args.phase1_epochs),
        "--batch_size", str(args.batch_size),
        "--oversample",
        "--resume", "auto",   # no-op on a fresh run
        "--save_dir", str(exp_dir / "weights"),
        "--eval_dir", str(exp_dir / "eval"),
        "--seed", str(args.seed),
    ]
    evaluate = [
        _python(), "-m", TRAINER_MODULE,
        "--config", args.config,
        "--mode", "eval",
        "--backbone", args.backbone,
        "--checkpoint", _BEST_CKPT,
        "--clf_loss", clf_loss,
        "--eval_dir", str(exp_dir / "eval"),
        "--seed", str(args.seed),
    ]
    evaluate.append("--no_tta" if args.no_tta else "--use_tta")
    return train, evaluate


def _ranking_train_cmd(args: argparse.Namespace, ranking_loss: str,
                       ranking_head: str, exp_dir: Path) -> List[str]:
    cmd = [
        _python(), "-m", TRAINER_MODULE,
        "--config", args.config,
        "--mode", "train",
        "--backbone", args.backbone,
        "--checkpoint", args.clf_checkpoint,
        "--encoder_only",
        "--rank_from_logits",
        "--ranking_loss", ranking_loss,
        "--ranking_head", ranking_head,
        "--clf_loss_weight", "0",
        "--ranking_loss_weight", "1",
        "--epochs", str(args.phase2_epochs),
        "--batch_size", str(args.batch_size),
        "--oversample",
        "--selection_criterion", args.selection_criterion,
        "--resume", "auto",   # no-op on a fresh run
        "--save_dir", str(exp_dir / "weights"),
        "--eval_dir", str(exp_dir / "eval"),
        "--seed", str(args.seed),
    ]
    if ranking_loss == "SpineRank":
        cmd += _spinerank_flags(args)
    return cmd


def _ranking_eval_cmd(args: argparse.Namespace, ranking_loss: str,
                      ranking_head: str, eval_dir: Path,
                      threshold_method: Optional[str] = None) -> List[str]:
    cmd = [
        _python(), "-m", TRAINER_MODULE,
        "--config", args.config,
        "--mode", "eval",
        "--backbone", args.backbone,
        "--checkpoint", _BEST_CKPT,
        "--rank_from_logits",
        "--ranking_loss", ranking_loss,
        "--ranking_head", ranking_head,
        "--eval_dir", str(eval_dir),
        "--seed", str(args.seed),
    ]
    if threshold_method:
        cmd += ["--threshold_method", threshold_method]
    cmd.append("--no_tta" if args.no_tta else "--use_tta")
    return cmd


def build_experiments(args: argparse.Namespace) -> List[Experiment]:
    """Enumerate every selected experiment as an :class:`Experiment`."""
    out: List[Experiment] = []
    root = Path(args.results_root)
    bb = args.backbone
    selected = set(args.group) if args.group else set(GROUPS)

    if "phase1" in selected:
        for loss in PHASE1_LOSSES:
            exp_dir = root / "phase1" / loss
            train, evaluate = _phase1_cmds(args, loss, exp_dir)
            out.append(Experiment("phase1", loss, train, evaluate,
                                  exp_dir, args.seed, bb))

    def add_ranking(group: str, loss: str, head: str) -> None:
        exp_dir = root / "ranking" / bb / f"{loss.lower()}_{head}"
        out.append(Experiment(
            group, f"{loss.lower()}_{head}",
            _ranking_train_cmd(args, loss, head, exp_dir),
            _ranking_eval_cmd(args, loss, head, exp_dir / "eval"),
            exp_dir, args.seed, bb,
        ))

    if "paper" in selected:
        add_ranking("paper", "SpineRank", "mlp")
    if "ranking_baselines" in selected:
        for loss in RANKING_BASELINES:
            add_ranking("ranking_baselines", loss, "mlp")
    if "head_ablation" in selected:
        for head in HEAD_ABLATION:
            if head == "mlp" and "paper" in selected:
                continue  # identical to the paper run
            add_ranking("head_ablation", "SpineRank", head)

    if "threshold_ablation" in selected:
        exp_dir = root / "ranking" / bb / "spinerank_mlp"
        for method in THRESHOLD_METHODS:
            out.append(Experiment(
                "threshold_ablation", f"spinerank_mlp[{method}]", None,
                _ranking_eval_cmd(args, "SpineRank", "mlp",
                                  exp_dir / f"eval_{method}",
                                  threshold_method=method),
                exp_dir, args.seed, bb,
            ))
    return out


def resolve_best_ckpt(exp: Experiment) -> Optional[Path]:
    """Return ``best.pt`` of a run, else its newest ``.pt`` checkpoint."""
    wdir = exp.exp_dir / "weights" / exp.backbone
    best = wdir / "best.pt"
    if best.is_file():
        return best
    candidates = sorted(wdir.glob("*.pt"), key=lambda p: p.stat().st_mtime,
                        reverse=True)
    return candidates[0] if candidates else None


def _fmt(cmd: List[str]) -> str:
    """Shell-quote an argv for display / copy-paste."""
    return " ".join(shlex.quote(c) for c in cmd)


def run_experiment(exp: Experiment, dry_run: bool) -> int:
    """Run (or print) one experiment's train + eval commands."""
    print(f"\n{'=' * 78}\n[{exp.group}] {exp.name}  (seed={exp.seed}, "
          f"backbone={exp.backbone})\n{'=' * 78}")
    if exp.train_cmd is not None:
        print("TRAIN:", _fmt(exp.train_cmd))
        if not dry_run:
            exp.exp_dir.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(exp.train_cmd, cwd=REPO_ROOT).returncode
            if rc != 0:
                print(f"  training failed (exit {rc})")
                return rc

    if dry_run:
        best_display = str(exp.exp_dir / "weights" / exp.backbone / "best.pt")
        eval_cmd = [best_display if c == _BEST_CKPT else c for c in exp.eval_cmd]
    else:
        best = resolve_best_ckpt(exp)
        if best is None:
            print("  no checkpoint found - skipping eval")
            return 1
        eval_cmd = [str(best) if c == _BEST_CKPT else c for c in exp.eval_cmd]

    print("EVAL: ", _fmt(eval_cmd))
    if not dry_run:
        rc = subprocess.run(eval_cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            print(f"  evaluation failed (exit {rc})")
            return rc
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the driver."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--group", nargs="+", choices=GROUPS, default=None,
                   help="Experiment families to include (default: all).")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Trainer YAML config (default: {DEFAULT_CONFIG}).")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   choices=["resnet18", "resnet34", "resnet50"],
                   help=f"Encoder backbone (default: {DEFAULT_BACKBONE}).")
    p.add_argument("--clf-checkpoint", default=None,
                   help="Phase-1 classification checkpoint that ranking runs "
                        "start from (default: <results-root>/phase1/ce/"
                        "weights/<backbone>/best.pt).")
    p.add_argument("--results-root", default="results",
                   help="Root directory for all run outputs (default: results).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED}).")
    p.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS,
                   help=f"Phase-1 epochs (default: {PHASE1_EPOCHS}).")
    p.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS,
                   help=f"Phase-2 ranking epochs (default: {PHASE2_EPOCHS}).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help=f"Batch size (default: {DEFAULT_BATCH}).")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation in eval.")
    p.add_argument("--margin-base", type=float, default=MARGIN_BASE,
                   help=f"SpineRank margin_base (default {MARGIN_BASE}).")
    p.add_argument("--margin-scale", type=float, default=MARGIN_SCALE,
                   help=f"SpineRank margin_scale (default {MARGIN_SCALE}).")
    p.add_argument("--lambda-sim", type=float, default=LAMBDA_SIM,
                   help=f"SpineRank lambda_sim / C2 (default {LAMBDA_SIM}).")
    p.add_argument("--selection-criterion", default=DEFAULT_SELECTION_CRITERION,
                   choices=["bal_acc", "qwk", "spearman",
                            "composite_rank", "roc_auc", "mcc"],
                   help="Best-checkpoint criterion "
                        f"(default {DEFAULT_SELECTION_CRITERION}).")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true",
                      help="List enumerated experiments and exit.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Print commands without executing (default).")
    mode.add_argument("--run", action="store_true",
                      help="Execute the train + eval commands locally.")
    args = p.parse_args(argv)

    if args.clf_checkpoint is None:
        args.clf_checkpoint = str(
            Path(args.results_root) / "phase1" / "ce" / "weights"
            / args.backbone / "best.pt"
        )
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Driver entry point. Returns a process exit code."""
    args = parse_args(argv)
    experiments = build_experiments(args)

    if args.list:
        print(f"Enumerated {len(experiments)} experiments "
              f"(results_root={args.results_root}):")
        current = None
        for exp in experiments:
            if exp.group != current:
                current = exp.group
                print(f"\n  [{current}]")
            print(f"    - {exp.name:28s} seed={exp.seed} backbone={exp.backbone}")
        return 0

    dry_run = not args.run
    if dry_run:
        print("# DRY RUN - printing commands only (use --run to execute)")
    elif any(e.group != "phase1" for e in experiments) and \
            not Path(args.clf_checkpoint).is_file() and \
            not (args.group is None or "phase1" in args.group):
        print(f"Phase-1 checkpoint not found: {args.clf_checkpoint}\n"
              "Run --group phase1 first or pass --clf-checkpoint.")
        return 1

    failures = 0
    for exp in experiments:
        rc = run_experiment(exp, dry_run=dry_run)
        failures += int(rc != 0)

    print(f"\n{'=' * 78}")
    print(f"Done: {len(experiments)} experiments, {failures} failure(s).")
    print(f"{'=' * 78}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
