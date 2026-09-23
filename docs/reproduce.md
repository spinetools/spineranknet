# Reproducing the paper

Details behind the short [README](../README.md): data layout, the individual training commands, the trainer flags, the outputs and where the code differs from the paper.

## Installation details

The tested environment in [`requirements.txt`](../requirements.txt) is Python 3.11, PyTorch 2.5.0 and NumPy 1.26.4. `pip install -r requirements.txt` installs the runtime dependencies without the package. The wheel carries the importable package only; take [`config/`](../config) and [`scripts/`](../scripts) from a clone or the source distribution.

| Extra | Adds | Needed for |
|---|---|---|
| `corn` | `coral-pytorch` | CORN baseline (`--clf_loss corn`) |
| `dicom` | `pydicom`, `scikit-image` | DICOM / polygon helpers in `spineranknet.dataloaders.utils` |
| `extras` | `tensorboard`, `tabulate` | TensorBoard logging, console tables in the ablation script |
| `dev` | `ruff`, `build` | Linting and packaging |
| `all` | `corn` + `dicom` + `extras` | Everything above |

## Data

- IVD volumes were extracted with SpineNetV2 vertebra localisation, disc-centred rotation and resampling, and split by patient into train / validation / test (80 / 10 / 10 %), stratified by centre and grade distribution.
- Each 192 × 320 × 15 volume is cropped to the 128 × 256 × 12 network input at load time. Training samples one of the available series (T1w or T2w) at random; the test set is restricted to T2w (`T2_S1`) unless you pass `--no_test_filter`.
- `genodisc_grading.json` in the data root is read only by `scripts/eda_dataset.py`.

The paths in `config/*.yaml` start with `data/GENODISCv2/`, relative to the working directory. To use a copy stored elsewhere, use any of:

```bash
export GENODISC_ROOT=/path/to/GENODISCv2                          # replaces the data/GENODISCv2 prefix
python -m spineranknet.baseline.train_ranking_mse ... --data_root /path/to/GENODISCv2
python -m spineranknet.baseline.train_ranking_mse ... \
    --genodict_path /path/to/geno_ivd_v1.pkl --ivd_path /path/to/IVDs-numpy/   # IVD_PATH ends with "/"
```

Path fields in a YAML config (`GENODICT_PATH`, `IVD_PATH`, `SAVE_DIR`, `EVAL_DIR`, …) may refer to environment variables, e.g. `IVD_PATH: ${GENODISC_ROOT}/IVDs-numpy/`. A leading `~` is expanded too, and an unset or empty variable is an error unless the matching command-line flag overrides the field. `GENODISC_ROOT` and `--data_root` replace only the `data/GENODISCv2` prefix; absolute paths are used as written. `data/`, `results/`, `.cache/` and all checkpoint, array and pickle files are git-ignored.

## Grading tasks

The paper reports 11 tasks per IVD level; grade indices start at 0 (Pfirrmann I–V is 0–4). Names follow `TASK_DEFINITIONS` in [`spineranknet/config.py`](../spineranknet/config.py).

| Task | Config name | Grades |
|---|---|---|
| Pfirrmann grade | `Pfirrmann` | I, II, III, IV, V |
| Disc narrowing | `Narrowing` | None, Mild, Moderate, Severe |
| Central canal stenosis | `CentralCanalStenosis` | None, Mild, Moderate, Severe |
| Spondylolisthesis | `Spondylolisthesis` | Normal, Moderate, Severe |
| Upper / lower endplate defect | `UpperEndplateDefect`, `LowerEndplateDefect` | Normal, Slight, Moderate, Severe |
| Foraminal stenosis (left / right) | `ForaminalStenosisLeft`, `ForaminalStenosisRight` | Absent, Mild, Moderate, Severe |
| Herniation | `Herniation` | Normal, Slight, Moderate, Large |
| Anterior / posterior bulging | `AnteriorBulging`, `PosteriorBulging` | Normal, Slight, Moderate, Severe |

[`config/ranking_logit_heads.yaml`](../config/ranking_logit_heads.yaml) also trains `UpperModic` / `LowerModic` heads and an `IVDlevel` head (no ranking head); they are not among the 11 reported tasks.

## Training and evaluation

Everything goes through `python -m spineranknet.baseline.train_ranking_mse` (installed as `spineranknet-train`) with `--mode clf` (classification only), `train` (classification + ranking; Phase 2 with `--clf_loss_weight 0`), `rank` (ranking only) or `eval`.

### Driver

[`scripts/run_all_experiments.py`](../scripts/run_all_experiments.py) prints the commands by default and runs them with `--run`. It passes `--resume auto`, so an interrupted run continues from its latest checkpoint.

| Group | Runs | Output directory |
|---|---|---|
| `phase1` | CE, CORN and MAE<sub>ℓ</sub> baselines; the CE run is the Phase-1 checkpoint for all ranking runs | `results/phase1/<loss>/` |
| `paper` | SpineRankNet: SpineRank loss, MLP head on the logits | `results/ranking/resnet18/spinerank_mlp/` |
| `ranking_baselines` | RankNet and DeepRankSVM | `results/ranking/resnet18/{ranknet,deepranksvm}_mlp/` |
| `head_ablation` | `linear`, `kan` and `transformer` heads | `results/ranking/resnet18/spinerank_<head>/` |
| `threshold_ablation` | `grid`, `isotonic`, `youden` and `gmm` thresholds | `results/ranking/resnet18/spinerank_mlp/eval_<method>/` |

Options: `--results-root` (default `results`), `--clf-checkpoint`, `--seed` (42), `--phase1-epochs` (300), `--phase2-epochs` (200), `--batch-size` (32), `--no-tta`.

### Manual commands

Phase 1, classification pre-training with weighted cross-entropy:

```bash
python -m spineranknet.baseline.train_ranking_mse \
    --config config/ranking_logit_heads.yaml --mode clf --backbone resnet18 \
    --clf_loss ce --epochs 300 --batch_size 32 --oversample \
    --selection_criterion bal_acc --seed 42 \
    --save_dir results/phase1/ce/weights --eval_dir results/phase1/ce/eval
```

Phase 2, SpineRankNet ranking heads on the Phase-1 logits:

```bash
python -m spineranknet.baseline.train_ranking_mse \
    --config config/ranking_logit_heads.yaml --mode train --backbone resnet18 \
    --checkpoint results/phase1/ce/weights/resnet18/best.pt --encoder_only \
    --rank_from_logits --ranking_loss SpineRank --ranking_head mlp \
    --clf_loss_weight 0 --ranking_loss_weight 1 \
    --spinerank_margin_base 0.5 --spinerank_margin_scale 1.5 --spinerank_c2 0.25 \
    --epochs 200 --batch_size 32 --oversample --selection_criterion bal_acc --seed 42 \
    --save_dir results/ranking/resnet18/spinerank_mlp/weights \
    --eval_dir results/ranking/resnet18/spinerank_mlp/eval
```

Evaluation on the test set (54-view TTA on by default; thresholds fitted on the validation set):

```bash
python -m spineranknet.baseline.train_ranking_mse \
    --config config/ranking_logit_heads.yaml --mode eval --backbone resnet18 \
    --checkpoint results/ranking/resnet18/spinerank_mlp/weights/resnet18/best.pt \
    --rank_from_logits --ranking_loss SpineRank --ranking_head mlp \
    --threshold_method grid --seed 42 \
    --eval_dir results/ranking/resnet18/spinerank_mlp/eval
```

Threshold methods (`--threshold_method`), with the paper's BA / QWK: `grid` (default; coarse 200-point sweep, two refinements, Nelder–Mead polish) 63.3 % / 0.762, `isotonic` 55.7 % / 0.744, `youden` 58.8 % / 0.685, `gmm` 54.3 % / 0.448.

### Outputs

- Checkpoints: `<save_dir>/<backbone>/best.pt`, with the effective `train_config.yaml` beside it (`--mode eval` without `--config` loads it automatically).
- Test metrics: `<eval_dir>/<backbone>/evaluation_metrics.json`, `hybrid_metrics.json` (read by `scripts/generate_ranking_loss_table.py`), per-task prediction CSVs (read by `scripts/aggregate_spineranknet_ablation.py`) and plots.
- **Privacy.** Prediction CSVs, `viz_cache` files and `train_config.yaml` contain Genodisc sample IDs or local paths: keep them private. Figures label cases `#1`, `#2`, … unless you pass `--show_ids`; never publish figures made with `--show_ids`.

### Key flags

Command-line flags override the YAML config; `--help` lists them all.

| Flag | Meaning |
|---|---|
| `--config`, `--mode` | YAML config (`config/ranking_logit_heads.yaml` for the paper); `clf`, `train`, `rank` or `eval` |
| `--backbone` | `resnet18` (paper), `resnet34`, `resnet50` |
| `--checkpoint`, `--encoder_only` | Start from a checkpoint; with `--rank_from_logits`, `--encoder_only` loads the encoder and the classification heads |
| `--resume` | Resume from a full training state (`.pt` path or `auto`) |
| `--rank_from_logits` | Ranking heads read each task's classification logits (SpineRankNet) |
| `--ranking_loss`, `--ranking_head` | `SpineRank`, `RankNet`, `DeepRankSVM`, …; `mlp` (paper), `linear`, `kan`, `transformer` |
| `--clf_loss` | Phase-1 loss: `ce`, `corn`, `mae`, … |
| `--clf_loss_weight`, `--ranking_loss_weight` | Weights of the two loss families (Phase 2: 0 and 1) |
| `--spinerank_margin_base`, `--spinerank_margin_scale`, `--spinerank_c2` | SpineRank m<sub>base</sub> (0.5), m<sub>scale</sub> (1.5) and λ<sub>sim</sub> (0.25 in the paper) |
| `--spinerank_no_adaptive_margin`, `--spinerank_no_severity_weight`, `--spinerank_no_similarity` | Component ablations |
| `--epochs`, `--batch_size`, `--lr`, `--encoder_lr`, `--patience` | Schedule; `--lr` is the heads' learning rate. `--patience 0` is treated as 30 |
| `--selection_criterion`, `--oversample` | Checkpoint selection metric (`bal_acc` in the paper); oversampling of rare label combinations |
| `--threshold_method`, `--threshold_metric` | `grid` (default), `isotonic`, `youden`, `gmm`; the metric the grid search optimises (default composite 0.5·BA + 0.3·QWK + 0.2·(1 − MAE/(K−1))) |
| `--use_tta` / `--no_tta` | 54-view test-time augmentation (2 flips × 3 × 3 shifts × 3 slice shifts), on by default |
| `--save_dir`, `--eval_dir`, `--seed` | Output directories for checkpoints and evaluation; random seed |
| `--data_root`, `--genodict_path`, `--ivd_path` | Data location (see [Data](#data)) |
| `--show_ids` | Real sample IDs in figure titles instead of `#1`, `#2`, …; local inspection only |

### Tables, figures and EDA

```bash
python scripts/generate_ranking_loss_table.py --results_dir results/ranking --backbone resnet18
python scripts/aggregate_spineranknet_ablation.py --results_dir results/ranking --backbone resnet18
python scripts/plot_loss_intuition.py --out results/figures --no-gif
python scripts/eda_dataset.py --out-dir results/eda     # reads $GENODISC_ROOT; no GPU needed
```

`scripts/explore_results.py` summarises `hybrid_metrics.json` files from `seed_<N>/` run folders. `scripts/eda_dataset.py` also reads `genodisc-pathologies-demographics.csv` and `genodisc.csv` from the data root. The `severity_grid_manifest.csv` it writes contains subject identifiers and scan dates: keep it private.

## Differences between the paper and the code

- Table 1 comes from a single run (seed 42) per method; expect small differences across GPUs, drivers and library versions.
- The paper writes the ranking head as acting on the embedding z (Eq. 5); the released model follows Fig. 1 (`--rank_from_logits`).
- AdamW uses two learning-rate groups in every mode, including Phase 1: `ENCODER_LR` 1e-4 and `LR` 1e-3. The paper gives lr 1e-3 for Phase 1 (`--encoder_lr 1e-3`).
- The config sets `WEIGHT_DECAY: 1.0e-5`; the paper lists 1e-4.
- The driver passes `--oversample` in both phases; the paper does not describe it.
- The paper's minimum of 50 epochs before early stopping is not enforced; early stopping can trigger from epoch 31.
- The ranking loss is averaged over all ranking heads (the 11 tasks plus `UpperModic` / `LowerModic`) instead of summed, which only rescales it.
- The learning rate follows cosine annealing (eta_min 1e-6), which the paper does not mention.
- `train_config.yaml` records local data and output paths: remove them before sharing a run folder or weights.
