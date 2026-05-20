# DLAV Trajectory Planner

Phase 2 / Phase 3 codebase for the DLAV 2026 final project.

## Scope

This repository contains:

- the PyTorch training and inference code used for our later Phase 2 experiments
- saved experiment metadata inside ignored `outputs/` folders
- helper scripts for SCITAS / Izar runs
- an ensemble utility reproducing the final CSV averaging step

Some cluster launcher scripts in `scripts/` are historical convenience files. For the final reported Phase 2 runs, prefer the commands documented in this README and the hyperparameters saved inside each checkpoint `config`.

This repository does not claim full end-to-end reproducibility of the exact best leaderboard model history. The final submission came from resumed finetuning runs, and the exact older code snapshot that originally produced the parent checkpoint is no longer recoverable from git.

## Dataset layout

The official starter notebooks expect:

```text
train/         5000 labeled synthetic samples
val/           1000 labeled synthetic samples
test_public/   1000 public test samples
```

For Phase 3, the real-domain notebook additionally uses:

```text
val_real/
test_public_real/
```

These folders are ignored by git and should not be committed.

## Model

The planner uses:

- a pretrained ResNet-34 image backbone
- a 2-layer GRU history encoder
- an MLP fusion block for vision and history features
- an autoregressive GRUCell decoder predicting 60 future steps

At submission time only the predicted `x, y` trajectory coordinates are written to CSV.

## Best Phase 2 result

The best Milestone 2 leaderboard submission was an ensemble, not a single checkpoint.

- Final CSV: `outputs/v8_adeloss_3seed_average/submission_mean_4.csv`
- Ensemble members:
  - `outputs/v8_seed42_adeloss/submission_best_checkpoint.csv`
  - `outputs/v8_seed42_adeloss_4/submission_best_checkpoint.csv`
  - `outputs/v8_seed123_adeloss_5/submission_best_checkpoint.csv`
  - `outputs/v8_seed3407_adeloss_3/submission_best_checkpoint.csv`

All four runs were finetuned from:

```text
outputs/aux_from_baseline/best_checkpoint.pt:weights-only
```

with `--reset-trajectory-head-on-resume`.

## Reproducibility note

What is still reproducible from this repository:

- the current training code in `src/`
- the exact hyperparameters saved inside the released finetune checkpoints
- the exact inference procedure
- the exact ensemble averaging step

What is not fully reproducible anymore:

- the exact older code version that originally produced `outputs/aux_from_baseline/best_checkpoint.pt`

For submission, the correct thing to do is to state this explicitly. It is better to be honest about partial reproducibility than to provide a misleading “from scratch” claim.

## Environment

```bash
pip install -r requirements.txt
```

## Training

Example training command for Phase 3 (real-domain data, no depth/segmentation/command):

```bash
python -m src.train \
  --train-dir val_real \
  --val-dir val_real \
  --test-dir test_public_real \
  --output-dir outputs/v8_phase3 \
  --image-height 224 \
  --image-width 336 \
  --image-feature-dim 256 \
  --history-hidden-dim 128 \
  --history-layers 2 \
  --fusion-dim 256 \
  --fusion-heads 4 \
  --dropout 0.1 \
  --batch-size 16 \
  --num-workers 4 \
  --epochs 50 \
  --lr 9e-5 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --heading-weight 0.05 \
  --fde-weight 0.15 \
  --ade-weight 1.23 \
  --time-weight-start 1.0 \
  --time-weight-end 1.4 \
  --seed 42 \
  --submission-csv-name submission_best_checkpoint.csv
```

## Inference

```bash
python -m src.infer \
  --checkpoint outputs/v8_phase3/best_checkpoint.pt \
  --test-dir test_public_real \
  --output-csv outputs/v8_phase3/submission_best_checkpoint.csv
```

## Ensemble

To reproduce the final Phase 2 averaging step:

```bash
python scripts/make_phase2_ensemble.py \
  outputs/v8_seed42_adeloss/submission_best_checkpoint.csv \
  outputs/v8_seed42_adeloss_4/submission_best_checkpoint.csv \
  outputs/v8_seed123_adeloss_5/submission_best_checkpoint.csv \
  outputs/v8_seed3407_adeloss_3/submission_best_checkpoint.csv \
  --output outputs/v8_adeloss_3seed_average/submission_mean_4.csv
```

