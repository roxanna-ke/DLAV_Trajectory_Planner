#!/bin/bash
#SBATCH --job-name=dlav-p3-spattn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/DLAV/DLAV_Trajectory_Planner}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/phase3_spatial_attention_seed42}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"

mkdir -p "$OUTPUT_ROOT" "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

module purge
module load gcc python
source "$VENV_DIR/bin/activate"

CACHE_ROOT="${SCRATCH:-$HOME/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

srun python -m src.train \
  --train-dir "$DATA_ROOT/train" \
  --real-dir "$DATA_ROOT/val_real" \
  --real-val-fraction 0.5 \
  --real-oversample 1 \
  --style-stat-samples 200 \
  --test-dir "$DATA_ROOT/test_public_real" \
  --output-dir "$OUTPUT_ROOT" \
  --image-height 224 \
  --image-width 336 \
  --image-feature-dim 256 \
  --history-hidden-dim 128 \
  --history-layers 2 \
  --fusion-dim 256 \
  --fusion-heads 4 \
  --dropout 0.1 \
  --future-steps 60 \
  --batch-size 32 \
  --num-workers 2 \
  --epochs 80 \
  --lr 1e-3 \
  --backbone-lr 5e-5 \
  --weight-decay 1e-4 \
  --heading-weight 0.1 \
  --fde-weight 1.0 \
  --time-weight-start 1.0 \
  --time-weight-end 2.0 \
  --seed 42 \
  --submission-csv-name submission_best_checkpoint.csv
