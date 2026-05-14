#!/bin/bash
#SBATCH --job-name=dlav-v8-s42
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/DLAV/DLAV_Trajectory_Planner}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/v8_seed42}"
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
  --val-dir "$DATA_ROOT/val" \
  --test-dir "$DATA_ROOT/test_public" \
  --output-dir "$OUTPUT_ROOT" \
  --use-segmentation-head \
  --use-depth-head \
  --image-height 224 \
  --image-width 336 \
  --image-feature-dim 256 \
  --history-hidden-dim 128 \
  --command-feature-dim 32 \
  --history-layers 2 \
  --dropout 0.1 \
  --batch-size 32 \
  --num-workers 4 \
  --epochs 50 \
  --lr 5e-4 \
  --backbone-lr 3e-5 \
  --weight-decay 1e-4 \
  --heading-weight 0.05 \
  --fde-weight 0.1 \
  --depth-loss-weight 0.1 \
  --segmentation-loss-weight 0.5 \
  --aux-warmup-epochs 5 \
  --aux-ramp-epochs 10 \
  --time-weight-start 1.0 \
  --time-weight-end 1.5 \
  --seed 42 \
  --submission-csv-name submission_best_checkpoint.csv
