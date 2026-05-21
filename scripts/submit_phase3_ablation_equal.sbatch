#!/bin/bash
#SBATCH --job-name=dlav-p3-equal
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/phase3_ablation_equal}"
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
  --train-dir "$DATA_ROOT/val_real" \
  --val-dir "$DATA_ROOT/val_real" \
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
  --camera-feature-weight 1.0 \
  --history-feature-weight 1.0 \
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
