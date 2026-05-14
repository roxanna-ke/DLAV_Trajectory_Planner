#!/bin/bash
#SBATCH --job-name=dlav-gate-sp-noaux
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/DLAV/DLAV_Trajectory_Planner}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/ablation_gated_spatial_noaux}"
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
  --output-dir "$OUTPUT_ROOT" \
  --use-layer3-spatial-pooling \
  --layer3-spatial-grid-height 4 \
  --layer3-spatial-grid-width 6 \
  --image-height 224 \
  --image-width 336 \
  --batch-size 64 \
  --num-workers 4 \
  --epochs 50 \
  --lr 1e-3 \
  --backbone-lr 5e-5 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --ade-weight 1.0 \
  --fde-weight 0.2 \
  --heading-weight 0.1 \
  --time-weight-start 1.0 \
  --time-weight-end 2.0 \
  --seed 42
