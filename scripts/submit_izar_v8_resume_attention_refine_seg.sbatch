#!/bin/bash
#SBATCH --job-name=dlav-v8-attn-seg
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/v8_resume_attention_refine_seg}"
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
  --resume "$PROJECT_ROOT/outputs/egoframe_v8/best_checkpoint.pt:weights-only" \
  --reset-trajectory-head-on-resume \
  --use-segmentation-head \
  --image-height 224 \
  --image-width 336 \
  --batch-size 64 \
  --num-workers 4 \
  --epochs 50 \
  --lr 5e-4 \
  --backbone-lr 2e-5 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --heading-weight 0.1 \
  --fde-weight 0.15 \
  --segmentation-loss-weight 0.1 \
  --aux-warmup-epochs 10 \
  --aux-ramp-epochs 15 \
  --time-weight-start 1.0 \
  --time-weight-end 2.0 \
  --seed 42 \
  --submission-csv-name submission_best_checkpoint.csv
