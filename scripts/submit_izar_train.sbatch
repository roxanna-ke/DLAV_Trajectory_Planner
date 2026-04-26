#!/bin/bash
#SBATCH --job-name=dlav-m1-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/DLAV/DLAV_Trajectory_Planner}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/phase1_resnet_gru}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"

mkdir -p "$OUTPUT_ROOT"

cd "$PROJECT_ROOT"

module purge
module load gcc python py-virtualenv
source "$VENV_DIR/bin/activate"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

srun python -m src.train \
  --train-dir "$DATA_ROOT/train" \
  --val-dir "$DATA_ROOT/val" \
  --output-dir "$OUTPUT_ROOT" \
  --backbone resnet18 \
  --batch-size 32 \
  --epochs 15 \
  --num-workers 8 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --heading-weight 0.05 \
  --seed 42 \
  --no-pretrained
