#!/bin/bash
#SBATCH --job-name=dlav-m1-infer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/DLAV/DLAV_Trajectory_Planner}"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/DLAV/DLAV_Trajectory_Planner}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/phase1_resnet_gru}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_ROOT/best_checkpoint.pt}"
SUBMISSION_PATH="${SUBMISSION_PATH:-$OUTPUT_ROOT/submission_phase1.csv}"

mkdir -p "$(dirname "$SUBMISSION_PATH")"

cd "$PROJECT_ROOT"

module purge
module load gcc python py-virtualenv
source "$VENV_DIR/bin/activate"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

srun python -m src.infer \
  --checkpoint "$CHECKPOINT_PATH" \
  --test-dir "$DATA_ROOT/test_public" \
  --output-csv "$SUBMISSION_PATH" \
  --batch-size 64 \
  --num-workers 4
