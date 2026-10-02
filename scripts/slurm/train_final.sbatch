#!/bin/bash
#SBATCH --job-name=egodrive-final
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/final_sim2real.yaml}"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
python scripts/train.py --config "$CONFIG"
