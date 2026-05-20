# SCITAS Izar Submission Guide

This repository can be trained on EPFL SCITAS `izar`, which is the academic GPU cluster for student and course accounts.

Official references:

- Izar overview: https://scitas-doc.epfl.ch/supercomputers/izar/
- Slurm job submission: https://scitas-doc.epfl.ch/user-guide/using-clusters/running-jobs/
- Slurm QOS and limits: https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-qos-partitions/
- Python virtual environments: https://scitas-doc.epfl.ch/user-guide/software/python/python-venv/
- Storage guidance: https://scitas-doc.epfl.ch/user-guide/data-management/how-to-use-filesystems/

## 1. Current layout on Izar

This guide is aligned with your current setup:

```text
$HOME/DLAV_Trajectory_Planner
├── outputs
├── README.md
├── src
├── test_public
├── train
└── val
```

Environment:

```bash
source ~/venvs/mtr-izar/bin/activate
```

The provided Slurm scripts therefore assume:

```text
PROJECT_ROOT=$HOME/DLAV_Trajectory_Planner
DATA_ROOT=$HOME/projects/DALV/DLAV_Trajectory_Planner
VENV_DIR=$HOME/venvs/mtr-izar
OUTPUT_ROOT=$HOME/DLAV_Trajectory_Planner/outputs/egoframe_v10_depth_spatial
```

## 2. Create the environment

SCITAS recommends loading the same modules when creating and using the virtual environment.

```bash
ssh <username>@izar.hpc.epfl.ch
cd ~/DLAV_Trajectory_Planner

module purge
module load gcc python py-virtualenv
source ~/venvs/mtr-izar/bin/activate
pip install --no-cache-dir -r requirements.txt
```

If `mtr-izar` is already prepared and working, you do not need to create a new virtual environment.

## 3. Data layout

Your current data root is:

```text
$HOME/projects/DALV/DLAV_Trajectory_Planner
```

and the scripts expect:

```text
$HOME/projects/DALV/DLAV_Trajectory_Planner/train
$HOME/projects/DALV/DLAV_Trajectory_Planner/val
$HOME/projects/DALV/DLAV_Trajectory_Planner/test_public
```

## 4. Notes about pretrained ResNet

The default training command uses pretrained **ResNet-34** weights from torchvision. If the weights are not cached yet, PyTorch may try to download them.

On HPC, the safe options are:

1. Pre-cache the weights once before launching the batch job.
2. Or disable pretrained weights with `--no-pretrained`.

Example pre-cache command:

```bash
source ~/venvs/mtr-izar/bin/activate
python - <<'PY'
from torchvision.models import resnet34, ResNet34_Weights
resnet34(weights=ResNet34_Weights.DEFAULT)
print("ResNet-34 weights cached.")
PY
```

All commands in this repository are run directly via:

```bash
python -m src.train
python -m src.infer
```

## 5. Submit training

The repository provides a ready-to-use Slurm script:

```bash
mkdir -p logs
sbatch scripts/submit_izar_train.sbatch
```

Default resource request:

- partition: `gpu`
- qos: `gpu`
- GPUs: `1`
- CPUs per task: `8`
- memory: `32G`
- walltime: `48:00:00`

The script trains with segmentation, depth, and layer3 spatial pooling aux tasks enabled.

### Override directories without editing the script

```bash
PROJECT_ROOT=$HOME/DLAV_Trajectory_Planner \
DATA_ROOT=$HOME/projects/DALV/DLAV_Trajectory_Planner \
OUTPUT_ROOT=$HOME/DLAV_Trajectory_Planner/outputs/egoframe_v10_depth_spatial \
VENV_DIR=$HOME/venvs/mtr-izar \
sbatch scripts/submit_izar_train.sbatch
```

## 6. Submit inference and generate Kaggle CSV

After training:

```bash
mkdir -p logs
sbatch scripts/submit_izar_infer.sbatch
```

This script expects:

- checkpoint: `$HOME/DLAV_Trajectory_Planner/outputs/egoframe_v10_depth_spatial/best_checkpoint.pt`
- test data: `$HOME/projects/DALV/DLAV_Trajectory_Planner/test_public`

The submission file will be written to:

```text
$HOME/DLAV_Trajectory_Planner/outputs/egoframe_v10_depth_spatial/submission_phase2.csv
```

## 7. Debug and monitoring

Check the queue:

```bash
squeue -u $USER
```

Watch logs:

```bash
tail -f logs/dlav-p2-train-<jobid>.out
tail -f logs/dlav-p2-infer-<jobid>.out
```

For short tests, use the `debug` QOS with a short walltime. The official Izar limits allow `debug` jobs up to one hour.

Example sanity-check command:

```bash
sbatch --qos=debug --time=00:20:00 scripts/submit_izar_train.sbatch
```

## 8. Important platform-specific notes

- Do not run long training jobs on the login node.
- Use `srun` inside the batch script for the Python command.
- Your current setup keeps both code and outputs in `/home`. This is acceptable for this project size, but larger future runs should move heavy outputs to `/scratch`.
- Keep requested walltime and memory realistic; SCITAS explicitly warns that over-requesting resources increases queue time.