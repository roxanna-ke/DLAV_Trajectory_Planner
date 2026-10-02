# DLAV Trajectory Planner

This repository contains the engineering version of the DLAV trajectory-prediction project. The primary pipeline is the final sim-to-real planner: a ResNet-34 camera encoder, history-conditioned `SpatialAttention`, and an autoregressive GRU decoder for a 60-step future trajectory.

The repository also preserves the original milestone notebooks, source files, SLURM jobs, and local training artifacts. The historical implementation is under `archive/original/`; the large local data and output archives are under `local_artifacts/` and are intentionally excluded from GitHub.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract `local_artifacts/DLAV_Trajectory_Planner_data.tar` into the repository root and run:

```bash
python scripts/train.py --config configs/final_sim2real.yaml
python scripts/evaluate.py \
  --config configs/final_sim2real.yaml \
  --checkpoint outputs/final_sim2real/best_checkpoint.pt
python scripts/infer.py \
  --config configs/final_sim2real.yaml \
  --checkpoint outputs/final_sim2real/best_checkpoint.pt \
  --output-csv outputs/final_sim2real/submission.csv
```

The generated submission uses the Kaggle schema:

```text
id, x_1, y_1, ..., x_60, y_60
```

## Repository structure

```text
.
├── configs/                 # final, command-conditioned, and auxiliary configs
├── egodrive/                # active project package
│   ├── data.py
│   ├── train.py / infer.py / evaluate.py
│   ├── metrics.py
│   └── models/
│       ├── planner.py       # EgoDrivePlanner
│       ├── fusion.py        # SpatialAttention
│       ├── decoders.py      # autoregressive and legacy decoders
│       └── auxiliary.py     # depth and semantic heads
├── scripts/                 # thin train/evaluate/infer wrappers
├── notebooks/archive/       # original course notebooks
├── archive/original/        # original source, scripts, README, and poster
├── local_artifacts/         # local-only tar archives; not pushed
├── requirements.txt
└── README.md
```

## Model

The final model uses an ego-frame GRU history encoder and a ResNet-34 visual trunk. `SpatialAttention` uses the history representation as a query over the visual feature map. The attended visual feature and history feature are fused with an MLP. An autoregressive `GRUCell` predicts one step at a time; each output is fed into the next step and xy displacements are cumulatively summed.

The auxiliary configuration retains the milestone-2 depth and semantic segmentation heads. The command-conditioned configuration retains the milestone-1 command embedding and FiLM conditioning. Alternative fusion and decoder implementations remain available in the archived source snapshot.

## Local artifacts

The following files are intentionally local-only and are ignored by Git:

```text
local_artifacts/DLAV_Trajectory_Planner_data.tar
local_artifacts/DLAV_Trajectory_Planner_outputs.tar
local_artifacts/DLAV_Trajectory_Planner_misc.tar
```

The archives contain the original `train`, `val`, `val_real`, `test_public`, `test_public_real`, `outputs`, `logs`, and miscellaneous local files. To restore the data locally, extract the archives from the repository root.
