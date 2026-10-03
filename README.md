# EgoDrive

EgoDrive is an end-to-end trajectory-planning project for the `Deep Learning for Autonomous Vehicles` course. It predicts a 60-step future trajectory from a front-view camera image and recent ego-motion history. The repository presents the final sim-to-real planner as the primary workflow while retaining reusable command-conditioned, auxiliary-task, fusion, and decoder implementations in the active codebase.

The final model uses a ResNet-34 visual encoder, history-conditioned spatial attention for visual-history fusion, and an autoregressive `GRUCell` decoder. Sim-to-real generalization is supported through AdaIN style transfer, synthetic-plus-real data mixing, and real-domain validation/fine-tuning workflows.

## Repository structure

```text
.
├── configs/                 # YAML experiment configurations
│   ├── final_sim2real.yaml
│   ├── milestone1_command.yaml
│   └── milestone2_auxiliary.yaml
├── scripts/                 # command-line entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── slurm/train_final.sbatch
├── egodrive/                # active project implementation
│   ├── data.py              # datasets, ego-frame transforms, and AdaIN
│   ├── train.py / infer.py / evaluate.py
│   ├── metrics.py           # trajectory and auxiliary losses/metrics
│   ├── config.py / factory.py / utils.py
│   └── models/
│       ├── planner.py       # EgoDrivePlanner
│       ├── fusion.py        # SpatialAttention
│       ├── decoders.py      # autoregressive and direct decoders
│       └── auxiliary.py     # depth and semantic heads
├── local_artifacts/         # local-only data/output tar archives; not pushed
├── requirements.txt
└── README.md
```

Historical `archive/` and `notebooks/` directories are not part of the current GitHub version. The active repository is intentionally focused on the reusable project implementation.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository's local data and generated outputs are stored outside Git in tar archives. To restore the local working environment, run these commands from the repository root:

```bash
tar -xf local_artifacts/DLAV_Trajectory_Planner_data.tar
tar -xf local_artifacts/DLAV_Trajectory_Planner_outputs.tar
```

The configurations use paths relative to the repository root:

- `final_sim2real.yaml`: `./train`, `./val_real`, and `./test_public_real`
- `milestone1_command.yaml`: `./train`, `./val`, and `./test_public`
- `milestone2_auxiliary.yaml`: `./train`, `./val`, and `./test_public`

Train the final model:

```bash
python scripts/train.py --config configs/final_sim2real.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/final_sim2real.yaml \
  --checkpoint outputs/final_sim2real/best_checkpoint.pt
```

Generate a Kaggle submission CSV:

```bash
python scripts/infer.py \
  --config configs/final_sim2real.yaml \
  --checkpoint outputs/final_sim2real/best_checkpoint.pt \
  --output-csv outputs/final_sim2real/submission.csv
```

The submission schema is:

```text
id, x_1, y_1, x_2, y_2, ..., x_60, y_60
```

## Model architecture

The final planner is implemented in `egodrive/models/planner.py`.

- **Visual encoder.** A ResNet-34 convolutional trunk produces a spatial visual feature map. The visual encoder itself does not perform history attention.
- **History encoder.** The recent poses are transformed into the current ego frame and encoded by a 2-layer GRU.
- **Fusion.** `SpatialAttention` in `egodrive/models/fusion.py` uses the history hidden state as a query over the ResNet feature map. The attended visual feature and history feature are then combined by an MLP.
- **Decoder.** `AutoregressiveGRUDecoder` uses a `GRUCell` to roll out one four-dimensional prediction at a time. Each predicted step is fed into the next step; xy deltas are cumulatively integrated into the predicted trajectory.

The final configuration does not use the driving command, depth labels, or semantic labels. The command-conditioned configuration adds command embedding and FiLM conditioning, while the auxiliary configuration adds depth and 15-class semantic segmentation heads. `DirectResidualDecoder` is retained as an alternative non-autoregressive decoder.

## Data representation

A labelled sample is a pickle dictionary with fields such as:

```python
{
    "camera": ndarray(H, W, 3),
    "sdc_history_feature": ndarray(21, 3),   # x, y, heading
    "sdc_future_feature": ndarray(60, 3),     # labelled splits only
}
```

The dataset converts the pose history into ego-frame features:

```text
[ego_x, ego_y, sin(Δheading), cos(Δheading)]
```

Therefore, the active history encoder input dimension is **4**. It is not 6 and does not include `vel_x` or `vel_y`.

Milestone 1 and 2 samples may additionally contain `driving_command`. Milestone-2 samples provide depth and semantic labels. Real final-data samples may omit these fields.

## Experiments and configurations

| Configuration | Inputs | Main feature |
| --- | --- | --- |
| `configs/milestone1_command.yaml` | camera, command, history | command embedding + FiLM conditioning |
| `configs/milestone2_auxiliary.yaml` | camera, command, history, depth, semantic labels | auxiliary depth/segmentation heads |
| `configs/final_sim2real.yaml` | camera, history | history-conditioned spatial attention + AdaIN + sim/real mixing |

The final sim-to-real configuration removes command and auxiliary labels. It mixes synthetic training data with a real-data split and can apply AdaIN using real-image statistics.

## Results

The following values are recorded project results. The first two are local validation ADE values; the last is a Kaggle public leaderboard score.

| Stage | Score |
| --- | ---: |
| Stage 1: 80 epochs, sim/real mix + AdaIN | 1.6263 |
| Stage 2: 20-epoch fine-tune | 1.6264 |
| Kaggle public leaderboard | 1.4788 |

## Key hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| Image size | 224 × 336 | Resized camera input |
| Batch size | 32 | Training batch size |
| Stage 1 epochs | 80 | Final sim-to-real training stage |
| Stage 2 epochs | 20 | Fine-tuning stage |
| Weight decay | 1e-4 | AdamW regularization |
| Gradient clip | 5.0 | Maximum gradient norm |
| AdaIN α | 0.5 | Style blending strength |
| Real train samples | 500 | Mixed from `val_real` |
| History input dim | 4 | `[ego_x, ego_y, sin(Δheading), cos(Δheading)]` |
| History hidden dim | 128 | GRU hidden size |
| Image feature dim | 256 | Attention output and fusion context size |
| Future steps | 60 | Autoregressive rollout length |

## Reproducibility

For each experiment, record the YAML configuration, checkpoint, random seed, source commit, and data split. Large datasets and generated outputs remain in the local-only `local_artifacts/` directory and are excluded from GitHub.
