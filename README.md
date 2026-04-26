# EgoDrive Milestone 1

Milestone 1 implementation for the DLAV 2026 final project. This repository uses the official starter dataset layout downloaded by the course notebook:

- `train/` with 5000 labeled samples
- `val/` with 1000 labeled samples
- `test_public/` with 1000 unlabeled samples

The model follows the Milestone 1 input constraints and only uses:

- `camera`
- `driving_command`
- `sdc_history_feature`

`depth` and `semantic_label` are ignored.

## Repository layout

```text
.
├── egodrive/
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── notebooks/
│   └── DLAV_Phase1.ipynb
├── train.py
├── infer.py
├── requirements.txt
└── README.md
```

## Model

The planner uses:

- a pretrained `ResNet` backbone to encode the camera image
- an embedding for `driving_command`
- a linear projection for each history step
- a `GRU` that consumes the fused `camera + command + history` sequence
- an MLP head that predicts the future `60 x 3` trajectory

At inference time the Kaggle submission keeps only the predicted `x, y` values.

## Environment

Install the core dependencies in your environment:

```bash
pip install -r requirements.txt
```

If you want pretrained ImageNet weights and they are not cached yet, PyTorch will download them the first time you train without `--no-pretrained`.

## Data

Use the data downloaded by the official notebook, so that the repository contains:

```text
train/
val/
test_public/
```

The older `dlav-2026-phase-1/` Kaggle subset is not used by the scripts.

## Train

Example training command:

```bash
python train.py \
  --train-dir train \
  --val-dir val \
  --output-dir outputs/phase1_resnet_gru \
  --backbone resnet18 \
  --epochs 15 \
  --batch-size 32
```

Useful options:

- `--freeze-backbone` freezes the ResNet encoder
- `--no-pretrained` disables ImageNet initialization
- `--max-train-samples` and `--max-val-samples` help with quick sanity checks

The best checkpoint is saved to:

```text
outputs/phase1_resnet_gru/best_checkpoint.pt
```

## Kaggle submission

Generate the CSV for leaderboard upload with:

```bash
python infer.py \
  --checkpoint outputs/phase1_resnet_gru/best_checkpoint.pt \
  --test-dir test_public \
  --output-csv submission_phase1.csv
```

The generated file follows the required format:

```text
id, x_1, y_1, x_2, y_2, ..., x_60, y_60
```

## Notes

- The notebook in `notebooks/` is kept as the original course reference.
- The script entry points are now `train.py` and `infer.py`.
- Validation is reported with ADE and FDE, while the training loss uses Smooth L1 on the trajectory with a smaller heading weight.
