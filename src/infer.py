from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from egodrive.data import DrivingDataset, list_pickle_files
from egodrive.model import EgoDrivePlanner
from egodrive.utils import get_device


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference and generate a Kaggle submission CSV."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/phase1_resnet_gru/best_checkpoint.pt",
    )
    parser.add_argument("--test-dir", default="test_public")
    parser.add_argument("--output-csv", default="submission_phase1.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser


def load_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    image_size_override: int | None,
) -> tuple[EgoDrivePlanner, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    image_size = (
        image_size_override if image_size_override is not None else config["image_size"]
    )

    model = EgoDrivePlanner(
        backbone_name=config["backbone"],
        pretrained_backbone=False,
        image_feature_dim=config["image_feature_dim"],
        history_feature_dim=config["history_feature_dim"],
        command_feature_dim=config["command_feature_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        gru_layers=config["gru_layers"],
        dropout=config["dropout"],
        future_steps=config["future_steps"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, image_size


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device()
    model, image_size = load_model(
        args.checkpoint,
        device=device,
        image_size_override=args.image_size,
    )

    test_files = list_pickle_files(args.test_dir, args.max_test_samples)
    dataset = DrivingDataset(test_files, test=True, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions = []
    sample_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)

            future = model(camera, history, command)
            predictions.append(future[..., :2].cpu().numpy())
            sample_ids.extend(batch["sample_id"].tolist())

    all_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = all_predictions.reshape(all_predictions.shape[0], -1)

    column_names = ["id"]
    for step in range(1, all_predictions.shape[1] + 1):
        column_names.extend([f"x_{step}", f"y_{step}"])

    submission = pd.DataFrame(flat_predictions)
    submission.insert(0, "id", sample_ids)
    submission.columns = column_names
    submission.to_csv(args.output_csv, index=False)

    print(f"Saved submission to {args.output_csv}")
    print(f"Rows: {len(submission)}, columns: {len(submission.columns)}")


if __name__ == "__main__":
    main()
