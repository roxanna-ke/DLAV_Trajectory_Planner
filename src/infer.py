from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DrivingDataset, decode_xy_from_ego, list_pickle_files
from src.model import EgoDrivePlanner
from src.utils import ensure_dir, get_device, rename_legacy_checkpoint_keys


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference and generate a Kaggle submission CSV."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/hybrid_resnet34_local_fusion/best_checkpoint.pt",
    )
    parser.add_argument("--test-dir", default="test_public")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path. Defaults to a submission CSV next to the checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser


def load_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    image_size_override: int | None,
    image_height_override: int | None,
    image_width_override: int | None,
) -> tuple[EgoDrivePlanner, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    if image_size_override is not None:
        image_height = image_size_override
        image_width = image_size_override
    else:
        image_height = image_height_override
        image_width = image_width_override
        if image_height is None:
            image_height = config.get("image_height", config.get("image_size", 224))
        if image_width is None:
            image_width = config.get("image_width", config.get("image_size", 224))

    model = EgoDrivePlanner(
        pretrained_backbone=False,
        image_feature_dim=config["image_feature_dim"],
        history_hidden_dim=config["history_hidden_dim"],
        command_feature_dim=config["command_feature_dim"],
        history_layers=config["history_layers"],
        fusion_dim=config["fusion_dim"],
        fusion_heads=config["fusion_heads"],
        dropout=config["dropout"],
        future_steps=config["future_steps"],
    )
    renamed_state = rename_legacy_checkpoint_keys(checkpoint["model_state_dict"])
    load_result = model.load_state_dict(renamed_state, strict=False)
    if load_result.missing_keys:
        print(f"Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys (ignored): {load_result.unexpected_keys}")
    model.to(device)
    model.eval()
    return model, image_height, image_width


def resolve_output_csv(
    checkpoint_path: str | Path,
    output_csv: str | None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if output_csv is not None:
        resolved_output = Path(output_csv)
    else:
        resolved_output = checkpoint_path.parent / f"submission_{checkpoint_path.stem}.csv"
    ensure_dir(resolved_output.parent)
    return resolved_output


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device()
    if args.image_size is not None:
        args.image_height = args.image_size
        args.image_width = args.image_size

    output_csv_path = resolve_output_csv(args.checkpoint, args.output_csv)

    model, image_height, image_width = load_model(
        args.checkpoint,
        device=device,
        image_size_override=args.image_size,
        image_height_override=args.image_height,
        image_width_override=args.image_width,
    )

    test_files = list_pickle_files(args.test_dir, args.max_test_samples)
    dataset = DrivingDataset(
        test_files,
        test=True,
        image_height=image_height,
        image_width=image_width,
    )
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
            last_pos = batch["last_pos"].to(device)
            last_heading = batch["last_heading"].to(device)

            outputs = model(camera, history, command)
            prediction_xy = decode_xy_from_ego(
                outputs["trajectory"][..., :2],
                origin_xy=last_pos,
                origin_heading=last_heading,
            )
            predictions.append(prediction_xy.cpu().numpy())
            sample_ids.extend(batch["sample_id"].tolist())

    all_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = all_predictions.reshape(all_predictions.shape[0], -1)

    column_names = ["id"]
    for step in range(1, all_predictions.shape[1] + 1):
        column_names.extend([f"x_{step}", f"y_{step}"])

    submission = pd.DataFrame(flat_predictions)
    submission.insert(0, "id", sample_ids)
    submission.columns = column_names
    submission.to_csv(output_csv_path, index=False)

    print(f"Saved submission to {output_csv_path}")
    print(f"Rows: {len(submission)}, columns: {len(submission.columns)}")


if __name__ == "__main__":
    main()
