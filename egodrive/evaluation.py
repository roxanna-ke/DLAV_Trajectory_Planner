from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import DrivingDataset, decode_xy_from_ego, list_pickle_files
from .factory import build_model
from .utils import get_device, move_batch_to_device


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint["model_config"])
    # The checkpoint already contains backbone weights; inference must not
    # attempt to download ImageNet weights again.
    model_config["pretrained_backbone"] = False
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_to_csv(
    model: torch.nn.Module,
    files: list[Path],
    output_csv: str | Path,
    *,
    device: torch.device,
    image_height: int,
    image_width: int,
    batch_size: int = 32,
    num_workers: int = 0,
    require_command: bool = False,
) -> Path:
    dataset = DrivingDataset(files, test=True, image_height=image_height, image_width=image_width, require_command=require_command)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions, ids = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            kwargs = {"command": batch["command"]} if "command" in batch else {}
            outputs = model(batch["camera"], batch["history"], **kwargs)
            xy = decode_xy_from_ego(outputs["trajectory"][..., :2], origin_xy=batch["last_pos"], origin_heading=batch["last_heading"])
            predictions.append(xy.cpu().numpy())
            ids.extend(batch["sample_id"])
    if not predictions:
        raise ValueError("No test samples found")
    values = np.concatenate(predictions, axis=0)
    columns = ["id"] + [axis for step in range(1, values.shape[1] + 1) for axis in (f"x_{step}", f"y_{step}")]
    frame = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns[1:])
    frame.insert(0, "id", ids)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def validate_submission(path: str | Path, future_steps: int = 60) -> None:
    frame = pd.read_csv(path)
    expected = ["id"] + [axis for step in range(1, future_steps + 1) for axis in (f"x_{step}", f"y_{step}")]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected submission columns: {list(frame.columns)}")
    if frame.isna().any().any() or not np.isfinite(frame.iloc[:, 1:].to_numpy()).all():
        raise ValueError("Submission contains NaN or infinite values")
