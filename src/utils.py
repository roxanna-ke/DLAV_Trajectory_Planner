from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def rename_legacy_checkpoint_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    renamed_state: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        if key.startswith("backbone.") or key.startswith("vision_encoder."):
            new_key = "stem." + key.split(".", 1)[1]
        elif key.startswith("decoder_cell."):
            new_key = "trajectory_decoder_cell." + key.split(".", 1)[1]
        elif key.startswith("output_head."):
            new_key = "trajectory_output_head." + key.split(".", 1)[1]

        renamed_state[new_key] = value

    return renamed_state
