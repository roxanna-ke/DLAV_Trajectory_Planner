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


def rename_legacy_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("backbone.") or key.startswith("vision_encoder."):
            key = "stem." + key.split(".", 1)[1]

        if key.startswith("stem."):
            suffix = key.split(".", 1)[1]
            parts = suffix.split(".", 1)
            try:
                stem_index = int(parts[0])
            except ValueError:
                renamed_state[key] = value
                continue

            remainder = parts[1] if len(parts) > 1 else ""
            if stem_index <= 6:
                new_key = f"stem_layer3.{stem_index}"
            elif stem_index == 7:
                new_key = "stem_layer4"
            else:
                renamed_state[key] = value
                continue

            if remainder:
                new_key = f"{new_key}.{remainder}"
            renamed_state[new_key] = value
            continue

        renamed_state[key] = value

    return renamed_state
