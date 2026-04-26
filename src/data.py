from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

COMMAND_TO_INDEX = {
    "forward": 0,
    "left": 1,
    "right": 2,
}


def build_camera_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def list_pickle_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(data_dir)
    files = sorted(root.glob("*.pkl"), key=lambda path: int(path.stem))
    if limit is not None:
        files = files[:limit]
    return files


class DrivingDataset(Dataset):
    def __init__(
        self,
        file_list: Iterable[str | Path],
        *,
        test: bool = False,
        image_size: int = 224,
    ) -> None:
        self.samples = [Path(path) for path in file_list]
        self.test = test
        self.camera_transform = build_camera_transform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        sample_path = self.samples[idx]
        with sample_path.open("rb") as handle:
            data = pickle.load(handle)

        camera = Image.fromarray(data["camera"])
        camera_tensor = self.camera_transform(camera)

        command = COMMAND_TO_INDEX[data["driving_command"]]
        history = torch.as_tensor(data["sdc_history_feature"], dtype=torch.float32)

        item: dict[str, torch.Tensor | int] = {
            "camera": camera_tensor,
            "command": torch.tensor(command, dtype=torch.long),
            "history": history,
            "sample_id": int(sample_path.stem),
        }

        if not self.test:
            item["future"] = torch.as_tensor(
                data["sdc_future_feature"], dtype=torch.float32
            )

        return item
