from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

COMMAND_TO_INDEX = {
    "forward": 0,
    "left": 1,
    "right": 2,
}


def build_eval_camera_transform(
    image_height: int = 224,
    image_width: int = 336,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_height, image_width),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_train_camera_transform(
    image_height: int = 224,
    image_width: int = 336,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_height, image_width),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
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
        augment: bool = False,
        image_height: int = 224,
        image_width: int = 336,
    ) -> None:
        self.samples = [Path(path) for path in file_list]
        self.test = test
        self.augment = augment
        self.image_height = image_height
        self.image_width = image_width
        self.camera_transform = (
            build_eval_camera_transform(
                image_height=image_height,
                image_width=image_width,
            )
            if test or not augment
            else build_train_camera_transform(
                image_height=image_height,
                image_width=image_width,
            )
        )

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
            item["depth"] = self._resize_depth_map(data["depth"])
            item["semantic_label"] = self._resize_segmentation_map(data["semantic_label"])

        return item

    def _resize_depth_map(self, depth_map: object) -> torch.Tensor:
        depth_tensor = torch.as_tensor(depth_map, dtype=torch.float32)
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        elif depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported depth map shape: {tuple(depth_tensor.shape)}")

        resized = torch_f.interpolate(
            depth_tensor.unsqueeze(0),
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _resize_segmentation_map(self, segmentation_map: object) -> torch.Tensor:
        segmentation_tensor = torch.as_tensor(segmentation_map, dtype=torch.float32)
        if segmentation_tensor.ndim == 2:
            segmentation_tensor = segmentation_tensor.unsqueeze(0)
        elif segmentation_tensor.ndim == 3:
            segmentation_tensor = segmentation_tensor.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Unsupported segmentation map shape: {tuple(segmentation_tensor.shape)}"
            )

        resized = torch_f.interpolate(
            segmentation_tensor.unsqueeze(0),
            size=(self.image_height, self.image_width),
            mode="nearest",
        )
        return resized.squeeze(0).squeeze(0).long()
