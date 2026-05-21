from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def encode_pose_sequence(
    sequence: torch.Tensor,
    *,
    origin_xy: torch.Tensor,
    origin_heading: torch.Tensor,
) -> torch.Tensor:
    relative_xy = sequence[:, :2] - origin_xy
    cos_heading = torch.cos(origin_heading)
    sin_heading = torch.sin(origin_heading)
    xy_ego = torch.stack(
        [
            cos_heading * relative_xy[:, 0] + sin_heading * relative_xy[:, 1],
            -sin_heading * relative_xy[:, 0] + cos_heading * relative_xy[:, 1],
        ],
        dim=-1,
    )
    heading = sequence[:, 2:3] - origin_heading
    return torch.cat([xy_ego, torch.sin(heading), torch.cos(heading)], dim=-1)


def decode_xy_from_ego(
    relative_xy: torch.Tensor,
    *,
    origin_xy: torch.Tensor,
    origin_heading: torch.Tensor,
) -> torch.Tensor:
    cos_heading = torch.cos(origin_heading)
    sin_heading = torch.sin(origin_heading)

    for _ in range(relative_xy.ndim - origin_heading.ndim - 1):
        cos_heading = cos_heading.unsqueeze(-1)
        sin_heading = sin_heading.unsqueeze(-1)
    for _ in range(relative_xy.ndim - origin_xy.ndim):
        origin_xy = origin_xy.unsqueeze(-2)

    world_x = cos_heading * relative_xy[..., 0] - sin_heading * relative_xy[..., 1]
    world_y = sin_heading * relative_xy[..., 0] + cos_heading * relative_xy[..., 1]
    return torch.stack([world_x, world_y], dim=-1) + origin_xy


class GaussianNoise:
    """Add additive Gaussian noise to a tensor (applied after Normalize)."""

    def __init__(self, sigma: float = 0.02, p: float = 0.5) -> None:
        self.sigma = sigma
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn_like(tensor) * self.sigma
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, p={self.p})"


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
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.25,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            GaussianNoise(sigma=0.02, p=0.5),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    )


def list_pickle_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(data_dir).expanduser()
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

        history_raw = torch.as_tensor(data["sdc_history_feature"], dtype=torch.float32)
        last_pos = history_raw[-1, :2].clone()
        last_heading = history_raw[-1, 2].clone()
        history = encode_pose_sequence(
            history_raw,
            origin_xy=last_pos,
            origin_heading=last_heading,
        )
        if self.augment:
            history[:, :2] += torch.randn_like(history[:, :2]) * 0.05

        item: dict[str, torch.Tensor | int] = {
            "camera": camera_tensor,
            "history": history,
            "last_pos": last_pos,
            "last_heading": last_heading,
            "sample_id": int(sample_path.stem),
        }

        if not self.test:
            future_raw = torch.as_tensor(
                data["sdc_future_feature"], dtype=torch.float32
            )
            item["future"] = encode_pose_sequence(
                future_raw,
                origin_xy=last_pos,
                origin_heading=last_heading,
            )

        return item
