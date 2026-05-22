from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
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
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def adain_style_transfer(
    content_img: torch.Tensor,
    style_stats: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> torch.Tensor:
    c_mean = content_img.mean(dim=[1, 2], keepdim=True)
    c_std = content_img.std(dim=[1, 2], keepdim=True) + 1e-5
    s_mean = style_stats["mean"]
    s_std = style_stats["std"]

    normalized = (content_img - c_mean) / c_std
    stylized = normalized * s_std + s_mean
    return alpha * stylized + (1 - alpha) * content_img


def list_pickle_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(data_dir).expanduser()
    files = sorted(root.glob("*.pkl"), key=lambda path: int(path.stem))
    if limit is not None:
        files = files[:limit]
    return files


def compute_real_style_stats(
    file_list: Iterable[str | Path],
    *,
    image_height: int = 224,
    image_width: int = 336,
    max_samples: int = 200,
) -> dict[str, torch.Tensor]:
    to_tensor = transforms.Compose(
        [
            transforms.Resize(
                (image_height, image_width),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    means = []
    stds = []
    for sample_path in list(file_list)[:max_samples]:
        with Path(sample_path).open("rb") as handle:
            data = pickle.load(handle)
        image = to_tensor(Image.fromarray(data["camera"]))
        means.append(image.mean(dim=[1, 2]))
        stds.append(image.std(dim=[1, 2]))
    if not means:
        raise ValueError("Cannot compute style statistics from an empty file list.")
    return {
        "mean": torch.stack(means).mean(0).view(3, 1, 1),
        "std": torch.stack(stds).mean(0).view(3, 1, 1),
    }


class DrivingDataset(Dataset):
    def __init__(
        self,
        file_list: Iterable[str | Path],
        *,
        test: bool = False,
        augment: bool = False,
        image_height: int = 224,
        image_width: int = 336,
        style_stats: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.samples = [Path(path) for path in file_list]
        self.test = test
        self.augment = augment
        self.style_stats = style_stats
        self.image_height = image_height
        self.image_width = image_width
        self.resize = transforms.Resize(
            (image_height, image_width),
            interpolation=InterpolationMode.BILINEAR,
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )
        self.random_grayscale = transforms.RandomGrayscale(p=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.eval_camera_transform = build_eval_camera_transform(
            image_height=image_height,
            image_width=image_width,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        sample_path = self.samples[idx]
        with sample_path.open("rb") as handle:
            data = pickle.load(handle)

        camera = Image.fromarray(data["camera"])
        if self.augment:
            camera = self.resize(camera)
            camera = self.color_jitter(camera)
            camera = self.random_grayscale(camera)
            camera_tensor = self.to_tensor(camera)
            if self.style_stats is not None:
                camera_tensor = adain_style_transfer(camera_tensor, self.style_stats)
            camera_tensor = self.normalize(camera_tensor)
        else:
            camera_tensor = self.eval_camera_transform(camera)

        history_raw = torch.as_tensor(data["sdc_history_feature"], dtype=torch.float32)
        last_pos = history_raw[-1, :2].clone()
        last_heading = history_raw[-1, 2].clone()
        history = encode_pose_sequence(
            history_raw,
            origin_xy=last_pos,
            origin_heading=last_heading,
        )

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
