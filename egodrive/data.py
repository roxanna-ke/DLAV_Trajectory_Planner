from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

COMMAND_TO_INDEX = {"forward": 0, "left": 1, "right": 2}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def encode_pose_sequence(
    sequence: torch.Tensor,
    *,
    origin_xy: torch.Tensor,
    origin_heading: torch.Tensor,
) -> torch.Tensor:
    relative_xy = sequence[:, :2] - origin_xy
    cosine = torch.cos(origin_heading)
    sine = torch.sin(origin_heading)
    xy_ego = torch.stack(
        [cosine * relative_xy[:, 0] + sine * relative_xy[:, 1],
         -sine * relative_xy[:, 0] + cosine * relative_xy[:, 1]],
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
    cosine = torch.cos(origin_heading)
    sine = torch.sin(origin_heading)
    while cosine.ndim < relative_xy.ndim - 1:
        cosine = cosine.unsqueeze(-1)
        sine = sine.unsqueeze(-1)
    while origin_xy.ndim < relative_xy.ndim:
        origin_xy = origin_xy.unsqueeze(-2)
    world_x = cosine * relative_xy[..., 0] - sine * relative_xy[..., 1]
    world_y = sine * relative_xy[..., 0] + cosine * relative_xy[..., 1]
    return torch.stack([world_x, world_y], dim=-1) + origin_xy


class GaussianNoise:
    def __init__(self, sigma: float = 0.02, probability: float = 0.5) -> None:
        self.sigma = sigma
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.probability:
            return tensor + torch.randn_like(tensor) * self.sigma
        return tensor


def list_pickle_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    files = sorted(Path(data_dir).expanduser().glob("*.pkl"), key=lambda path: int(path.stem))
    return files if limit is None else files[:limit]


def compute_real_style_stats(
    file_list: Iterable[str | Path],
    *,
    image_height: int = 224,
    image_width: int = 336,
    max_samples: int = 200,
) -> dict[str, torch.Tensor]:
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    means, stds = [], []
    for path in list(file_list)[:max_samples]:
        with Path(path).open("rb") as handle:
            sample = pickle.load(handle)
        image = transform(Image.fromarray(sample["camera"]))
        means.append(image.mean(dim=(1, 2)))
        stds.append(image.std(dim=(1, 2)))
    if not means:
        raise ValueError("Cannot compute style statistics from an empty file list")
    return {
        "mean": torch.stack(means).mean(0).view(3, 1, 1),
        "std": torch.stack(stds).mean(0).view(3, 1, 1),
    }


def adain_style_transfer(
    content_img: torch.Tensor,
    style_stats: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> torch.Tensor:
    content_mean = content_img.mean(dim=(1, 2), keepdim=True)
    content_std = content_img.std(dim=(1, 2), keepdim=True) + 1e-5
    normalized = (content_img - content_mean) / content_std
    stylized = normalized * style_stats["std"] + style_stats["mean"]
    return alpha * stylized + (1.0 - alpha) * content_img


class DrivingDataset(Dataset):
    """Load synthetic, auxiliary-task, or real samples through one interface."""

    def __init__(
        self,
        file_list: Iterable[str | Path],
        *,
        test: bool = False,
        augment: bool = False,
        require_command: bool = False,
        include_auxiliary: bool = False,
        image_height: int = 224,
        image_width: int = 336,
        style_stats: dict[str, torch.Tensor] | None = None,
        style_alpha: float = 0.5,
    ) -> None:
        self.samples = [Path(path) for path in file_list]
        self.test = test
        self.augment = augment and not test
        self.require_command = require_command
        self.include_auxiliary = include_auxiliary
        self.image_height = image_height
        self.image_width = image_width
        self.style_stats = style_stats
        self.style_alpha = style_alpha
        self.resize = transforms.Resize((image_height, image_width), interpolation=InterpolationMode.BILINEAR)
        self.eval_transform = transforms.Compose([
            self.resize,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.train_color = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        self.train_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        path = self.samples[index]
        with path.open("rb") as handle:
            sample = pickle.load(handle)

        image = Image.fromarray(sample["camera"])
        if self.augment:
            image_tensor = self.train_to_tensor(self.train_color(self.resize(image)))
            if self.style_stats is not None:
                image_tensor = adain_style_transfer(image_tensor, self.style_stats, self.style_alpha)
            image_tensor = self.normalize(image_tensor)
        else:
            image_tensor = self.eval_transform(image)

        history_raw = torch.as_tensor(sample["sdc_history_feature"], dtype=torch.float32)
        last_pos = history_raw[-1, :2].clone()
        last_heading = history_raw[-1, 2].clone()
        history = encode_pose_sequence(history_raw, origin_xy=last_pos, origin_heading=last_heading)
        if self.augment:
            history[:, :2] += torch.randn_like(history[:, :2]) * 0.03

        item: dict[str, torch.Tensor | int] = {
            "camera": image_tensor,
            "history": history,
            "last_pos": last_pos,
            "last_heading": last_heading,
            "sample_id": int(path.stem),
        }

        if "driving_command" in sample:
            item["command"] = torch.tensor(COMMAND_TO_INDEX[sample["driving_command"]], dtype=torch.long)
        elif self.require_command:
            raise KeyError(f"Sample {path} has no driving_command")

        if not self.test:
            if "sdc_future_feature" not in sample:
                raise KeyError(f"Sample {path} has no sdc_future_feature")
            future_raw = torch.as_tensor(sample["sdc_future_feature"], dtype=torch.float32)
            item["future"] = encode_pose_sequence(future_raw, origin_xy=last_pos, origin_heading=last_heading)
            if self.include_auxiliary:
                if "depth" not in sample or "semantic_label" not in sample:
                    raise KeyError(f"Sample {path} is missing auxiliary labels")
                item["depth"] = self._resize_depth(sample["depth"])
                item["semantic_label"] = self._resize_segmentation(sample["semantic_label"])
        return item

    def _resize_depth(self, depth: object) -> torch.Tensor:
        tensor = torch.as_tensor(depth, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        resized = F.interpolate(tensor.unsqueeze(0), size=(self.image_height, self.image_width), mode="bilinear", align_corners=False).squeeze(0)
        return torch.log1p(resized) / torch.log1p(torch.tensor(255.0))

    def _resize_segmentation(self, labels: object) -> torch.Tensor:
        tensor = torch.as_tensor(labels, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        resized = F.interpolate(tensor.unsqueeze(0), size=(self.image_height, self.image_width), mode="nearest")
        return resized.squeeze(0).squeeze(0).long()
