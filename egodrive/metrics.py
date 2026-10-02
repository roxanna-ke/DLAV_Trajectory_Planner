from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def displacement_errors(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = prediction[..., :2] - target[..., :2]
    return torch.linalg.norm(error, dim=-1).mean(), torch.linalg.norm(error[:, -1], dim=-1).mean()


def trajectory_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    time_weights: torch.Tensor | None = None,
    heading_weight: float = 0.1,
    ade_weight: float = 1.0,
    fde_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    if time_weights is None:
        time_weights = prediction.new_ones((1, prediction.size(1), 1))
    xy_residual = F.smooth_l1_loss(prediction[..., :2], target[..., :2], reduction="none")
    xy_loss = (xy_residual * time_weights).mean()
    heading_loss = F.smooth_l1_loss(prediction[..., 2:], target[..., 2:])
    ade = torch.linalg.norm(prediction[..., :2] - target[..., :2], dim=-1).mean()
    fde = torch.linalg.norm(prediction[:, -1, :2] - target[:, -1, :2], dim=-1).mean()
    loss = xy_loss + heading_weight * heading_loss + ade_weight * ade + fde_weight * fde
    return loss, {"loss": float(loss.detach()), "xy_loss": float(xy_loss.detach()), "heading_loss": float(heading_loss.detach()), "ade": float(ade.detach()), "fde": float(fde.detach())}


def auxiliary_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    depth_weight: float = 0.1,
    segmentation_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = next(iter(outputs.values())).new_zeros(())
    metrics: dict[str, float] = {}
    if "depth" in outputs and "depth" in batch:
        value = F.smooth_l1_loss(outputs["depth"], batch["depth"])
        total = total + depth_weight * value
        metrics["depth_loss"] = float(value.detach())
    if "segmentation_logits" in outputs and "semantic_label" in batch:
        labels = batch["semantic_label"].clone()
        labels[(labels < 0) | (labels > 14)] = 255
        value = F.cross_entropy(outputs["segmentation_logits"], labels, ignore_index=255)
        total = total + segmentation_weight * value
        metrics["segmentation_loss"] = float(value.detach())
    return total, metrics


def build_time_weights(steps: int, start: float = 1.0, end: float = 2.0) -> torch.Tensor:
    weights = torch.linspace(start, end, steps)
    return (weights / weights.mean()).view(1, steps, 1)
