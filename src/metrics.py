from __future__ import annotations

import torch


def displacement_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    deltas = prediction[..., :2] - target[..., :2]
    ade = torch.linalg.norm(deltas, dim=-1).mean()
    fde = torch.linalg.norm(deltas[:, -1], dim=-1).mean()
    return ade, fde


def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    heading_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    xy_loss = torch.nn.functional.smooth_l1_loss(
        prediction[..., :2], target[..., :2]
    )
    heading_loss = torch.nn.functional.smooth_l1_loss(
        prediction[..., 2], target[..., 2]
    )
    loss = xy_loss + heading_weight * heading_loss
    metrics = {
        "xy_loss": float(xy_loss.detach().item()),
        "heading_loss": float(heading_loss.detach().item()),
        "loss": float(loss.detach().item()),
    }
    return loss, metrics
