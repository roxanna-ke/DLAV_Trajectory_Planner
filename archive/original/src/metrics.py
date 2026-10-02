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
    prediction_xy = prediction[..., :2].contiguous()
    target_xy = target[..., :2].contiguous()
    prediction_heading = prediction[..., 2:].contiguous()
    target_heading = target[..., 2:].contiguous()

    xy_loss = torch.nn.functional.smooth_l1_loss(
        prediction_xy, target_xy
    )
    heading_loss = torch.nn.functional.smooth_l1_loss(
        prediction_heading, target_heading
    )
    loss = xy_loss + heading_weight * heading_loss
    metrics = {
        "xy_loss": float(xy_loss.detach().item()),
        "heading_loss": float(heading_loss.detach().item()),
        "loss": float(loss.detach().item()),
    }
    return loss, metrics