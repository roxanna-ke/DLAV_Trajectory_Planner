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


def depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.nn.functional.smooth_l1_loss(prediction.contiguous(), target.contiguous())
    metrics = {"depth_loss": float(loss.detach().item())}
    return loss, metrics


def segmentation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_safe = target.clamp(max=14).contiguous()
    loss = torch.nn.functional.cross_entropy(
        prediction.contiguous(),
        target_safe,
        ignore_index=255,
    )
    metrics = {"segmentation_loss": float(loss.detach().item())}
    return loss, metrics


def multitask_loss(
    trajectory_prediction: torch.Tensor,
    trajectory_target: torch.Tensor,
    depth_prediction: torch.Tensor | None,
    depth_target: torch.Tensor | None,
    segmentation_prediction: torch.Tensor | None,
    segmentation_target: torch.Tensor | None,
    *,
    heading_weight: float = 0.1,
    depth_loss_weight: float = 0.05,
    segmentation_loss_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss, metrics = trajectory_loss(
        trajectory_prediction,
        trajectory_target,
        heading_weight=heading_weight,
    )

    if depth_prediction is not None and depth_target is not None:
        aux_loss, depth_metrics = depth_loss(depth_prediction, depth_target)
        total_loss = total_loss + depth_loss_weight * aux_loss
        metrics.update(depth_metrics)
    else:
        metrics["depth_loss"] = 0.0

    if segmentation_prediction is not None and segmentation_target is not None:
        aux_loss, segmentation_metrics = segmentation_loss(
            segmentation_prediction,
            segmentation_target,
        )
        total_loss = total_loss + segmentation_loss_weight * aux_loss
        metrics.update(segmentation_metrics)
    else:
        metrics["segmentation_loss"] = 0.0

    metrics["loss"] = float(total_loss.detach().item())
    return total_loss, metrics
