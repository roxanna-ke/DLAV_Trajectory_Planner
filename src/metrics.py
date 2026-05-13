from __future__ import annotations

import torch
import torch.nn.functional as torch_f


def displacement_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    deltas = prediction[..., :2] - target[..., :2]
    ade = torch.linalg.norm(deltas, dim=-1).mean()
    fde = torch.linalg.norm(deltas[:, -1], dim=-1).mean()
    return ade, fde


def trajectory_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    time_weights: torch.Tensor,
    heading_weight: float,
    fde_weight: float,
    ade_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_xy = prediction[..., :2].contiguous()
    target_xy = target[..., :2].contiguous()
    pred_heading = prediction[..., 2:].contiguous()
    target_heading = target[..., 2:].contiguous()

    # ADE — directly optimizes the competition metric (DOMINANT)
    ade_loss = torch.linalg.norm(pred_xy - target_xy, dim=-1).mean()

    # Time-weighted xy loss (for per-step accuracy with later-step emphasis)
    xy_residual = torch_f.smooth_l1_loss(
        pred_xy,
        target_xy,
        reduction="none",
        beta=1.0,
    )
    xy_loss = (xy_residual * time_weights).mean()

    heading_loss = torch_f.smooth_l1_loss(
        pred_heading,
        target_heading,
        beta=1.0,
    )
    fde_loss = torch.linalg.norm(pred_xy[:, -1] - target_xy[:, -1], dim=-1).mean()

    total_loss = ade_weight * ade_loss + xy_loss + heading_weight * heading_loss + fde_weight * fde_loss
    metrics = {
        "ade_loss": float(ade_loss.detach().item()),
        "xy_loss": float(xy_loss.detach().item()),
        "heading_loss": float(heading_loss.detach().item()),
        "fde_loss": float(fde_loss.detach().item()),
        "loss": float(total_loss.detach().item()),
    }
    return total_loss, metrics


def depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Smooth L1 loss on log1p-normalized depth predictions."""
    loss = torch_f.smooth_l1_loss(prediction.contiguous(), target.contiguous())
    metrics = {"depth_loss": float(loss.detach().item())}
    return loss, metrics


def segmentation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_safe = target.clamp(max=14).contiguous()
    loss = torch_f.cross_entropy(
        prediction.contiguous(),
        target_safe,
        ignore_index=255,
    )
    metrics = {"segmentation_loss": float(loss.detach().item())}
    return loss, metrics


def compute_total_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    time_weights: torch.Tensor,
    heading_weight: float,
    fde_weight: float,
    ade_weight: float,
    segmentation_loss_weight: float,
    depth_loss_weight: float,
    use_segmentation_head: bool = False,
    use_depth_head: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine trajectory_objective + segmentation_loss + depth_loss with per-task weights."""
    loss, metrics = trajectory_objective(
        outputs["trajectory"],
        targets["future"],
        time_weights=time_weights,
        heading_weight=heading_weight,
        fde_weight=fde_weight,
        ade_weight=ade_weight,
    )

    # Segmentation auxiliary loss
    weighted_segmentation_loss = torch.tensor(0.0, device=loss.device)
    if (
        use_segmentation_head
        and outputs.get("segmentation_logits") is not None
        and "semantic_label" in targets
    ):
        aux_loss, seg_metrics = segmentation_loss(
            outputs["segmentation_logits"],
            targets["semantic_label"],
        )
        weighted_segmentation_loss = segmentation_loss_weight * aux_loss
        loss = loss + weighted_segmentation_loss
        metrics.update(seg_metrics)
    else:
        metrics["segmentation_loss"] = 0.0

    # Depth auxiliary loss
    weighted_depth_loss = torch.tensor(0.0, device=loss.device)
    if (
        use_depth_head
        and outputs.get("depth_prediction") is not None
        and "depth" in targets
    ):
        aux_loss, depth_metrics = depth_loss(
            outputs["depth_prediction"],
            targets["depth"],
        )
        weighted_depth_loss = depth_loss_weight * aux_loss
        loss = loss + weighted_depth_loss
        metrics.update(depth_metrics)
    else:
        metrics["depth_loss"] = 0.0

    metrics["loss"] = float(loss.detach().item())
    metrics["weighted_segmentation_loss"] = float(weighted_segmentation_loss.detach().item())
    metrics["weighted_depth_loss"] = float(weighted_depth_loss.detach().item())
    return loss, metrics
