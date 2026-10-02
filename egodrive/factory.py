from __future__ import annotations

from typing import Any

from .models import EgoDrivePlanner


def build_model(config: dict[str, Any]) -> EgoDrivePlanner:
    return EgoDrivePlanner(
        pretrained_backbone=bool(config.get("pretrained_backbone", True)),
        image_feature_dim=int(config.get("image_feature_dim", 256)),
        history_hidden_dim=int(config.get("history_hidden_dim", 128)),
        command_feature_dim=int(config.get("command_feature_dim", 32)),
        history_layers=int(config.get("history_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        future_steps=int(config.get("future_steps", 60)),
        use_command=bool(config.get("use_command", False)),
        fusion=str(config.get("fusion", "history_conditioned_spatial_attention")),
        decoder=str(config.get("decoder", "autoregressive_gru")),
        use_depth_head=bool(config.get("use_depth_head", False)),
        use_segmentation_head=bool(config.get("use_segmentation_head", False)),
        num_segmentation_classes=int(config.get("num_segmentation_classes", 15)),
    )
