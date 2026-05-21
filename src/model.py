from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class EgoDrivePlanner(nn.Module):
    def __init__(
        self,
        *,
        pretrained_backbone: bool = True,
        image_feature_dim: int = 256,
        history_hidden_dim: int = 128,
        history_layers: int = 2,
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        dropout: float = 0.1,
        future_steps: int = 60,
        camera_feature_weight: float = 1.0,
        history_feature_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.camera_feature_weight = camera_feature_weight
        self.history_feature_weight = history_feature_weight

        # Vision encoder: ResNet-34 stem (conv layers) + global average pooling
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained_backbone else None)
        self.stem_layer3 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        self.stem_layer4 = backbone.layer4
        self.vision_dim = 512
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # History encoder: 2-layer GRU
        gru_dropout = dropout if history_layers > 1 else 0.0
        self.history_encoder = nn.GRU(
            input_size=4,
            hidden_size=history_hidden_dim,
            num_layers=history_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        # Fusion MLP: concat [vis, hist] -> context
        fusion_in = self.vision_dim + history_hidden_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Direct residual decoder: constant-velocity prior + learned correction.
        self.trajectory_residual_head = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(image_feature_dim, future_steps * 4),
        )

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = camera.size(0)

        # Vision: layer3 + layer4 + global pool
        layer3 = self.stem_layer3(camera)                 # (B, 256, h3, w3)
        fmap = self.stem_layer4(layer3)                   # (B, 512, h4, w4)
        vis = self.global_pool(fmap).flatten(1)           # (B, 512)

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Fusion: concat [vis, hist] -> context vector
        ctx = self.context_mlp(
            torch.cat(
                [
                    vis * self.camera_feature_weight,
                    history_features * self.history_feature_weight,
                ],
                dim=1,
            )
        )

        residual = self.trajectory_residual_head(ctx).view(B, self.future_steps, 4)

        # The encoded history is in the current ego frame, so the last xy is near zero.
        # Use a short moving average of recent velocity as a stable future prior.
        history_velocity = history[:, 1:, :2] - history[:, :-1, :2]
        recent_velocity = history_velocity[:, -5:].mean(dim=1)
        steps = torch.arange(
            1,
            self.future_steps + 1,
            device=history.device,
            dtype=history.dtype,
        ).view(1, self.future_steps, 1)
        xy_prior = steps * recent_velocity.unsqueeze(1)

        # Residual xy corrects the kinematic prior; heading is predicted directly.
        trajectory = torch.cat(
            [
                xy_prior + residual[..., :2],
                residual[..., 2:],
            ],
            dim=-1,
        )

        outputs: dict[str, torch.Tensor] = {"trajectory": trajectory}
        return outputs
