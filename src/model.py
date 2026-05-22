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

        # Autoregressive trajectory decoder, following TransFuser's waypoint GRU style.
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=4,
            hidden_size=image_feature_dim,
        )
        self.trajectory_output_head = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(image_feature_dim, 4),
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

        # The encoded history is in the current ego frame, so the last xy is near zero.
        # Use a short moving average of recent velocity as a stable per-step prior.
        history_velocity = history[:, 1:, :2] - history[:, :-1, :2]
        recent_velocity = history_velocity[:, -5:].mean(dim=1)

        decoder_hidden = ctx
        decoder_input = torch.zeros(
            B,
            4,
            device=history.device,
            dtype=history.dtype,
        )
        decoder_input[:, 3] = 1.0
        xy = decoder_input[:, :2]
        trajectory_steps = []

        for _ in range(self.future_steps):
            decoder_hidden = self.trajectory_decoder_cell(decoder_input, decoder_hidden)
            step_output = self.trajectory_output_head(decoder_hidden)

            xy = xy + recent_velocity + step_output[:, :2]
            heading = step_output[:, 2:]
            decoder_input = torch.cat([xy, heading], dim=1)
            trajectory_steps.append(decoder_input)

        trajectory = torch.stack(trajectory_steps, dim=1)

        outputs: dict[str, torch.Tensor] = {"trajectory": trajectory}
        return outputs
