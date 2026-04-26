from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)


def _build_resnet(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
    elif backbone_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class EgoDrivePlanner(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = True,
        image_feature_dim: int = 128,
        history_feature_dim: int = 64,
        command_feature_dim: int = 16,
        gru_hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.1,
        future_steps: int = 60,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps

        self.image_encoder, image_backbone_dim = _build_resnet(
            backbone_name, pretrained_backbone
        )
        self.image_projection = nn.Sequential(
            nn.Linear(image_backbone_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.command_embedding = nn.Embedding(3, command_feature_dim)
        self.history_projection = nn.Sequential(
            nn.Linear(3, history_feature_dim),
            nn.ReLU(inplace=True),
        )

        gru_input_dim = history_feature_dim + image_feature_dim + command_feature_dim
        gru_dropout = dropout if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, future_steps * 3),
        )

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_steps, _ = history.shape

        image_features = self.image_encoder(camera)
        image_features = self.image_projection(image_features)
        image_context = image_features.unsqueeze(1).expand(-1, history_steps, -1)

        command_features = self.command_embedding(command)
        command_context = command_features.unsqueeze(1).expand(-1, history_steps, -1)

        history_features = self.history_projection(history)

        gru_input = torch.cat(
            [history_features, image_context, command_context],
            dim=-1,
        )
        _, hidden = self.gru(gru_input)
        plan_features = hidden[-1]

        future = self.decoder(plan_features)
        return future.view(batch_size, self.future_steps, 3)
