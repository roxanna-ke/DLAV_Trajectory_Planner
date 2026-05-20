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
    ) -> None:
        super().__init__()
        self.future_steps = future_steps

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

        # Fusion MLP: concat [vis, hist] → context
        fusion_in = self.vision_dim + history_hidden_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Autoregressive GRU decoder
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=4, hidden_size=image_feature_dim,
        )
        self.trajectory_output_head = nn.Linear(image_feature_dim, 4)

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, _, H, W = camera.shape

        # Vision: layer3 + layer4 + global pool
        layer3 = self.stem_layer3(camera)                 # (B, 256, h3, w3)
        fmap = self.stem_layer4(layer3)                   # (B, 512, h4, w4)
        vis = self.global_pool(fmap).flatten(1)           # (B, 512)

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Fusion: concat [vis, hist] → context vector
        ctx = self.context_mlp(torch.cat([vis, history_features], dim=1))

        # Autoregressive GRU decoder
        device = camera.device
        h_dec = ctx
        step_input = torch.zeros(B, 4, device=device)
        step_outputs = []
        for _ in range(self.future_steps):
            h_dec = self.trajectory_decoder_cell(step_input, h_dec)
            step_out = self.trajectory_output_head(h_dec)
            step_outputs.append(step_out)
            step_input = step_out

        # Stack: (B, T, 4) where 4 = [dx, dy, sin(heading), cos(heading)]
        pred_steps = torch.stack(step_outputs, dim=1)
        # Cumsum on xy deltas to get absolute relative positions
        # Heading sin/cos are predicted as absolute values (not deltas)
        trajectory = torch.cat([
            torch.cumsum(pred_steps[..., :2], dim=1),
            pred_steps[..., 2:],
        ], dim=-1)

        outputs: dict[str, torch.Tensor] = {"trajectory": trajectory}
        return outputs