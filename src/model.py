from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class SpatialAttention(nn.Module):
    def __init__(self, query_dim: int, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        B, _, H, W = feat_map.shape
        Q = self.q_proj(query).unsqueeze(1)
        K = self.k_proj(feat_map).view(B, -1, H * W)
        V = self.v_proj(feat_map).view(B, -1, H * W).transpose(1, 2)

        scores = torch.bmm(Q, K) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        context = torch.bmm(attn, V).squeeze(1)
        context = self.norm(context)
        return self.out_proj(context)


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

        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained_backbone else None)
        self.stem_layer3 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.stem_layer4 = backbone.layer4

        self.spatial_attn = SpatialAttention(
            query_dim=history_hidden_dim,
            feat_dim=512,
            hidden_dim=256,
        )

        gru_dropout = dropout if history_layers > 1 else 0.0
        self.history_encoder = nn.GRU(
            input_size=6,
            hidden_size=history_hidden_dim,
            num_layers=history_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        fusion_in = 256 + history_hidden_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=4,
            hidden_size=image_feature_dim,
        )
        self.trajectory_output_head = nn.Linear(image_feature_dim, 4)

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = camera.size(0)
        vel = torch.zeros_like(history[..., :2])
        vel[:, 1:] = history[:, 1:, :2] - history[:, :-1, :2]
        vel[:, 0] = vel[:, 1]
        history_enhanced = torch.cat([history, vel], dim=-1)

        layer3 = self.stem_layer3(camera)
        fmap = self.stem_layer4(layer3)

        _, hist_h = self.history_encoder(history_enhanced)
        hist_feat = hist_h[-1]

        vis_attn = self.spatial_attn(hist_feat, fmap)

        ctx = self.context_mlp(torch.cat([vis_attn, hist_feat], dim=1))

        h_dec = ctx
        step_input = torch.zeros(B, 4, device=camera.device, dtype=camera.dtype)
        steps = []
        for _ in range(self.future_steps):
            h_dec = self.trajectory_decoder_cell(step_input, h_dec)
            out = self.trajectory_output_head(h_dec)
            steps.append(out)
            step_input = out

        pred = torch.stack(steps, dim=1)
        trajectory = torch.cat(
            [
                torch.cumsum(pred[..., :2], dim=1),
                pred[..., 2:],
            ],
            dim=-1,
        )
        return {"trajectory": trajectory}
