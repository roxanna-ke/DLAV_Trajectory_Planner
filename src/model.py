from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torchvision.models import ResNet34_Weights, resnet34


class EgoDrivePlanner(nn.Module):
    def __init__(
        self,
        *,
        pretrained_backbone: bool = True,
        image_feature_dim: int = 256,
        history_hidden_dim: int = 128,
        command_feature_dim: int = 32,
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
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
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

        # Fusion MLP: concat [vis, hist] -> context.
        # command_feature_dim is kept in the constructor for old configs/scripts.
        fusion_in = self.vision_dim + history_hidden_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Coarse autoregressive decoder with per-step context and timestep conditioning.
        self.timestep_embedding = nn.Embedding(future_steps, image_feature_dim)
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=4 + image_feature_dim + image_feature_dim,
            hidden_size=image_feature_dim,
        )
        self.trajectory_output_head = nn.Sequential(
            nn.Linear(image_feature_dim + image_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(image_feature_dim, 4),
        )

        # Refinement stage: attend layer3 spatial tokens with waypoint-aware queries.
        attn_heads = fusion_heads if image_feature_dim % fusion_heads == 0 else 1
        self.layer3_token_projection = nn.Conv2d(256, image_feature_dim, kernel_size=1)
        self.refinement_query_projection = nn.Sequential(
            nn.Linear(4 + image_feature_dim + image_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.refinement_attention = nn.MultiheadAttention(
            embed_dim=image_feature_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.trajectory_refinement_decoder = nn.GRU(
            input_size=4 + image_feature_dim + image_feature_dim,
            hidden_size=image_feature_dim,
            num_layers=1,
            batch_first=True,
        )
        self.trajectory_refinement_head = nn.Sequential(
            nn.Linear(image_feature_dim + image_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(image_feature_dim, 4),
        )
        # Start close to the coarse decoder behavior; refinement learns residuals.
        nn.init.zeros_(self.trajectory_refinement_head[-1].weight)
        nn.init.zeros_(self.trajectory_refinement_head[-1].bias)

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, _, H, W = camera.shape

        # Vision: tap layer3 for lightweight spatial cues, then layer4 for global context.
        layer3 = self.stem[:7](camera)                    # (B, 256, h, w)
        fmap = self.stem[7](layer3)                       # (B, 512, h, w)
        vis = self.global_pool(fmap).flatten(1)           # (B, 512)

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Fusion: concat [vis, hist] -> context vector
        ctx = self.context_mlp(torch.cat([vis, history_features], dim=1))

        # Stage 1: coarse autoregressive rollout with per-step ctx/time injection.
        device = camera.device
        h_dec = ctx
        step_input = torch.zeros(B, 4, device=device)
        step_outputs = []
        time_indices = torch.arange(self.future_steps, device=device)
        time_features = self.timestep_embedding(time_indices)
        for step_idx in range(self.future_steps):
            decoder_input = torch.cat(
                [step_input, ctx, time_features[step_idx].unsqueeze(0).expand(B, -1)],
                dim=1,
            )
            h_dec = self.trajectory_decoder_cell(decoder_input, h_dec)
            step_out = self.trajectory_output_head(torch.cat([h_dec, ctx], dim=1))
            step_outputs.append(step_out)
            step_input = step_out

        # Stage 1: coarse autoregressive rollout in ego-relative coordinates.
        pred_steps = torch.stack(step_outputs, dim=1)
        coarse_trajectory = torch.cat([
            torch.cumsum(pred_steps[..., :2], dim=1),
            pred_steps[..., 2:],
        ], dim=-1)

        # Stage 2: trajectory-aware refinement attends over layer3 spatial tokens.
        ctx_sequence = ctx.unsqueeze(1).expand(-1, self.future_steps, -1)
        time_sequence = time_features.unsqueeze(0).expand(B, -1, -1)
        spatial_tokens = self.layer3_token_projection(layer3).flatten(2).transpose(1, 2)
        refinement_query = self.refinement_query_projection(
            torch.cat([coarse_trajectory, ctx_sequence, time_sequence], dim=-1)
        )
        attended_spatial, _ = self.refinement_attention(
            refinement_query,
            spatial_tokens,
            spatial_tokens,
        )
        refinement_input = torch.cat(
            [coarse_trajectory, ctx_sequence, attended_spatial],
            dim=-1,
        )
        refinement_hidden, _ = self.trajectory_refinement_decoder(refinement_input)
        refinement_delta = self.trajectory_refinement_head(
            torch.cat([refinement_hidden, attended_spatial], dim=-1)
        )
        trajectory = coarse_trajectory + refinement_delta

        outputs: dict[str, torch.Tensor] = {"trajectory": trajectory}
        return outputs
