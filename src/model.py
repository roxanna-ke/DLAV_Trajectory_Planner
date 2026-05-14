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
        use_depth_head: bool = True,
        use_segmentation_head: bool = True,
        num_segmentation_classes: int = 15,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.use_depth_head = use_depth_head
        self.use_segmentation_head = use_segmentation_head

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

        # Command embedding
        self.command_embedding = nn.Embedding(3, command_feature_dim)

        # FiLM conditioning: command → vision modulation
        self.film_scale = nn.Linear(command_feature_dim, self.vision_dim)
        self.film_bias = nn.Linear(command_feature_dim, self.vision_dim)

        # Fusion MLP: concat [vis, hist, cmd] → context
        fusion_in = self.vision_dim + history_hidden_dim + command_feature_dim
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

        # Lightweight shared auxiliary decoder (like notebook)
        # Gradients flow back to stem, helping backbone learn perception features
        self.aux_decoder = None
        self.aux_layer4_projection = None
        self.aux_layer3_projection = None
        self.aux_token_projection = None
        self.semantic_head = None
        self.depth_head = None

        if use_depth_head or use_segmentation_head:
            self.aux_layer4_projection = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.aux_layer3_projection = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.aux_decoder = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.aux_token_projection = nn.Conv2d(128, image_feature_dim, kernel_size=1)
        if use_segmentation_head:
            self.semantic_head = nn.Conv2d(128, num_segmentation_classes, kernel_size=1)
        if use_depth_head:
            self.depth_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor,
        aux_token_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        B, _, H, W = camera.shape

        # Vision: tap layer3 for lightweight spatial cues, then layer4 for global context.
        layer3 = self.stem[:7](camera)                    # (B, 256, h, w)
        fmap = self.stem[7](layer3)                       # (B, 512, h, w)
        vis = self.global_pool(fmap).flatten(1)           # (B, 512)

        # FiLM conditioning on command
        command_features = self.command_embedding(command)
        scale = torch.sigmoid(self.film_scale(command_features))
        bias = self.film_bias(command_features)
        vis = vis * scale + bias

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Fusion: concat [vis, hist, cmd] → context vector
        ctx = self.context_mlp(torch.cat([vis, history_features, command_features], dim=1))

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

        # Stage 2: trajectory-aware refinement attends over layer3 tokens and
        # auxiliary perception tokens.
        ctx_sequence = ctx.unsqueeze(1).expand(-1, self.future_steps, -1)
        time_sequence = time_features.unsqueeze(0).expand(B, -1, -1)
        spatial_tokens = self.layer3_token_projection(layer3).flatten(2).transpose(1, 2)
        aux = None
        if self.aux_decoder is not None:
            aux_layer4 = self.aux_layer4_projection(fmap)
            aux_layer4 = torch_f.interpolate(
                aux_layer4,
                size=layer3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            aux_layer3 = self.aux_layer3_projection(layer3)
            aux = self.aux_decoder(torch.cat([aux_layer4, aux_layer3], dim=1))
            if aux_token_scale > 0.0:
                aux_tokens = self.aux_token_projection(aux).flatten(2).transpose(1, 2)
                aux_tokens = aux_tokens * aux_token_scale
                spatial_tokens = torch.cat([spatial_tokens, aux_tokens], dim=1)
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

        # Auxiliary heads: shared feature map → lightweight decoder → upsample
        # Gradients flow back to stem, which is the key mechanism for aux to help planning
        if self.aux_decoder is not None and (self.depth_head is not None or self.semantic_head is not None):
            if aux is None:
                aux_layer4 = self.aux_layer4_projection(fmap)
                aux_layer4 = torch_f.interpolate(
                    aux_layer4,
                    size=layer3.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                aux_layer3 = self.aux_layer3_projection(layer3)
                aux = self.aux_decoder(torch.cat([aux_layer4, aux_layer3], dim=1))
            if self.semantic_head is not None:
                semantic_logits = self.semantic_head(aux)  # (B, C, h, w)
                semantic_logits = torch_f.interpolate(
                    semantic_logits, size=(H, W), mode="bilinear", align_corners=False,
                )
                outputs["segmentation_logits"] = semantic_logits
            if self.depth_head is not None:
                depth_pred = self.depth_head(aux)         # (B, 1, h, w)
                depth_pred = torch_f.interpolate(
                    depth_pred, size=(H, W), mode="bilinear", align_corners=False,
                )
                depth_pred = torch.sigmoid(depth_pred)
                outputs["depth"] = depth_pred

        return outputs
