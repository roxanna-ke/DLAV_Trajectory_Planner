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

        # Auxiliary feature dimension (always 128; zeros when aux heads are off, so shapes stay consistent across baseline ↔ aux resume)
        self.aux_feat_dim = 128

        # Fusion MLP: concat [vis, hist, cmd, aux_feat] → context
        fusion_in = self.vision_dim + history_hidden_dim + command_feature_dim + self.aux_feat_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Autoregressive GRU decoder (ctx injected at every step)
        self.ctx_proj = nn.Linear(image_feature_dim, 4)
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=8, hidden_size=image_feature_dim,  # 4 (step) + 4 (ctx_proj)
        )
        self.trajectory_output_head = nn.Linear(image_feature_dim, 4)

        # Lightweight shared auxiliary decoder (like notebook)
        # Gradients flow back to stem, helping backbone learn perception features
        self.aux_decoder = None
        self.semantic_head = None
        self.depth_head = None

        if use_depth_head or use_segmentation_head:
            self.aux_decoder = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        if use_segmentation_head:
            self.semantic_head = nn.Conv2d(128, num_segmentation_classes, kernel_size=1)
        if use_depth_head:
            self.depth_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, _, H, W = camera.shape

        # Vision: stem → feature map, then global pool → (B, 512)
        fmap = self.stem(camera)                          # (B, 512, h, w)
        vis = self.global_pool(fmap).flatten(1)           # (B, 512)

        # FiLM conditioning on command
        command_features = self.command_embedding(command)
        scale = torch.sigmoid(self.film_scale(command_features))
        bias = self.film_bias(command_features)
        vis = vis * scale + bias

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Auxiliary decoder: shared features for planner context + aux heads
        # Detach fmap before aux_decoder so aux gradients don't disrupt the
        # already-trained stem.  Aux heads still learn to decode stem features,
        # and aux_feat (pooled from aux) still conditions the planner.
        aux = None
        aux_feat = None
        if self.aux_decoder is not None and (self.depth_head is not None or self.semantic_head is not None):
            aux = self.aux_decoder(fmap.detach())          # (B, 128, h, w)
            aux_feat = self.global_pool(aux).flatten(1)   # (B, 128) — for context fusion

        device = camera.device

        # Fusion: concat [vis, hist, cmd, aux_feat] → context vector
        # aux_feat is real when aux heads are active, zeros otherwise (keeps shapes consistent for resume)
        if aux_feat is None:
            aux_feat = torch.zeros(B, self.aux_feat_dim, device=device)
        fusion_parts = [vis, history_features, command_features, aux_feat]
        ctx = self.context_mlp(torch.cat(fusion_parts, dim=1))

        # Autoregressive GRU decoder with per-step ctx injection
        h_dec = ctx
        ctx_step = self.ctx_proj(ctx)                      # (B, 4) — projected ctx for each step
        step_input = torch.zeros(B, 4, device=device)
        step_outputs = []
        for _ in range(self.future_steps):
            gru_input = torch.cat([step_input, ctx_step], dim=1)  # (B, 8)
            h_dec = self.trajectory_decoder_cell(gru_input, h_dec)
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

        # Auxiliary heads: reuse aux features computed above
        if aux is not None:
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
                # No sigmoid — directly regress log1p-normalized depth target
                outputs["depth"] = depth_pred

        return outputs
