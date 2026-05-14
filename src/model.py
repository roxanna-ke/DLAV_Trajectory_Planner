from __future__ import annotations

import math

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
        command_feature_dim: int = 8,
        history_layers: int = 2,
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        dropout: float = 0.1,
        future_steps: int = 60,
        use_segmentation_head: bool = True,
        use_depth_head: bool = False,
        num_segmentation_classes: int = 15,
        use_layer3_spatial_pooling: bool = False,
        layer3_spatial_scale: float = 0.1,
        layer3_spatial_grid_height: int = 4,
        layer3_spatial_grid_width: int = 6,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.use_segmentation_head = use_segmentation_head
        self.use_depth_head = use_depth_head
        self.use_layer3_spatial_pooling = use_layer3_spatial_pooling
        self.spatial_feature_dim = 256

        # Vision encoder: ResNet-34 — split stem to access layer3 output for spatial pooling
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained_backbone else None)
        self.stem_layer3 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        self.stem_layer4 = backbone.layer4
        self.layer3_dim = 256
        self.vision_dim = 512
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Layer3 spatial grid pooling: preserve spatial layout as a separate fusion input
        self.layer3_spatial_pool = None
        self.layer3_spatial_projection = None
        if use_layer3_spatial_pooling:
            self.layer3_spatial_pool = nn.AdaptiveAvgPool2d(
                (layer3_spatial_grid_height, layer3_spatial_grid_width)
            )
            spatial_flat_dim = self.layer3_dim * layer3_spatial_grid_height * layer3_spatial_grid_width
            self.layer3_spatial_projection = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(spatial_flat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, self.spatial_feature_dim),
                nn.ReLU(inplace=True),
            )

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

        # Base fusion MLP: concat [global vis, hist, cmd] → context
        base_fusion_in = self.vision_dim + history_hidden_dim + command_feature_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(base_fusion_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.LayerNorm(image_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_delta_projection = None
        self.spatial_gate = None
        if use_layer3_spatial_pooling:
            self.spatial_delta_projection = nn.Sequential(
                nn.Linear(self.spatial_feature_dim, image_feature_dim),
                nn.LayerNorm(image_feature_dim),
                nn.ReLU(inplace=True),
            )
            self.spatial_gate = nn.Linear(base_fusion_in, image_feature_dim)
            initial_scale = min(max(layer3_spatial_scale, 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.spatial_gate.weight)
            nn.init.constant_(
                self.spatial_gate.bias,
                math.log(initial_scale / (1.0 - initial_scale)),
            )

        # Autoregressive GRU decoder
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=4, hidden_size=image_feature_dim,
        )
        self.trajectory_output_head = nn.Linear(image_feature_dim, 4)

        # Lightweight shared auxiliary decoder (like notebook)
        # Gradients flow back to stem, helping backbone learn perception features
        self.aux_decoder = None
        self.semantic_head = None
        self.depth_head = None

        if use_segmentation_head or use_depth_head:
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

        # Vision: stem split — layer3 for spatial, layer4 for global pool + aux
        fmap_layer3 = self.stem_layer3(camera)              # (B, 256, 14, 21)
        fmap = self.stem_layer4(fmap_layer3)                # (B, 512, 7, 10)
        vis_global = self.global_pool(fmap).flatten(1)      # (B, 512)

        if (
            self.use_layer3_spatial_pooling
            and self.layer3_spatial_pool is not None
            and self.layer3_spatial_projection is not None
        ):
            vis_spatial = self.layer3_spatial_projection(
                self.layer3_spatial_pool(fmap_layer3)
            )                                              # (B, 256)
        else:
            vis_spatial = torch.zeros(
                B,
                self.spatial_feature_dim,
                device=camera.device,
                dtype=vis_global.dtype,
            )

        command_features = self.command_embedding(command)

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Base fusion: concat [global vis, hist, cmd] → context vector
        ctx_base_input = torch.cat(
            [vis_global, history_features, command_features],
            dim=1,
        )
        ctx_base = self.context_mlp(ctx_base_input)
        ctx = ctx_base

        # Gated residual spatial fusion: ctx = ctx_base + gate * spatial_delta
        if (
            self.use_layer3_spatial_pooling
            and self.spatial_delta_projection is not None
            and self.spatial_gate is not None
        ):
            spatial_delta = self.spatial_delta_projection(vis_spatial)
            gate = torch.sigmoid(self.spatial_gate(ctx_base_input))
            ctx = ctx_base + gate * spatial_delta

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

        # Auxiliary heads: shared feature map → lightweight decoder → upsample
        # Gradients flow back to stem, which is the key mechanism for aux to help planning
        if self.aux_decoder is not None:
            aux = self.aux_decoder(fmap)                  # (B, 128, h, w)
            if self.semantic_head is not None:
                semantic_logits = self.semantic_head(aux)  # (B, C, h, w)
                semantic_logits = torch_f.interpolate(
                    semantic_logits, size=(H, W), mode="bilinear", align_corners=False,
                )
                outputs["segmentation_logits"] = semantic_logits
            if self.depth_head is not None:
                depth_pred = self.depth_head(aux)          # (B, 1, h, w)
                depth_pred = torch_f.interpolate(
                    depth_pred, size=(H, W), mode="bilinear", align_corners=False,
                )
                outputs["depth_prediction"] = depth_pred

        return outputs
