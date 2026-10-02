from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34

from .auxiliary import AuxiliaryPerceptionHeads
from .decoders import AutoregressiveGRUDecoder, DirectResidualDecoder
from .fusion import SpatialAttention


class EgoDrivePlanner(nn.Module):
    """Configurable planner with the spatial-attention final as its default."""

    def __init__(
        self,
        *,
        pretrained_backbone: bool = True,
        image_feature_dim: int = 256,
        history_hidden_dim: int = 128,
        command_feature_dim: int = 32,
        history_layers: int = 2,
        dropout: float = 0.1,
        future_steps: int = 60,
        use_command: bool = False,
        fusion: str = "history_conditioned_spatial_attention",
        decoder: str = "autoregressive_gru",
        use_depth_head: bool = False,
        use_segmentation_head: bool = False,
        num_segmentation_classes: int = 15,
    ) -> None:
        super().__init__()
        if fusion not in {"history_conditioned_spatial_attention", "command_film_concat", "global_concat"}:
            raise ValueError(f"Unsupported fusion: {fusion}")
        if decoder not in {"autoregressive_gru", "direct_residual"}:
            raise ValueError(f"Unsupported decoder: {decoder}")
        if fusion == "command_film_concat" and not use_command:
            raise ValueError("command_film_concat requires use_command=True")

        self.future_steps = future_steps
        self.use_command = use_command
        self.fusion = fusion
        self.decoder_name = decoder
        self.vision_dim = 512
        self.layer3_dim = 256

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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        history_input_dim = 4 if not use_command else 4
        self.history_encoder = nn.GRU(
            input_size=history_input_dim,
            hidden_size=history_hidden_dim,
            num_layers=history_layers,
            batch_first=True,
            dropout=dropout if history_layers > 1 else 0.0,
        )

        self.command_embedding = nn.Embedding(3, command_feature_dim) if use_command else None
        self.spatial_attention = (
            SpatialAttention(history_hidden_dim, self.vision_dim, image_feature_dim)
            if fusion == "history_conditioned_spatial_attention"
            else None
        )
        self.film_scale = nn.Linear(command_feature_dim, self.vision_dim) if use_command else None
        self.film_bias = nn.Linear(command_feature_dim, self.vision_dim) if use_command else None

        visual_dim = image_feature_dim if self.spatial_attention is not None else self.vision_dim
        fusion_in = visual_dim + history_hidden_dim + (command_feature_dim if use_command else 0)
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        if decoder == "autoregressive_gru":
            self.decoder = AutoregressiveGRUDecoder(image_feature_dim, future_steps)
        else:
            self.decoder = DirectResidualDecoder(image_feature_dim, future_steps)

        self.auxiliary_heads = (
            AuxiliaryPerceptionHeads(self.layer3_dim, self.vision_dim, num_segmentation_classes)
            if use_depth_head or use_segmentation_head
            else None
        )
        self.use_depth_head = use_depth_head
        self.use_segmentation_head = use_segmentation_head

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = camera.shape
        layer3 = self.stem_layer3(camera)
        layer4 = self.stem_layer4(layer3)
        _, hidden = self.history_encoder(history)
        history_features = hidden[-1]

        command_features = None
        if self.use_command:
            if command is None:
                raise ValueError("command is required for this model configuration")
            command_features = self.command_embedding(command)

        if self.spatial_attention is not None:
            visual_features = self.spatial_attention(history_features, layer4)
        else:
            visual_features = self.global_pool(layer4).flatten(1)

        if command_features is not None and self.film_scale is not None:
            visual_features = visual_features * torch.sigmoid(self.film_scale(command_features)) + self.film_bias(command_features)

        fusion_features = [visual_features, history_features]
        if command_features is not None:
            fusion_features.append(command_features)
        context = self.context_mlp(torch.cat(fusion_features, dim=1))
        raw_prediction = self.decoder(context)
        trajectory = torch.cat(
            [torch.cumsum(raw_prediction[..., :2], dim=1), raw_prediction[..., 2:]],
            dim=-1,
        )
        outputs = {"trajectory": trajectory}

        if self.auxiliary_heads is not None:
            aux = self.auxiliary_heads(layer3, layer4, (height, width))
            if self.use_depth_head:
                outputs["depth"] = aux["depth"]
            if self.use_segmentation_head:
                outputs["segmentation_logits"] = aux["segmentation_logits"]
        return outputs
