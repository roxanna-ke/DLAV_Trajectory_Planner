from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torchvision.models import ResNet34_Weights, resnet34


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip_projection = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        self.block = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch_f.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip = self.skip_projection(skip)
        return self.block(torch.cat([x, skip], dim=1))


class DensePredictionDecoder(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.up4 = UpBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.up3 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.up2 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.refine_112 = ConvBlock(in_channels=64, out_channels=64)
        self.refine_224 = ConvBlock(in_channels=64, out_channels=32)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.up4(features["layer4"], features["layer3"])
        x = self.up3(x, features["layer2"])
        x = self.up2(x, features["layer1"])
        x = torch_f.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine_112(x)
        x = torch_f.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        refined = self.refine_224(x)
        prediction = self.head(refined)
        if return_features:
            return prediction, refined
        return prediction


class AuxiliaryTokenEncoder(nn.Module):
    def __init__(self, in_channels: int, token_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(64, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).flatten(1)
        return self.projection(x)


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
        use_depth_token: bool = False,
        use_segmentation_token: bool = False,
        num_segmentation_classes: int = 15,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.use_depth_head = use_depth_head
        self.use_segmentation_head = use_segmentation_head
        self.use_depth_token = use_depth_token and use_depth_head
        self.use_segmentation_token = use_segmentation_token and use_segmentation_head

        # Vision encoder: ResNet-34 with global average pooling → 512-d
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained_backbone else None)
        backbone.fc = nn.Identity()
        self.vision_encoder = backbone
        self.vision_dim = 512

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

        # Fusion MLP: concat [vis, hist, cmd, (tokens)] → context
        fusion_in = self.vision_dim + history_hidden_dim + command_feature_dim
        if self.use_depth_token:
            fusion_in += image_feature_dim
        if self.use_segmentation_token:
            fusion_in += image_feature_dim

        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, image_feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(image_feature_dim * 2, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Autoregressive GRU decoder
        self.trajectory_decoder_cell = nn.GRUCell(
            input_size=4, hidden_size=image_feature_dim,
        )
        self.trajectory_output_head = nn.Linear(image_feature_dim, 4)

        # Auxiliary decoders (need multi-scale features from backbone)
        # Multi-scale features are extracted via _get_aux_features which walks
        # the vision_encoder backbone layers manually.

        self.depth_decoder = DensePredictionDecoder(1) if use_depth_head else None
        self.segmentation_decoder = (
            DensePredictionDecoder(num_segmentation_classes)
            if use_segmentation_head
            else None
        )
        self.depth_token_encoder = (
            AuxiliaryTokenEncoder(in_channels=33, token_dim=image_feature_dim)
            if self.use_depth_token
            else None
        )
        self.segmentation_token_encoder = (
            AuxiliaryTokenEncoder(
                in_channels=32 + num_segmentation_classes,
                token_dim=image_feature_dim,
            )
            if self.use_segmentation_token
            else None
        )

    def _get_aux_features(self, camera: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features from vision_encoder backbone for aux decoders."""
        backbone = self.vision_encoder
        x = backbone.conv1(camera)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        layer1 = backbone.layer1(x)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Vision: global average pooling → (B, 512)
        vis = self.vision_encoder(camera)

        # FiLM conditioning on command
        command_features = self.command_embedding(command)
        scale = torch.sigmoid(self.film_scale(command_features))
        bias = self.film_bias(command_features)
        vis = vis * scale + bias

        # History encoding
        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]

        # Auxiliary tasks (need multi-scale features, detached from backbone)
        depth_token = None
        segmentation_token = None
        outputs: dict[str, torch.Tensor] = {}

        if self.depth_decoder is not None:
            with torch.no_grad():
                aux_features = {k: v.detach() for k, v in self._get_aux_features(camera).items()}
            depth_prediction, depth_features = self.depth_decoder(
                aux_features, return_features=True,
            )
            depth_prediction = torch.sigmoid(depth_prediction)
            outputs["depth"] = depth_prediction
            if self.depth_token_encoder is not None:
                depth_token = self.depth_token_encoder(
                    torch.cat([depth_features.detach(), depth_prediction.detach()], dim=1)
                )

        if self.segmentation_decoder is not None:
            with torch.no_grad():
                aux_features = {k: v.detach() for k, v in self._get_aux_features(camera).items()}
            segmentation_logits, segmentation_features = self.segmentation_decoder(
                aux_features, return_features=True,
            )
            outputs["segmentation_logits"] = segmentation_logits
            if self.segmentation_token_encoder is not None:
                segmentation_probabilities = torch.softmax(segmentation_logits.detach(), dim=1)
                segmentation_token = self.segmentation_token_encoder(
                    torch.cat(
                        [segmentation_features.detach(), segmentation_probabilities],
                        dim=1,
                    )
                )

        # Fusion: concat all features → context vector
        fused = [vis, history_features, command_features]
        if self.use_depth_token and depth_token is not None:
            fused.append(depth_token)
        if self.use_segmentation_token and segmentation_token is not None:
            fused.append(segmentation_token)
        ctx = self.context_mlp(torch.cat(fused, dim=1))

        # Autoregressive GRU decoder
        batch_size = camera.size(0)
        device = camera.device
        h_dec = ctx
        step_input = torch.zeros(batch_size, 4, device=device)
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
        outputs["trajectory"] = trajectory
        return outputs
