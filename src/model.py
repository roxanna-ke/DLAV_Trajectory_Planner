from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torchvision.models import ResNet34_Weights, resnet34


class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        x = self.maxpool(stem)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }


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


class LightweightFusion(nn.Module):
    def __init__(
        self,
        *,
        history_hidden_dim: int,
        command_feature_dim: int,
        image_feature_dim: int,
        fusion_dim: int,
        dropout: float,
        use_depth_token: bool,
        use_segmentation_token: bool,
    ) -> None:
        super().__init__()
        self.use_depth_token = use_depth_token
        self.use_segmentation_token = use_segmentation_token

        self.layer3_scale = nn.Linear(command_feature_dim, 256)
        self.layer3_bias = nn.Linear(command_feature_dim, 256)
        self.layer4_scale = nn.Linear(command_feature_dim, 512)
        self.layer4_bias = nn.Linear(command_feature_dim, 512)

        self.layer3_reduce = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.layer4_reduce = nn.Conv2d(512, 64, kernel_size=1, bias=False)
        self.layer3_pool = nn.AdaptiveAvgPool2d((4, 6))
        self.layer4_pool = nn.AdaptiveAvgPool2d((2, 3))

        visual_dim = 32 * 4 * 6 + 64 * 2 * 3
        self.visual_mlp = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        fusion_in = image_feature_dim + history_hidden_dim + command_feature_dim
        if use_depth_token:
            fusion_in += image_feature_dim
        if use_segmentation_token:
            fusion_in += image_feature_dim

        hidden_dim = max(fusion_dim, 256)
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        *,
        features: dict[str, torch.Tensor],
        history_features: torch.Tensor,
        command_features: torch.Tensor,
        depth_token: torch.Tensor | None = None,
        segmentation_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer3_scale = torch.sigmoid(self.layer3_scale(command_features)).unsqueeze(-1).unsqueeze(-1)
        layer3_bias = self.layer3_bias(command_features).unsqueeze(-1).unsqueeze(-1)
        layer4_scale = torch.sigmoid(self.layer4_scale(command_features)).unsqueeze(-1).unsqueeze(-1)
        layer4_bias = self.layer4_bias(command_features).unsqueeze(-1).unsqueeze(-1)

        layer3_features = features["layer3"] * layer3_scale + layer3_bias
        layer4_features = features["layer4"] * layer4_scale + layer4_bias

        layer3_grid = self.layer3_pool(self.layer3_reduce(layer3_features)).flatten(1)
        layer4_grid = self.layer4_pool(self.layer4_reduce(layer4_features)).flatten(1)
        visual_features = self.visual_mlp(torch.cat([layer3_grid, layer4_grid], dim=1))

        fused_features = [visual_features, history_features, command_features]
        if self.use_depth_token and depth_token is not None:
            fused_features.append(depth_token)
        if self.use_segmentation_token and segmentation_token is not None:
            fused_features.append(segmentation_token)

        return self.context_mlp(torch.cat(fused_features, dim=1))


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

        self.backbone = ResNet34Backbone(pretrained=pretrained_backbone)

        gru_dropout = dropout if history_layers > 1 else 0.0
        self.history_encoder = nn.GRU(
            input_size=3,
            hidden_size=history_hidden_dim,
            num_layers=history_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.command_embedding = nn.Embedding(3, command_feature_dim)
        self.fusion = LightweightFusion(
            history_hidden_dim=history_hidden_dim,
            command_feature_dim=command_feature_dim,
            image_feature_dim=image_feature_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_depth_token=use_depth_head,
            use_segmentation_token=use_segmentation_head,
        )

        self.trajectory_decoder = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, future_steps * 3),
        )

        self.depth_decoder = DensePredictionDecoder(1) if use_depth_head else None
        self.segmentation_decoder = (
            DensePredictionDecoder(num_segmentation_classes)
            if use_segmentation_head
            else None
        )
        self.depth_token_encoder = (
            AuxiliaryTokenEncoder(in_channels=33, token_dim=image_feature_dim)
            if use_depth_head
            else None
        )
        self.segmentation_token_encoder = (
            AuxiliaryTokenEncoder(
                in_channels=32 + num_segmentation_classes,
                token_dim=image_feature_dim,
            )
            if use_segmentation_head
            else None
        )

    def forward(
        self,
        camera: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(camera)
        depth_token = None
        segmentation_token = None

        _, history_hidden = self.history_encoder(history)
        history_features = history_hidden[-1]
        command_features = self.command_embedding(command)

        outputs: dict[str, torch.Tensor] = {}
        if self.depth_decoder is not None:
            depth_prediction, depth_features = self.depth_decoder(
                features,
                return_features=True,
            )
            outputs["depth"] = depth_prediction
            if self.depth_token_encoder is not None:
                depth_token = self.depth_token_encoder(
                    torch.cat([depth_features, depth_prediction], dim=1)
                )

        if self.segmentation_decoder is not None:
            segmentation_logits, segmentation_features = self.segmentation_decoder(
                features,
                return_features=True,
            )
            outputs["segmentation_logits"] = segmentation_logits
            if self.segmentation_token_encoder is not None:
                segmentation_probabilities = torch.softmax(segmentation_logits, dim=1)
                segmentation_token = self.segmentation_token_encoder(
                    torch.cat(
                        [segmentation_features, segmentation_probabilities],
                        dim=1,
                    )
                )

        fused_features = self.fusion(
            features=features,
            history_features=history_features,
            command_features=command_features,
            depth_token=depth_token,
            segmentation_token=segmentation_token,
        )
        trajectory_deltas = self.trajectory_decoder(fused_features)
        trajectory_deltas = trajectory_deltas.view(camera.size(0), self.future_steps, 3)
        trajectory = torch.cumsum(trajectory_deltas, dim=1)

        outputs["trajectory"] = trajectory
        return outputs
