from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryPerceptionHeads(nn.Module):
    """Optional depth and semantic heads used by the milestone-2 experiment."""

    def __init__(self, layer3_dim: int, layer4_dim: int, num_classes: int = 15) -> None:
        super().__init__()
        self.layer3_projection = nn.Sequential(
            nn.Conv2d(layer3_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer4_projection = nn.Sequential(
            nn.Conv2d(layer4_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Conv2d(128, 1, kernel_size=1)
        self.semantic_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(
        self,
        layer3: torch.Tensor,
        layer4: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        low = self.layer3_projection(layer3)
        high = self.layer4_projection(layer4)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.decoder(torch.cat([low, high], dim=1))
        depth = torch.sigmoid(F.interpolate(self.depth_head(fused), size=output_size, mode="bilinear", align_corners=False))
        semantic = F.interpolate(self.semantic_head(fused), size=output_size, mode="bilinear", align_corners=False)
        return {"depth": depth, "segmentation_logits": semantic}
