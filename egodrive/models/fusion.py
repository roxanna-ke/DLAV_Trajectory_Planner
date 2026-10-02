from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """History-conditioned attention pooling over a CNN feature map."""

    def __init__(self, query_dim: int, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = feat_map.shape
        query_proj = self.q_proj(query).unsqueeze(1)
        keys = self.k_proj(feat_map).flatten(2)
        values = self.v_proj(feat_map).flatten(2).transpose(1, 2)
        scores = torch.bmm(query_proj, keys) / (query_proj.size(-1) ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention, values).squeeze(1)
        return self.out_proj(self.norm(context))
