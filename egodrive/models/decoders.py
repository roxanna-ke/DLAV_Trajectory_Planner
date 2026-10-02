from __future__ import annotations

import torch
import torch.nn as nn


class AutoregressiveGRUDecoder(nn.Module):
    """Roll out trajectory deltas one step at a time."""

    def __init__(self, context_dim: int, future_steps: int, output_dim: int = 4) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.cell = nn.GRUCell(input_size=output_dim, hidden_size=context_dim)
        self.output_head = nn.Linear(context_dim, output_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = context
        step_input = context.new_zeros((context.size(0), self.output_head.out_features))
        outputs = []
        for _ in range(self.future_steps):
            hidden = self.cell(step_input, hidden)
            step_output = self.output_head(hidden)
            outputs.append(step_output)
            step_input = step_output
        return torch.stack(outputs, dim=1)


class DirectResidualDecoder(nn.Module):
    """Legacy non-autoregressive decoder retained for historical experiments."""

    def __init__(self, context_dim: int, future_steps: int, output_dim: int = 4) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, future_steps * output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(context).view(context.size(0), self.future_steps, -1)
