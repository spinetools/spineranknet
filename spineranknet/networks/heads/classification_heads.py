from __future__ import annotations

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F


__all__ = ["MLPHead", "KANHead", "TransformerHead", "create_head"]


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class KANHead(nn.Module):
    """Lightweight KAN-style classification head with a softmax grid."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 grid_size: int = 5) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.fc1 = nn.Linear(in_dim, hidden_dim * grid_size)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = x.view(x.size(0), -1, self.grid_size)
        x = F.softmax(x, dim=-1)
        x = (x * torch.arange(self.grid_size, device=x.device).float()).sum(dim=-1)
        return self.fc2(x)


class TransformerHead(nn.Module):
    """Single-layer Transformer-encoder classification head."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                      dim_feedforward=hidden_dim * 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.fc(x)


def create_head(head_type: str, in_dim: int, out_dim: int, hidden_dim: int = 256) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    elif head_type == "mlp":
        return MLPHead(in_dim, out_dim, hidden_dim)
    elif head_type == "kan":
        return KANHead(in_dim, out_dim, hidden_dim)
    elif head_type == "transformer":
        return TransformerHead(in_dim, out_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
