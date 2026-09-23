from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn as nn


class MLP(nn.Module):
    """Optional MLP head on top of backbone features."""

    def __init__(
            self,
            dims: List[int],
            activation: str = 'relu',
            dropout_rates: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.dims = dims
        self.activation = activation

        if dropout_rates is None:
            dropout_rates = [0.2] * (len(dims) - 2)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                if dropout_rates and i < len(dropout_rates):
                    layers.append(nn.Dropout(dropout_rates[i]))

                if activation.lower() == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation.lower() == 'gelu':
                    layers.append(nn.GELU())
                elif activation.lower() == 'silu':
                    layers.append(nn.SiLU())
                elif activation.lower() == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.01, inplace=True))

                layers.append(nn.BatchNorm1d(dims[i + 1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
