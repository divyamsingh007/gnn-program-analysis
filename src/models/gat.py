"""GAT baseline for Phase 4 PyG graph samples."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv

from .classifier import GraphClassifier


class GATClassifier(nn.Module):
    """Two-layer multi-head attention classifier."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        self.convs = nn.ModuleList(
            [
                GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout),
                GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = GraphClassifier(hidden_channels, num_classes, dropout)

    def encode(self, data) -> torch.Tensor:
        x = data.x
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index))
            x = self.dropout(x)
        return x

    def forward(self, data) -> torch.Tensor:
        return self.classifier(self.encode(data), data.batch)