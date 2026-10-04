"""GraphSAGE baseline for Phase 4 PyG graph samples."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .classifier import GraphClassifier


class GraphSAGEClassifier(nn.Module):
    """Two-layer GraphSAGE classifier."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList(
            [SAGEConv(in_channels, hidden_channels), SAGEConv(hidden_channels, hidden_channels)]
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