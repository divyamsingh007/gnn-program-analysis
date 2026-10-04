"""Shared graph-level classification utilities."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class GraphClassifier(nn.Module):
    """Pool node embeddings and classify each graph in a batch."""

    def __init__(self, hidden_channels: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        if hidden_channels < 1 or num_classes < 2:
            raise ValueError("hidden_channels must be positive and num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = global_mean_pool(node_embeddings, batch)
        return self.classifier(self.dropout(pooled))