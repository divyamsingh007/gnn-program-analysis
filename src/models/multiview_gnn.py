"""Relation-aware multi-view GNN for Phase 4 graph samples."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from .classifier import GraphClassifier


class RelationConv(MessagePassing):
    """Mean aggregation that preserves the Phase 3 edge relation identity."""

    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__(aggr="mean")
        self.node_projection = nn.Linear(in_channels, out_channels)
        self.relation_embedding = nn.Embedding(max(1, num_relations), out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        return self.propagate(edge_index, x=x, edge_type=edge_type)

    def message(self, x_j: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        return self.node_projection(x_j) + self.relation_embedding(edge_type)


class MultiViewGNN(nn.Module):
    """Relation-aware node encoder and graph-level vulnerability classifier."""

    def __init__(
        self,
        in_channels: int,
        num_relations: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_relations < 0:
            raise ValueError("num_relations cannot be negative")
        self.convs = nn.ModuleList(
            [
                RelationConv(in_channels, hidden_channels, num_relations),
                RelationConv(hidden_channels, hidden_channels, num_relations),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = GraphClassifier(hidden_channels, num_classes, dropout)

    def encode(self, data) -> torch.Tensor:
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long, device=data.x.device)
        x = data.x
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index, edge_type))
            x = self.dropout(x)
        return x

    def forward(self, data) -> torch.Tensor:
        return self.classifier(self.encode(data), data.batch)