"""PyTorch Geometric dataset helpers for Phase 4 program graphs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .preprocessing import batch_graphs_to_data, multiview_graph_to_data


def build_pyg_data(graph: Any, label: int | float | None = None):
    """Convert one encoded Phase 3 graph to a PyG ``Data`` object."""

    return multiview_graph_to_data(graph, label=label)


def build_pyg_dataset(graphs: Sequence[Any], labels: Sequence[int | float] | None = None):
    """Convert a sequence of encoded Phase 3 graphs to PyG data objects."""

    if labels is not None and len(labels) != len(graphs):
        raise ValueError("labels must contain one value for each graph")
    return batch_graphs_to_data(graphs, labels=labels)