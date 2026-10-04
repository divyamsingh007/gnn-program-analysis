"""Convert Phase 2 data-flow payloads into NetworkX graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .graph_utils import build_directed_graph


def build_dataflow_graph(dataflow_payload: dict[str, Any]) -> nx.DiGraph:
    """Build a directed def-use graph retaining variable identities."""

    graph = build_directed_graph(dataflow_payload, "dataflow")
    graph.graph["variables"] = dataflow_payload.get("variables", [])
    return graph
