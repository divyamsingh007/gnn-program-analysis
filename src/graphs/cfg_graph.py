"""Convert Phase 2 CFG payloads into NetworkX graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .graph_utils import build_directed_graph


def build_cfg_graph(cfg_payload: dict[str, Any]) -> nx.DiGraph:
    """Build a directed CFG graph retaining its control-flow edge labels."""

    graph = build_directed_graph(cfg_payload, "cfg")
    graph.graph["functions"] = cfg_payload.get("functions", [])
    return graph
