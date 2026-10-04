"""Convert Phase 2 AST payloads into NetworkX graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .graph_utils import build_directed_graph


def build_ast_graph(ast_payload: dict[str, Any]) -> nx.DiGraph:
    """Build a directed AST graph with ``ast_child`` edge attributes."""

    return build_directed_graph(ast_payload, "ast")
