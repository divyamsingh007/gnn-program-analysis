"""Build a single multi-view program graph from Phase 2 representations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from .ast_graph import build_ast_graph
from .cfg_graph import build_cfg_graph
from .dataflow_graph import build_dataflow_graph


VIEW_BUILDERS = {
    "ast": build_ast_graph,
    "cfg": build_cfg_graph,
    "dataflow": build_dataflow_graph,
}


def _location_key(attributes: dict[str, Any]) -> tuple[str, int, int] | None:
    location = attributes.get("location")
    if not isinstance(location, dict):
        return None
    file_name = location.get("file")
    line = location.get("line")
    column = location.get("column")
    if file_name is None or line is None or column is None:
        return None
    return (str(Path(file_name).resolve()), int(line), int(column))


def _add_view(
    merged_graph: nx.MultiDiGraph,
    graph: nx.DiGraph,
    view: str,
) -> None:
    """Copy a namespaced view and its internal relationships into the graph."""

    for node_id, attributes in graph.nodes(data=True):
        merged_graph.add_node(
            (view, node_id),
            **attributes,
            view=view,
            original_id=node_id,
        )
    for source, target, attributes in graph.edges(data=True):
        merged_graph.add_edge(
            (view, source),
            (view, target),
            **attributes,
            relation=attributes.get("type", "internal"),
            view=view,
        )


def _connect_by_location(
    merged_graph: nx.MultiDiGraph,
    source_view: str,
    target_view: str,
    relation: str,
) -> None:
    """Add inter-view edges where nodes share an exact source location."""

    target_nodes: dict[tuple[str, int, int], list[tuple[str, Any]]] = defaultdict(list)
    for node_id, attributes in merged_graph.nodes(data=True):
        if attributes["view"] != target_view:
            continue
        location = _location_key(attributes)
        if location is not None:
            target_nodes[location].append(node_id)

    for node_id, attributes in list(merged_graph.nodes(data=True)):
        if attributes["view"] != source_view:
            continue
        location = _location_key(attributes)
        if location is None:
            continue
        for target_id in target_nodes.get(location, []):
            merged_graph.add_edge(
                node_id,
                target_id,
                type="alignment",
                relation=relation,
                view="cross_view",
            )


def build_multiview_graph(
    ast_payload: dict[str, Any],
    cfg_payload: dict[str, Any],
    dataflow_payload: dict[str, Any],
) -> nx.MultiDiGraph:
    """Merge AST, CFG, and data-flow views into one relation-aware graph.

    Node identifiers are namespaced as ``(view, original_id)``. Exact source
    location matches create ``ast_to_cfg`` and ``ast_to_dataflow`` alignment
    edges; no heuristic or semantic links are inferred when locations differ.
    """

    payloads = {
        "ast": ast_payload,
        "cfg": cfg_payload,
        "dataflow": dataflow_payload,
    }
    source_files = {str(Path(payload["source_file"]).resolve()) for payload in payloads.values()}
    if len(source_files) != 1:
        raise ValueError("All graph views must originate from the same source file")

    graph = nx.MultiDiGraph(
        representation="multiview",
        schema_version="1.0",
        source_file=source_files.pop(),
        views=tuple(payloads),
    )
    for view, payload in payloads.items():
        _add_view(graph, VIEW_BUILDERS[view](payload), view)

    _connect_by_location(graph, "ast", "cfg", "ast_to_cfg")
    _connect_by_location(graph, "ast", "dataflow", "ast_to_dataflow")
    return graph
