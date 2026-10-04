"""Validation and conversion helpers shared by Phase 3 graph builders."""

from __future__ import annotations

from typing import Any

import networkx as nx


class GraphPayloadError(ValueError):
    """Raised when an extraction payload cannot be converted into a graph."""


def validate_payload(payload: dict[str, Any], representation: str) -> None:
    """Validate the common Phase 2 extraction payload contract."""

    required_keys = {"schema_version", "representation", "source_file", "nodes", "edges"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise GraphPayloadError(f"Missing extraction payload keys: {missing}")
    if payload["representation"] != representation:
        raise GraphPayloadError(
            f"Expected a {representation!r} payload, got {payload['representation']!r}"
        )
    if not isinstance(payload["nodes"], list) or not isinstance(payload["edges"], list):
        raise GraphPayloadError("Payload nodes and edges must be lists")


def build_directed_graph(
    payload: dict[str, Any],
    representation: str,
) -> nx.DiGraph:
    """Convert a validated node/edge payload to a directed NetworkX graph."""

    validate_payload(payload, representation)
    graph = nx.DiGraph(
        representation=representation,
        schema_version=payload["schema_version"],
        source_file=payload["source_file"],
    )

    for node in payload["nodes"]:
        if "id" not in node:
            raise GraphPayloadError("Every node must define an id")
        node_id = node["id"]
        if node_id in graph:
            raise GraphPayloadError(f"Duplicate node id: {node_id!r}")
        graph.add_node(node_id, **{key: value for key, value in node.items() if key != "id"})

    for edge in payload["edges"]:
        if "source" not in edge or "target" not in edge:
            raise GraphPayloadError("Every edge must define source and target")
        source = edge["source"]
        target = edge["target"]
        if source not in graph or target not in graph:
            raise GraphPayloadError(
                f"Edge {source!r} -> {target!r} references a missing node"
            )
        graph.add_edge(
            source,
            target,
            **{key: value for key, value in edge.items() if key not in {"source", "target"}},
        )

    return graph
