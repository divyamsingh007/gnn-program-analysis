"""Deterministic, inspectable node features for program graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx


UNKNOWN_KIND = "<unknown>"


def node_kind(attributes: dict[str, Any]) -> str:
    """Choose a stable categorical node label across all program views."""

    return str(
        attributes.get("kind")
        or attributes.get("ast_kind")
        or attributes.get("block_type")
        or UNKNOWN_KIND
    )


@dataclass(frozen=True)
class NodeFeatureEncoder:
    """A deterministic one-hot encoder for program-graph node categories."""

    vocabulary: tuple[str, ...]

    @property
    def feature_size(self) -> int:
        return len(self.vocabulary) + 1

    def encode_attributes(self, attributes: dict[str, Any]) -> list[float]:
        """Return a one-hot feature vector, reserving index zero for unknowns."""

        vector = [0.0] * self.feature_size
        try:
            index = self.vocabulary.index(node_kind(attributes)) + 1
        except ValueError:
            index = 0
        vector[index] = 1.0
        return vector


def fit_node_feature_encoder(graphs: Iterable[nx.Graph] | nx.Graph) -> NodeFeatureEncoder:
    """Create a stable vocabulary from one graph or an iterable of graphs."""

    graph_list = [graphs] if isinstance(graphs, nx.Graph) else list(graphs)
    vocabulary = sorted(
        {node_kind(attributes) for graph in graph_list for _, attributes in graph.nodes(data=True)}
    )
    return NodeFeatureEncoder(tuple(vocabulary))


def encode_graph_nodes(
    graph: nx.Graph,
    encoder: NodeFeatureEncoder,
    *,
    attribute: str = "features",
    copy: bool = True,
) -> nx.Graph:
    """Attach one-hot vectors to graph nodes and return the encoded graph."""

    encoded_graph = graph.copy() if copy else graph
    for _, attributes in encoded_graph.nodes(data=True):
        attributes[attribute] = encoder.encode_attributes(attributes)
    encoded_graph.graph["feature_size"] = encoder.feature_size
    encoded_graph.graph["feature_vocabulary"] = encoder.vocabulary
    return encoded_graph
