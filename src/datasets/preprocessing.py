"""Convert Phase 3 NetworkX graphs into PyTorch Geometric data objects."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Sequence

import networkx as nx
import torch


@dataclass(frozen=True)
class RelationEncoder:
    """Deterministic encoder for the relation labels on graph edges."""

    vocabulary: tuple[str, ...]

    def encode(self, relation: str) -> int:
        try:
            return self.vocabulary.index(relation)
        except ValueError as exc:
            raise ValueError(f"Unknown edge relation: {relation!r}") from exc


def fit_relation_encoder(graphs: Iterable[nx.MultiDiGraph] | nx.MultiDiGraph) -> RelationEncoder:
    """Build a stable relation vocabulary from one or more Phase 3 graphs."""

    graph_list = [graphs] if isinstance(graphs, nx.MultiDiGraph) else list(graphs)
    relations = sorted(
        {
            str(attributes.get("relation", attributes.get("type", "internal")))
            for graph in graph_list
            for _, _, attributes in graph.edges(data=True)
        }
    )
    return RelationEncoder(tuple(relations))


def _require_pyg() -> type:
    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Phase 4 conversion requires torch-geometric; install requirements.txt first"
        ) from exc
    return Data


def _node_sort_key(node_id: Any) -> str:
    """Provide deterministic ordering for namespaced, potentially mixed IDs."""

    return repr(node_id)


def _feature_vector(node_id: Any, attributes: dict[str, Any], attribute: str) -> list[float]:
    vector = attributes.get(attribute)
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise ValueError(f"Node {node_id!r} is missing a sequence of {attribute!r}")
    if not vector or not all(isinstance(value, Real) for value in vector):
        raise ValueError(f"Node {node_id!r} has an invalid {attribute!r} vector")
    return [float(value) for value in vector]


def multiview_graph_to_data(
    graph: nx.MultiDiGraph,
    *,
    label: int | float | None = None,
    feature_attribute: str = "features",
    relation_encoder: RelationEncoder | None = None,
):
    """Convert an encoded Phase 3 ``MultiDiGraph`` to a PyG ``Data`` object.

    Nodes are assigned deterministic integer indices. Relation labels are
    retained as integer ``edge_type`` values and the vocabulary is attached to
    the returned object for reproducible decoding.
    """

    Data = _require_pyg()
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("Phase 4 conversion requires a networkx.MultiDiGraph")

    ordered_nodes = sorted(graph.nodes(data=True), key=lambda item: _node_sort_key(item[0]))
    node_index = {node_id: index for index, (node_id, _) in enumerate(ordered_nodes)}
    features = [
        _feature_vector(node_id, attributes, feature_attribute)
        for node_id, attributes in ordered_nodes
    ]
    feature_size = len(features[0]) if features else 0
    if any(len(vector) != feature_size for vector in features):
        raise ValueError("All graph nodes must have feature vectors of the same size")

    encoder = relation_encoder or fit_relation_encoder(graph)
    edge_pairs: list[tuple[int, int]] = []
    edge_types: list[int] = []
    for source, target, attributes in graph.edges(data=True):
        relation = str(attributes.get("relation", attributes.get("type", "internal")))
        edge_pairs.append((node_index[source], node_index[target]))
        edge_types.append(encoder.encode(relation))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    if not edge_pairs:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    feature_tensor = (
        torch.tensor(features, dtype=torch.float32)
        if features
        else torch.empty((0, 0), dtype=torch.float32)
    )
    data = Data(
        x=feature_tensor,
        edge_index=edge_index,
        edge_type=torch.tensor(edge_types, dtype=torch.long),
    )
    if label is not None:
        data.y = torch.tensor([label])

    data.node_ids = [node_id for node_id, _ in ordered_nodes]
    data.node_view = [attributes.get("view") for _, attributes in ordered_nodes]
    data.relation_vocabulary = encoder.vocabulary
    data.feature_size = feature_size
    data.feature_vocabulary = graph.graph.get("feature_vocabulary", ())
    data.source_file = graph.graph.get("source_file")
    data.schema_version = graph.graph.get("schema_version", "1.0")
    data.representation = graph.graph.get("representation", "multiview")
    return data


def batch_graphs_to_data(
    graphs: Iterable[nx.MultiDiGraph],
    *,
    labels: Iterable[int | float] | None = None,
) -> list[Any]:
    """Convert graphs using one shared relation vocabulary."""

    graph_list = list(graphs)
    label_list = list(labels) if labels is not None else [None] * len(graph_list)
    if len(label_list) != len(graph_list):
        raise ValueError("labels must contain one value for each graph")
    encoder = fit_relation_encoder(graph_list)
    return [
        multiview_graph_to_data(graph, label=label, relation_encoder=encoder)
        for graph, label in zip(graph_list, label_list)
    ]