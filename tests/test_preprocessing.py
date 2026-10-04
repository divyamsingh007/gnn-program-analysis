import networkx as nx
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.datasets.dataset_loader import create_graph_loader  # noqa: E402
from src.datasets.preprocessing import (  # noqa: E402
    batch_graphs_to_data,
    multiview_graph_to_data,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        representation="multiview",
        schema_version="1.0",
        source_file="/tmp/sample.c",
        feature_vocabulary=("call", "return"),
    )
    graph.add_node(("ast", 2), view="ast", features=[1.0, 0.0])
    graph.add_node(("cfg", 1), view="cfg", features=[0.0, 1.0])
    graph.add_edge(("ast", 2), ("cfg", 1), relation="ast_to_cfg")
    return graph


def test_multiview_graph_conversion_is_deterministic_and_preserves_relations():
    data = multiview_graph_to_data(_graph(), label=1)

    assert data.x.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert data.edge_index.tolist() == [[0], [1]]
    assert data.edge_type.tolist() == [0]
    assert data.relation_vocabulary == ("ast_to_cfg",)
    assert data.y.tolist() == [1]
    assert data.node_ids == [("ast", 2), ("cfg", 1)]


def test_batch_conversion_shares_relation_vocabulary_and_loader_batches():
    first = _graph()
    second = _graph()
    second.add_edge(("cfg", 1), ("ast", 2), relation="internal")

    samples = batch_graphs_to_data([first, second])
    assert samples[0].relation_vocabulary == samples[1].relation_vocabulary
    loader = create_graph_loader(samples, batch_size=2)
    batch = next(iter(loader))
    assert batch.num_graphs == 2
    assert batch.x.shape == (4, 2)
