import networkx as nx
import pytest


pytest.importorskip("clang.cindex", reason="libclang is required for graph tests")

from src.extraction.ast_extractor import extract_ast  # noqa: E402
from src.extraction.cfg_extractor import extract_cfg  # noqa: E402
from src.extraction.clang_utils import DEFAULT_SAMPLE_FILE  # noqa: E402
from src.extraction.dataflow_extractor import extract_dataflow  # noqa: E402
from src.features.node_features import (  # noqa: E402
    encode_graph_nodes,
    fit_node_feature_encoder,
)
from src.graphs.ast_graph import build_ast_graph  # noqa: E402
from src.graphs.cfg_graph import build_cfg_graph  # noqa: E402
from src.graphs.dataflow_graph import build_dataflow_graph  # noqa: E402
from src.graphs.graph_builder import build_multiview_graph  # noqa: E402


@pytest.fixture()
def extraction_payloads() -> tuple[dict, dict, dict]:
    return (
        extract_ast(DEFAULT_SAMPLE_FILE),
        extract_cfg(DEFAULT_SAMPLE_FILE),
        extract_dataflow(DEFAULT_SAMPLE_FILE),
    )


def test_view_builders_preserve_phase_two_nodes_and_edges(extraction_payloads: tuple[dict, dict, dict]) -> None:
    ast_payload, cfg_payload, dataflow_payload = extraction_payloads
    graphs = (
        (build_ast_graph(ast_payload), ast_payload),
        (build_cfg_graph(cfg_payload), cfg_payload),
        (build_dataflow_graph(dataflow_payload), dataflow_payload),
    )

    for graph, payload in graphs:
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == len(payload["nodes"])
        assert graph.number_of_edges() == len(payload["edges"])


def test_multiview_graph_namespaces_views_and_adds_alignment_edges(
    extraction_payloads: tuple[dict, dict, dict],
) -> None:
    graph = build_multiview_graph(*extraction_payloads)

    assert isinstance(graph, nx.MultiDiGraph)
    assert {attributes["view"] for _, attributes in graph.nodes(data=True)} == {
        "ast",
        "cfg",
        "dataflow",
    }
    assert any(
        attributes["relation"] in {"ast_to_cfg", "ast_to_dataflow"}
        for _, _, attributes in graph.edges(data=True)
    )


def test_node_features_are_one_hot_and_do_not_mutate_by_default(
    extraction_payloads: tuple[dict, dict, dict],
) -> None:
    graph = build_multiview_graph(*extraction_payloads)
    encoder = fit_node_feature_encoder(graph)
    encoded_graph = encode_graph_nodes(graph, encoder)

    assert "features" not in next(iter(graph.nodes(data=True)))[1]
    assert encoded_graph.graph["feature_size"] == encoder.feature_size
    for _, attributes in encoded_graph.nodes(data=True):
        vector = attributes["features"]
        assert len(vector) == encoder.feature_size
        assert sum(vector) == 1.0
