import networkx as nx
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.datasets.dataset_loader import create_graph_loader  # noqa: E402
from src.datasets.preprocessing import batch_graphs_to_data  # noqa: E402
from src.evaluation.ablation import run_ablation  # noqa: E402
from src.evaluation.confusion_matrix import build_confusion_matrix  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.models.gcn import GCNClassifier  # noqa: E402


def _loader():
    graphs = []
    for _ in range(2):
        graph = nx.MultiDiGraph(representation="multiview", schema_version="1.0")
        graph.add_node(("ast", 0), view="ast", features=[1.0, 0.0])
        graph.add_node(("cfg", 0), view="cfg", features=[0.0, 1.0])
        graph.add_edge(("ast", 0), ("cfg", 0), relation="ast_to_cfg")
        graphs.append(graph)
    return create_graph_loader(batch_graphs_to_data(graphs, labels=[0, 1]), batch_size=2)


def test_metrics_and_confusion_matrix_are_deterministic():
    metrics = classification_metrics([0, 1, 1], [0, 0, 1], labels_order=[0, 1])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 0], [1, 1]]
    assert build_confusion_matrix([0, 1, 1], [0, 0, 1], labels_order=[0, 1]).tolist() == [
        [1, 0],
        [1, 1],
    ]


def test_model_evaluation_and_ablation_return_predictions():
    loader = _loader()
    result = evaluate_model(GCNClassifier(2, hidden_channels=8), loader)
    assert len(result["labels"]) == len(result["predictions"]) == 2
    results = run_ablation(
        {"gcn": lambda: GCNClassifier(2, hidden_channels=8)},
        loader,
    )
    assert set(results) == {"gcn"}
    assert "f1" in results["gcn"]
