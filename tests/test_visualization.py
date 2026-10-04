import networkx as nx
import pytest


pytest.importorskip("matplotlib")

from src.visualization.graph_visualizer import draw_program_graph  # noqa: E402
from src.visualization.result_plots import (  # noqa: E402
    plot_ablation_results,
    plot_confusion_matrix,
)
from src.visualization.training_plots import plot_training_history  # noqa: E402


def test_visualization_helpers_create_output_files(tmp_path):
    history_path = plot_training_history(
        {"train_loss": [1.0, 0.5], "val_accuracy": [0.5, 1.0]},
        tmp_path / "history.png",
    )
    matrix_path = plot_confusion_matrix([[2, 1], [0, 3]], tmp_path / "matrix.png")
    ablation_path = plot_ablation_results(
        {"gcn": {"f1": 0.7}, "gat": {"f1": 0.8}},
        tmp_path / "ablation.png",
    )
    graph = nx.MultiDiGraph()
    graph.add_node(("ast", 0), view="ast")
    graph.add_node(("cfg", 0), view="cfg")
    graph.add_edge(("ast", 0), ("cfg", 0))
    graph_path = draw_program_graph(graph, tmp_path / "graph.png")
    for path in (history_path, matrix_path, ablation_path, graph_path):
        assert path.exists()
        assert path.stat().st_size > 0
