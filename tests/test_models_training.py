import networkx as nx
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.datasets.dataset_loader import create_graph_loader  # noqa: E402
from src.datasets.preprocessing import batch_graphs_to_data  # noqa: E402
from src.models.gat import GATClassifier  # noqa: E402
from src.models.gcn import GCNClassifier  # noqa: E402
from src.models.graphsage import GraphSAGEClassifier  # noqa: E402
from src.models.multiview_gnn import MultiViewGNN  # noqa: E402
from src.training.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def _sample_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(representation="multiview", schema_version="1.0")
    graph.add_node(("ast", 0), view="ast", features=[1.0, 0.0])
    graph.add_node(("cfg", 0), view="cfg", features=[0.0, 1.0])
    graph.add_edge(("ast", 0), ("cfg", 0), relation="ast_to_cfg")
    graph.graph["feature_vocabulary"] = ("a", "b")
    return graph


def _loader():
    data = batch_graphs_to_data([_sample_graph(), _sample_graph()], labels=[0, 1])
    return create_graph_loader(data, batch_size=2)


def test_all_model_variants_produce_graph_logits():
    batch = next(iter(_loader()))
    models = [
        GCNClassifier(2, hidden_channels=8),
        GATClassifier(2, hidden_channels=4, heads=2),
        GraphSAGEClassifier(2, hidden_channels=8),
        MultiViewGNN(2, num_relations=len(batch.relation_vocabulary), hidden_channels=8),
    ]
    for model in models:
        assert model(batch).shape == (2, 2)


def test_trainer_updates_and_checkpoint_round_trip(tmp_path):
    loader = _loader()
    model = MultiViewGNN(2, num_relations=1, hidden_channels=8)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=0.01))
    history = trainer.fit(loader, epochs=2)
    assert len(history["train_loss"]) == 2
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, trainer.optimizer, epoch=2, history=history)
    restored = MultiViewGNN(2, num_relations=1, hidden_channels=8)
    assert load_checkpoint(checkpoint, restored)["epoch"] == 2
