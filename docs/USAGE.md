# Project Usage Guide

This guide explains how to install, run, test, and extend the program-analysis
pipeline. The project currently provides Python APIs for each stage rather than
a single command-line application.

## 1. What the project does

The pipeline turns a C or C++ source file into a graph suitable for graph
neural-network vulnerability classification:

1. Clang extracts an AST, an AST-derived intraprocedural CFG, and a local
   definition-use data-flow graph.
2. NetworkX builders convert the extraction payloads into three directed view
   graphs.
3. The views are merged into one namespaced, relation-aware
   `networkx.MultiDiGraph`.
4. Deterministic one-hot node features are attached.
5. The graph is converted to a PyTorch Geometric `Data` object.
6. A GCN, GAT, GraphSAGE, or relation-aware `MultiViewGNN` can classify the
   graph.
7. Evaluation metrics, ablations, and visualization artifacts can be produced.

The current extractors are intentionally conservative. The CFG is
AST-derived, and the data-flow graph models local lexical reaching
definitions; neither claims full LLVM CFG, alias, path-sensitive, or
interprocedural analysis.

## 2. Requirements

- Python 3.10 or newer
- Clang and LLVM with a working `libclang` installation
- A C/C++ compiler-compatible source file
- Python packages listed in `requirements.txt`

The project has been validated with PyTorch, PyTorch Geometric, NetworkX,
scikit-learn, Matplotlib, and the Clang Python bindings.

## 3. Installation

### Windows PowerShell

```powershell
git clone https://github.com/divyamsingh007/gnn-program-analysis.git
Set-Location gnn-program-analysis

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script activation, use the environment's interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

Install Clang first using the operating system package manager, then create a
virtual environment:

```bash
git clone https://github.com/divyamsingh007/gnn-program-analysis.git
cd gnn-program-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch and PyTorch Geometric builds appropriate
for the local CUDA version before or instead of the generic entries in
`requirements.txt`.

## 4. Verify the installation

Run the complete test suite:

```bash
python -m pytest -q
```

Compile all Python files and check imports:

```bash
python -m compileall -q src tests
python -c "import torch, torch_geometric, networkx, sklearn, matplotlib, clang.cindex; print('imports passed')"
```

The tests use `tests/sample.c` and cover extraction, graph construction, PyG
conversion, models, training, evaluation, and visualization.

## 5. Run extraction and build a multi-view graph

The following example can be saved as `scripts/run_pipeline.py` or run from a
Python shell at the repository root:

```python
from pathlib import Path

from src.extraction.ast_extractor import extract_ast
from src.extraction.cfg_extractor import extract_cfg
from src.extraction.dataflow_extractor import extract_dataflow
from src.features.node_features import encode_graph_nodes, fit_node_feature_encoder
from src.graphs.graph_builder import build_multiview_graph

source = Path("tests/sample.c").resolve()

ast_payload = extract_ast(source)
cfg_payload = extract_cfg(source)
dataflow_payload = extract_dataflow(source)

graph = build_multiview_graph(ast_payload, cfg_payload, dataflow_payload)
encoder = fit_node_feature_encoder(graph)
encoded_graph = encode_graph_nodes(graph, encoder)

print("nodes:", encoded_graph.number_of_nodes())
print("edges:", encoded_graph.number_of_edges())
print("feature size:", encoded_graph.graph["feature_size"])
```

Compiler arguments can be passed to each extractor when the source requires
include directories, language standards, or preprocessor definitions:

```python
payload = extract_ast(
    "path/to/source.c",
    compiler_args=["-std=c11", "-Ipath/to/includes", "-DMY_FLAG"],
)
```

All three views must refer to the same source file before they can be merged.

## 6. Convert graphs to PyTorch Geometric data

Phase 4 converts one or more encoded multi-view graphs into `Data` objects.
Use one shared relation vocabulary for a dataset:

```python
from src.datasets.dataset_loader import create_graph_loader
from src.datasets.preprocessing import batch_graphs_to_data

graphs = [encoded_graph]  # Add one encoded graph per source file.
labels = [1]              # Example: 0 = benign, 1 = vulnerable.

samples = batch_graphs_to_data(graphs, labels=labels)
loader = create_graph_loader(samples, batch_size=1, shuffle=True)

batch = next(iter(loader))
print(batch.x.shape)
print(batch.edge_index.shape)
print(batch.edge_type.shape)
print(batch.y)
```

Each sample contains:

- `x`: float32 node features;
- `edge_index`: graph connectivity with shape `[2, edge_count]`;
- `edge_type`: integer relation IDs;
- `y`: optional graph label;
- metadata such as `node_ids`, `node_view`, and relation vocabulary.

## 7. Train a model

Phase 5 provides four graph classifiers. The relation-aware model needs the
number of relation types in the shared dataset vocabulary:

```python
import torch

from src.models.multiview_gnn import MultiViewGNN
from src.training.train import train_model

relation_count = len(samples[0].relation_vocabulary)
model = MultiViewGNN(
    in_channels=samples[0].num_node_features,
    num_relations=relation_count,
    hidden_channels=64,
    num_classes=2,
)

history = train_model(
    model,
    loader,
    learning_rate=1e-3,
    epochs=10,
)
print(history)
```

Standard baselines use the same `Data`/`Batch` contract:

```python
from src.models.gcn import GCNClassifier
from src.models.gat import GATClassifier
from src.models.graphsage import GraphSAGEClassifier

gcn = GCNClassifier(in_channels=samples[0].num_node_features)
gat = GATClassifier(in_channels=samples[0].num_node_features)
sage = GraphSAGEClassifier(in_channels=samples[0].num_node_features)
```

For real experiments, split samples into training, validation, and test
loaders rather than training and evaluating on the same loader.

## 8. Save and restore checkpoints

```python
from src.training.checkpoints import load_checkpoint, save_checkpoint

save_checkpoint(
    "reports/checkpoints/model.pt",
    model,
    epoch=10,
    history=history,
)

restored_model = MultiViewGNN(
    in_channels=samples[0].num_node_features,
    num_relations=relation_count,
)
checkpoint = load_checkpoint("reports/checkpoints/model.pt", restored_model)
print("restored epoch:", checkpoint.get("epoch"))
```

Pass the optimizer to both functions when optimizer state must be resumed.

## 9. Evaluate and compare models

```python
from src.evaluation.evaluate import evaluate_model
from src.evaluation.ablation import run_ablation

metrics = evaluate_model(model, loader)
print(metrics["accuracy"])
print(metrics["precision"])
print(metrics["recall"])
print(metrics["f1"])
print(metrics["confusion_matrix"])

comparison = run_ablation(
    {
        "gcn": lambda: GCNClassifier(in_channels=samples[0].num_node_features),
        "gat": lambda: GATClassifier(in_channels=samples[0].num_node_features),
    },
    loader,
)
```

`evaluate_model` returns loss, aggregate metrics, labels, and predictions.
For stable confusion-matrix class ordering, call
`classification_metrics(..., labels_order=[0, 1])` directly.

## 10. Create plots and graph images

```python
from src.visualization.graph_visualizer import draw_program_graph
from src.visualization.result_plots import (
    plot_ablation_results,
    plot_confusion_matrix,
)
from src.visualization.training_plots import plot_training_history

plot_training_history(history, "reports/figures/training.png")
plot_confusion_matrix(
    metrics["confusion_matrix"],
    "reports/figures/confusion_matrix.png",
    class_names=["benign", "vulnerable"],
)
plot_ablation_results(comparison, "reports/figures/ablation.png", metric="f1")
draw_program_graph(encoded_graph, "reports/figures/program_graph.png")
```

The visualization functions create parent directories automatically and close
their Matplotlib figures after saving, so they work in headless environments.

## 11. Suggested project workflow

For a labeled dataset, repeat the following for every source file:

1. Extract AST, CFG, and data-flow payloads.
2. Build and feature-encode the multi-view graph.
3. Store the graph or convert it to a PyG `Data` sample.
4. Fit the shared relation vocabulary on the training set.
5. Split samples into train, validation, and test sets.
6. Train a baseline and the relation-aware `MultiViewGNN`.
7. Save the best checkpoint using validation loss or F1.
8. Evaluate once on the held-out test set.
9. Save metrics, confusion matrices, ablation charts, and representative graph
   visualizations under `reports/`.

Keep source files and generated datasets outside version control when they are
large or sensitive. The repository's `.gitignore` should be extended for local
datasets and generated reports as the experiment setup grows.

## 12. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the requirements again:

```bash
python -m pip install -r requirements.txt
```

### Clang or `libclang` cannot be loaded

Confirm that LLVM/Clang is installed and that the Python binding can locate its
shared library. On Windows, ensure the LLVM `bin` directory is available on
`PATH`. On Linux, install the distribution's `libclang` development package.

### PyTorch Geometric installation fails

Install the PyTorch build first, then follow the PyG installation instructions
for that exact PyTorch and CUDA combination. The generic `requirements.txt`
entry is suitable for environments where compatible wheels are available.

### A graph has no cross-view edges

Cross-view alignment is intentionally conservative: AST-to-CFG and
AST-to-dataflow links require an exact absolute file, line, and column match.
Missing matches do not indicate an extraction failure.

### `pip check` reports unrelated packages

`pip check` examines the entire Python environment, not only this project.
Use a fresh virtual environment to isolate project dependencies when unrelated
TensorFlow, MediaPipe, or other package conflicts are reported.

## 13. Phase documentation

The implementation details for each completed stage are recorded in:

- `docs/PHASE_1_IMPLEMENTATION.md`
- `docs/PHASE_2_IMPLEMENTATION.md`
- `docs/PHASE_3_IMPLEMENTATION.md`
- `docs/PHASE_4_IMPLEMENTATION.md`
- `docs/PHASE_5_IMPLEMENTATION.md`
- `docs/PHASE_6_IMPLEMENTATION.md`
- `docs/PHASE_7_IMPLEMENTATION.md`
