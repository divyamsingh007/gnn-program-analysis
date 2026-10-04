# Phase 4 Implementation Record

## Scope

Phase 4 converts the encoded Phase 3 `networkx.MultiDiGraph` representation
into PyTorch Geometric `Data` objects suitable for model input. Conversion is
implemented in `src/datasets/preprocessing.py`; the small wrappers in
`src/datasets/pyg_dataset.py` and `src/datasets/dataset_loader.py` provide
dataset and loader entry points without importing PyG until they are used.

## Graph-to-PyG Contract

`multiview_graph_to_data(...)`:

- assigns deterministic integer node indices by sorting the namespaced Phase 3
  node IDs;
- copies node feature vectors to `data.x` as a `float32` tensor;
- emits `data.edge_index` with shape `[2, edge_count]`, including an empty
  tensor with that shape for edgeless graphs;
- encodes every edge relation in `data.edge_type` using a deterministic,
  alphabetically sorted vocabulary;
- preserves `node_ids`, `node_view`, `source_file`, `representation`,
  `schema_version`, and feature/relation vocabularies as data metadata;
- optionally stores a graph label in `data.y`.

Internal and cross-view edges remain separate through their relation IDs.
`batch_graphs_to_data(...)` fits one shared relation vocabulary before
conversion, which prevents relation IDs from changing between samples.

## Dataset and Loading Helpers

`build_pyg_data(...)` and `build_pyg_dataset(...)` are the public wrappers for
single and multiple graph conversion. `create_graph_loader(...)` constructs a
`torch_geometric.loader.DataLoader` and validates the batch size. PyG imports
are lazy so extraction and graph-building tests remain usable in environments
that have not installed the Phase 4 runtime yet.

## Validation

The Phase 4 checks are in `tests/test_preprocessing.py`. They verify stable
node indexing, tensor shapes and dtypes, relation preservation, labels,
shared vocabularies, and batching.

Validation run on 2026-10-05:

```text
python -m pytest tests/test_clang_utils.py tests/test_ast.py tests/test_cfg.py tests/test_dataflow.py tests/test_graph.py -q
9 passed
```

The Phase 4 tests are skipped when `torch-geometric` is not installed. Install
the declared dependencies before running them:

```text
python -m pip install -r requirements.txt
python -m pytest tests/test_preprocessing.py -q
```
