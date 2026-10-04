# Phase 3 Implementation Record

## Scope

Phase 3 converts the Phase 2 AST, CFG, and data-flow dictionaries into
NetworkX graphs and creates a single relation-aware multi-view program graph.
It also supplies deterministic one-hot node features that Phase 4 can turn
into tensors without relying on a pretrained model.

## View Graphs

The following builders preserve the Phase 2 node IDs and edge attributes in
`networkx.DiGraph` instances:

- `src/graphs/ast_graph.py`: `build_ast_graph(...)`
- `src/graphs/cfg_graph.py`: `build_cfg_graph(...)`
- `src/graphs/dataflow_graph.py`: `build_dataflow_graph(...)`

`src/graphs/graph_utils.py` validates the shared payload contract before a
graph is built. It rejects missing required keys, duplicate node IDs, and edges
that reference nodes absent from the payload. This keeps invalid extraction
output from silently reaching model preparation.

## Multi-View Graph

`src/graphs/graph_builder.py` provides `build_multiview_graph(...)`, which
returns a `networkx.MultiDiGraph`.

- Every node uses a namespaced identity: `(view, original_id)`.
- Internal edges retain their Phase 2 type and identify their originating
  `view`.
- AST-to-CFG and AST-to-dataflow edges are added only when nodes have the same
  absolute source file, line, and column.
- Cross-view relations are labeled `ast_to_cfg` and `ast_to_dataflow`.

Exact location matching is intentional: it provides auditable links while
avoiding unsupported semantic assumptions. A missing location match produces
no cross-view edge rather than an approximate connection.

## Node Features

`src/features/node_features.py` provides a pure-Python, deterministic one-hot
encoding workflow:

1. `fit_node_feature_encoder(...)` collects and sorts node categories from one
   or more NetworkX graphs.
2. `encode_graph_nodes(...)` writes a one-hot vector to each node's
   `features` attribute and returns a copy by default.

The first vector position is reserved for unknown categories. Known categories
derive from the AST `kind`, CFG/DFG `ast_kind`, or a fallback block type. The
graph records both `feature_size` and `feature_vocabulary` as graph metadata.

This baseline is stable and inspectable. It does not yet encode token text,
type spellings, source positions, or learned embeddings; these can be layered
onto the same node attributes in a later experiment.

## Validation

`tests/test_graph.py` verifies that the individual view builders preserve node
and edge counts, the merged graph has all three namespaced views with alignment
relations, and attached feature vectors are one-hot without mutating the input
graph by default.

Validation run on 2026-10-05:

```text
python -m pytest tests/test_clang_utils.py tests/test_ast.py tests/test_cfg.py tests/test_dataflow.py tests/test_graph.py -q
9 passed in 11.95s
```

The configured project virtual environment now has the focused Phase 1-3
runtime packages installed: `libclang`, `networkx`, and `pytest`.

## Phase 4 Handoff

Phase 4 can consume an encoded `MultiDiGraph` by assigning each namespaced
node a stable integer index, converting `features` to `x`, and emitting edge
indices plus the preserved `relation` attribute. The multi-edge structure must
be retained until edge relations have been encoded for PyTorch Geometric.
