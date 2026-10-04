# Phase 5 Implementation Record

## Scope

Phase 5 adds the first trainable GNN models and supervised training loop for
the Phase 4 PyTorch Geometric graph contract. The implementation uses graph
classification: node representations are propagated through the program graph,
mean-pooled per graph, and classified as benign/vulnerable (or any configured
multi-class label set).

## Models

The public models consume a PyG `Data` or `Batch` object with `x`, `edge_index`,
and `batch`:

- `src/models/gcn.py`: two-layer `GCNClassifier`;
- `src/models/gat.py`: two-layer multi-head `GATClassifier`;
- `src/models/graphsage.py`: two-layer `GraphSAGEClassifier`;
- `src/models/multiview_gnn.py`: `MultiViewGNN`, which adds Phase 4 `edge_type`
  relation IDs to each message before mean aggregation.

All models return logits shaped `[batch_size, num_classes]`. The shared
`GraphClassifier` performs graph-level mean pooling and dropout-regularized
classification.

## Training and Checkpoints

`src/training/trainer.py` provides `Trainer.train_epoch`, `evaluate`, and
`fit`. It uses cross-entropy, reports average loss and accuracy, moves batches
to the selected device, and supports optional validation loaders.

`src/training/train.py::train_model` creates an Adam optimizer. `save_checkpoint`
and `load_checkpoint` persist model state, optional optimizer state, epoch, and
history while rejecting malformed checkpoint files.

## Validation

`tests/test_models_training.py` verifies that every model variant produces
batch-sized logits, that the relation-aware model trains for multiple epochs,
and that checkpoint state round-trips successfully.

Validation run on 2026-10-05:

```text
python -m pytest -q
13 passed
```

The model and training tests are skipped when `torch-geometric` is absent.
