# Phase 6 Implementation Record

## Scope

Phase 6 adds evaluation and ablation support on top of the Phase 5 graph
classifiers. The evaluation layer consumes the same PyTorch Geometric
`DataLoader` contract used for training and reports standard vulnerability
classification metrics.

## Evaluation APIs

- `src/evaluation/metrics.py::classification_metrics` returns weighted
  accuracy, precision, recall, F1, and a deterministic confusion matrix.
- `src/evaluation/confusion_matrix.py::build_confusion_matrix` exposes the
  confusion matrix as a NumPy array for plotting or downstream reporting.
- `src/evaluation/evaluate.py::evaluate_model` runs a model in evaluation mode,
  computes average cross-entropy loss, and returns labels and predictions in
  addition to the metrics.
- `src/evaluation/ablation.py::run_ablation` evaluates named model factories
  against the same loader, producing directly comparable result dictionaries.

Metrics explicitly validate aligned one-dimensional label/prediction inputs.
`zero_division=0` keeps reports deterministic for small or single-class test
sets rather than emitting undefined values.

## Validation

`tests/test_evaluation.py` verifies metric values, class ordering, confusion
matrix output, model evaluation, and the ablation result shape.

Validation run on 2026-10-05:

```text
python -m pytest -q
15 passed
```

Phase 6 deliberately leaves plotting and dataset splitting as separate
concerns: results are returned as ordinary Python/NumPy structures so the
existing visualization modules can consume them without coupling evaluation
to a particular plotting backend.
