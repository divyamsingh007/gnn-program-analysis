"""Classification metrics for graph-level vulnerability predictions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    *,
    labels_order: Iterable[int] | None = None,
) -> dict[str, float | list[list[int]]]:
    """Return stable binary/multiclass metrics and a confusion matrix."""

    targets = np.asarray(list(labels), dtype=int)
    outputs = np.asarray(list(predictions), dtype=int)
    if targets.shape != outputs.shape:
        raise ValueError("labels and predictions must have the same length")
    if targets.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional")
    class_labels = list(labels_order) if labels_order is not None else sorted(
        set(targets.tolist()) | set(outputs.tolist())
    )
    return {
        "accuracy": float(accuracy_score(targets, outputs)),
        "precision": float(precision_score(targets, outputs, average="weighted", zero_division=0)),
        "recall": float(recall_score(targets, outputs, average="weighted", zero_division=0)),
        "f1": float(f1_score(targets, outputs, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, outputs, labels=class_labels).tolist(),
    }