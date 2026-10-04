"""Confusion-matrix helpers for evaluation reports."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def build_confusion_matrix(
    labels: Iterable[int],
    predictions: Iterable[int],
    *,
    labels_order: Iterable[int] | None = None,
) -> np.ndarray:
    """Build a deterministic confusion matrix with an optional class order."""

    targets = list(labels)
    outputs = list(predictions)
    if len(targets) != len(outputs):
        raise ValueError("labels and predictions must have the same length")
    classes = list(labels_order) if labels_order is not None else sorted(set(targets) | set(outputs))
    return sklearn_confusion_matrix(targets, outputs, labels=classes)