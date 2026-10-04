"""Evaluation result plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def plot_confusion_matrix(
    matrix: Iterable[Iterable[int]],
    output_path: str | Path,
    *,
    class_names: Iterable[str] | None = None,
    title: str = "Confusion matrix",
) -> Path:
    """Save a labeled confusion-matrix heatmap."""

    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(list(list(row) for row in matrix), dtype=int)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    names = list(class_names) if class_names is not None else [str(i) for i in range(values.shape[0])]
    if len(names) != values.shape[0]:
        raise ValueError("class_names must match matrix dimensions")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names, title=title, xlabel="Predicted", ylabel="Actual")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)
    return target


def plot_ablation_results(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    *,
    metric: str = "f1",
) -> Path:
    """Save a bar chart comparing one metric across named ablations."""

    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must not be empty")
    values = []
    for name, result in results.items():
        if metric not in result:
            raise ValueError(f"result {name!r} does not contain metric {metric!r}")
        values.append(float(result[metric]))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.bar(list(results), values)
    axis.set(title=f"Ablation comparison: {metric}", ylabel=metric)
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)
    return target