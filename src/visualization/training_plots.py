"""Training-history plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def plot_training_history(
    history: dict[str, Iterable[float]],
    output_path: str | Path,
    *,
    title: str = "Training history",
) -> Path:
    """Save loss and optional validation-accuracy curves from a trainer history."""

    import matplotlib.pyplot as plt

    if "train_loss" not in history:
        raise ValueError("history must contain train_loss")
    train_loss = list(history["train_loss"])
    if not train_loss:
        raise ValueError("train_loss must not be empty")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)
    figure, axis = plt.subplots()
    axis.plot(epochs, train_loss, label="train_loss")
    for key in ("val_loss", "val_accuracy"):
        if key in history:
            values = list(history[key])
            if len(values) != len(train_loss):
                raise ValueError(f"{key} must have the same length as train_loss")
            axis.plot(epochs, values, label=key)
    axis.set(title=title, xlabel="Epoch", ylabel="Value")
    axis.legend()
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)
    return target