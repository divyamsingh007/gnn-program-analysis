"""Evaluate a Phase 5 graph classifier on a PyG DataLoader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .metrics import classification_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Collect predictions and return loss, classification metrics, and outputs."""

    from src.training.losses import classification_loss

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        batch = batch.to(target_device)
        logits = model(batch)
        labels = batch.y.view(-1).long()
        loss = classification_loss(logits, labels)
        predictions = logits.argmax(dim=-1)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        total_loss += float(loss) * labels.numel()
        total_items += labels.numel()
    result = classification_metrics(all_labels, all_predictions, labels_order=[0, 1])
    result["loss"] = total_loss / total_items if total_items else 0.0
    result["labels"] = all_labels
    result["predictions"] = all_predictions
    return result