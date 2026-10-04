"""Minimal, reproducible supervised training loop for graph classifiers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .losses import classification_loss


class Trainer:
    """Train and evaluate a graph classifier using PyG DataLoaders."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def _run_batch(self, batch: Any, training: bool) -> tuple[torch.Tensor, int]:
        batch = batch.to(self.device)
        logits = self.model(batch)
        loss = classification_loss(logits, batch.y)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.detach(), int(batch.y.view(-1).numel())

    def train_epoch(self, loader: Iterable[Any]) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            loss, count = self._run_batch(batch, training=True)
            total_loss += float(loss) * count
            total_items += count
        return total_loss / total_items if total_items else 0.0

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        correct = 0
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = classification_loss(logits, batch.y)
            labels = batch.y.view(-1).long()
            total_loss += float(loss) * labels.numel()
            total_items += labels.numel()
            correct += int((logits.argmax(dim=-1) == labels).sum())
        return {"loss": total_loss / total_items if total_items else 0.0, "accuracy": correct / total_items if total_items else 0.0}

    def fit(self, train_loader: Iterable[Any], validation_loader: Iterable[Any] | None = None, *, epochs: int = 1):
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        history: dict[str, list[float]] = {"train_loss": []}
        if validation_loader is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []
        for _ in range(epochs):
            history["train_loss"].append(self.train_epoch(train_loader))
            if validation_loader is not None:
                metrics = self.evaluate(validation_loader)
                history["val_loss"].append(metrics["loss"])
                history["val_accuracy"].append(metrics["accuracy"])
        return history