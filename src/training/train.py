"""Convenience entry point for training a Phase 5 graph classifier."""

from __future__ import annotations

import torch

from .trainer import Trainer


def train_model(model: torch.nn.Module, train_loader, validation_loader=None, *, learning_rate: float = 1e-3, epochs: int = 10, device: str = "cpu"):
    """Create a trainer and fit a supplied graph classifier."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, device=device)
    return trainer.fit(train_loader, validation_loader, epochs=epochs)