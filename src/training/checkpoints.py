"""Safe, explicit checkpoint persistence for Phase 5 models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, *, epoch: int | None = None, history: dict[str, list[float]] | None = None) -> None:
    """Save model state and optional optimizer/training metadata."""

    payload: dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if history is not None:
        payload["history"] = history
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint into a model and optionally an optimizer."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("Checkpoint must contain a model_state mapping")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload