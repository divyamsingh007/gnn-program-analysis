"""Run comparable model ablations over a shared evaluation loader."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from .evaluate import evaluate_model


def run_ablation(
    model_factories: dict[str, Callable[[], torch.nn.Module]],
    loader: Iterable[Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, dict[str, Any]]:
    """Evaluate named model factories and return a comparable result table."""

    if not model_factories:
        raise ValueError("model_factories must not be empty")
    results: dict[str, dict[str, Any]] = {}
    for name, factory in model_factories.items():
        if not name:
            raise ValueError("ablation names must not be empty")
        results[name] = evaluate_model(factory(), loader, device=device)
    return results