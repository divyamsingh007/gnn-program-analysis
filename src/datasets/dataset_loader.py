"""DataLoader construction for converted Phase 4 graph samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def create_graph_loader(
    data: Sequence[Any],
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    **kwargs: Any,
):
    """Create a PyG loader without importing PyG at module-import time."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    try:
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Phase 4 loading requires torch-geometric; install requirements.txt first"
        ) from exc
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, **kwargs)