"""Loss functions used by the Phase 5 training loop."""

from __future__ import annotations

import torch
from torch import nn


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy for graph labels stored as scalar or length-one tensors."""

    return nn.functional.cross_entropy(logits, labels.view(-1).long())