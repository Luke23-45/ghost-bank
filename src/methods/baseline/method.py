from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext


class BaselineMethod(Method):
    """Lower-bound baseline: standard cross-entropy fine-tuning without replay."""

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch
        return F.cross_entropy(pl_module(x), y)
