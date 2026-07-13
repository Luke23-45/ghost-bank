from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from src.methods.base import Method


class ClassBalancedMethod(Method):
    def __init__(self, loss_fn: nn.Module, class_counts: Sequence[int]) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.class_counts = list(class_counts)

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank=None,
    ) -> torch.Tensor:
        x, y = batch
        return self.loss_fn(pl_module(x), y, class_counts=self.class_counts)
