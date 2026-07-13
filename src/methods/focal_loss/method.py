from __future__ import annotations

import torch
import torch.nn as nn

from src.methods.base import Method


class FocalLossMethod(Method):
    def __init__(self, loss_fn: nn.Module) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank=None,
    ) -> torch.Tensor:
        x, y = batch
        return self.loss_fn(pl_module(x), y)
