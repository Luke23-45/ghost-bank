from __future__ import annotations

import torch
import torch.nn.functional as F

from src.methods.base import Method


class BaselineMethod(Method):
    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank=None,
    ) -> torch.Tensor:
        x, y = batch
        return F.cross_entropy(pl_module(x), y)
