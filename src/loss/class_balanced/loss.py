from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.loss.base import BaseLoss


class ClassBalancedLoss(BaseLoss):
    def __init__(self, beta: float = 0.999, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_counts: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if class_counts is None:
            return F.cross_entropy(logits, targets, reduction=self.reduction)

        weights = torch.tensor(
            [(1 - self.beta) / (1 - self.beta ** n) for n in class_counts],
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.cross_entropy(logits, targets, weight=weights, reduction=self.reduction)
