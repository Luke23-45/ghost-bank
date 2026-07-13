from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.base import BaseLoss


class LDAMLoss(BaseLoss):
    def __init__(
        self,
        cls_num_list: list[int] | None = None,
        max_m: float = 0.5,
        s: float = 30.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if cls_num_list is not None:
            m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float)))
            m_list = m_list * (max_m / m_list.max().item())
            self.register_buffer("m_list", m_list)
        else:
            self.m_list = None
        self.s = s
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.m_list is None:
            return F.cross_entropy(logits, targets, reduction=self.reduction)

        margin = self.m_list.to(logits.device)
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        logits_m = logits.clone()
        logits_m[index] -= margin[targets]

        return F.cross_entropy(self.s * logits_m, targets, reduction=self.reduction)
