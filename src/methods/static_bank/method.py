from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method


class StaticBankMethod(Method):
    def __init__(
        self,
        retrieval_budget: int = 8,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
    ) -> torch.Tensor:
        x, y = batch

        if bank is not None:
            bank.store([(x[i], y[i]) for i in range(len(y))])

            bank_x, bank_y = [], []
            if pl_module.global_step >= self.warmup_steps:
                for bx, by in bank.query(budget=self.retrieval_budget):
                    bank_x.append(bx.unsqueeze(0))
                    bank_y.append(by.unsqueeze(0))

            if bank_x:
                x = torch.cat([x] + bank_x)
                y = torch.cat([y] + bank_y)

        return F.cross_entropy(pl_module(x), y)
