from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class EDGBMethod(Method):
    needs_exposure_tracker = True

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
        tracker = getattr(pl_module, "exposure_tracker", None)

        if tracker is not None:
            for label in y:
                tracker.record(label)
        elif getattr(self, "_warned_missing_tracker", False) is False:
            LOGGER.warning(
                "EDGBMethod: no exposure_tracker found on pl_module. "
                "Exposure tracking and debt-based retrieval disabled."
            )
            self._warned_missing_tracker = True

        if bank is not None:
            bank.store([(x[i], y[i]) for i in range(len(y))])

            bank_x, bank_y = [], []
            if pl_module.global_step >= self.warmup_steps and tracker is not None:
                acc = tracker.accumulated()
                target = pl_module.global_step * x.size(0) / len(acc)
                target_per_class = [target] * len(acc)

                for bx, by in bank.query(
                    budget=self.retrieval_budget,
                    exposure=acc,
                    target_per_class=target_per_class,
                ):
                    bank_x.append(bx.unsqueeze(0))
                    bank_y.append(by.unsqueeze(0))
                    tracker.record(by)

            if bank_x:
                x = torch.cat([x] + bank_x)
                y = torch.cat([y] + bank_y)

        return F.cross_entropy(pl_module(x), y)
