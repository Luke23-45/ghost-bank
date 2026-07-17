from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class PIDGBMethod(Method):
    """PID-Controlled Replay method.

    Uses per-class loss as the error signal for a PID controller.
    The PID debt replaces count-based exposure debt, allocating replay
    budget proportional to each class's learning difficulty.

    When a class is absent from the current batch, its loss is estimated
    by evaluating the model on a sample of items from the replay buffer.
    This ensures minority classes receive fair PID attention.
    """

    needs_pid_controller = True

    needs_class_counts = True

    def __init__(
        self,
        retrieval_budget: int = 8,
        warmup_steps: int = 0,
        K_p: float = 1.0,
        K_i: float = 0.1,
        K_d: float = 0.5,
        pid_decay: float = 0.99,
        pid_smooth: float = 0.9,
        temperature: float = 1.0,
        bank_probe_size: int = 16,
        eval_absent_classes: bool = True,
        use_class_weights: bool = True,
    ) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.pid_decay = pid_decay
        self.pid_smooth = pid_smooth
        self.temperature = temperature
        self.bank_probe_size = bank_probe_size
        self.eval_absent_classes = eval_absent_classes
        self.use_class_weights = use_class_weights

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
    ) -> torch.Tensor:
        x, y = batch
        pid = getattr(pl_module, "pid_controller", None)

        # --- 1. Per-class loss on current batch (no grad) for PID signal ---
        debt = None
        if pid is not None:
            with torch.no_grad():
                logits = pl_module(x)
            per_class_loss = self._compute_per_class_loss(
                logits, y, pl_module.num_classes, pl_module, bank
            )
            debt = pid.update(per_class_loss)

        # --- 2. Store and (optionally) retrieve from bank ---
        if bank is not None:
            bank.store([(x[i], y[i]) for i in range(len(y))])

            bank_x, bank_y = [], []
            if pl_module.global_step >= self.warmup_steps and debt is not None:
                for bx, by in bank.query(
                    budget=self.retrieval_budget,
                    debt=debt,
                    temperature=self.temperature,
                ):
                    bank_x.append(bx.unsqueeze(0))
                    bank_y.append(by.unsqueeze(0))

            if bank_x:
                x = torch.cat([x] + bank_x)
                y = torch.cat([y] + bank_y)

        return F.cross_entropy(pl_module(x), y)

    def _compute_per_class_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        pl_module=None,
        bank: AbstractGhostBank | None = None,
    ) -> list[float | None]:
        losses: list[float | None] = []
        device = logits.device
        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], targets[mask], reduction="mean")
                losses.append(loss.item())
            elif (
                self.eval_absent_classes
                and pl_module is not None
                and bank is not None
                and hasattr(bank, "_bank")
                and c in bank._bank
                and len(bank._bank[c]) > 0
            ):
                items = bank._bank[c]
                n = min(self.bank_probe_size, len(items))
                sampled = random.sample(items, n)
                bx = torch.stack([item[0] for item in sampled])
                by = torch.tensor([item[1] for item in sampled], device=device)
                with torch.no_grad():
                    bank_logits = pl_module(bx.to(device))
                loss = F.cross_entropy(bank_logits, by, reduction="mean")
                losses.append(loss.item())
            else:
                losses.append(None)
        return losses
