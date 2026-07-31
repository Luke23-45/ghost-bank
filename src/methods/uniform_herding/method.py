from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext
from src.methods.static_bank.method import _augment_replay


class UniformHerdingMethod(Method):
    """Replay method that uses a fixed-total herding bank."""

    def __init__(
        self,
        retrieval_budget: int = 64,
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
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch
        if bank is None:
            return F.cross_entropy(pl_module(x), y)

        if context is not None and context.raw_x is not None and context.raw_y is not None:
            examples = list(zip(context.raw_x, context.raw_y.tolist()))
            bank.store(examples, raw_indices=context.raw_indices)
        else:
            x_cpu = x.detach().cpu()
            y_labels = y.detach().cpu().tolist()
            bank.store([(x_i.clone(), y_i) for x_i, y_i in zip(x_cpu, y_labels)])

        if pl_module.global_step >= self.warmup_steps:
            replay_items = bank.query(budget=self.retrieval_budget)
            replay_x = _augment_replay(
                replay_items,
                transform=context.train_transform if context is not None else None,
                rng=context.augment_rng if context is not None else None,
                device=y.device,
            )
            replay_y = (
                torch.tensor(
                    [int(item[1]) for item in replay_items],
                    device=y.device,
                    dtype=torch.long,
                )
                if replay_items
                else None
            )

            if replay_x is not None and replay_y is not None and replay_y.numel() > 0:
                x = torch.cat([x, replay_x], dim=0)
                y = torch.cat([y, replay_y], dim=0)

        return F.cross_entropy(pl_module(x), y)
