from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.bank.core.base import AbstractGhostBank


class Method(ABC):
    @abstractmethod
    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
    ) -> torch.Tensor:
        ...
