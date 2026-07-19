from __future__ import annotations

import random
from collections.abc import Collection, Sequence

import torch

from src.bank.core.allocator import allocate_by_debt
from src.bank.core.base import AbstractGhostBank, _to_int
from src.bank.core.exposure import compute_debt
from src.bank.core.retrieval import sample_by_allocation


class ExposureDebtGhostBank(AbstractGhostBank):
    """Exposure-debt-driven ghost bank with proportional allocation.

    Stores per-class example pools, then on ``query()`` computes
    exposure debt from the tracker state, allocates a retrieval
    budget proportionally, and retrieves samples.

    Pass ``exclude_classes=set()`` to store every class.
    """

    def __init__(
        self,
        num_classes: int,
        capacity_per_class: int,
        seed: int,
        exclude_classes: Collection[int] | None = None,
    ) -> None:
        excluded = set(exclude_classes) if exclude_classes is not None else set()
        self._bank: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._capacity = capacity_per_class
        self._rng = random.Random(seed)

        self._last_debt: list[float] = []
        self._last_allocation: list[int] = []

    @staticmethod
    def _to_tensor_label(y: object) -> torch.Tensor:
        if torch.is_tensor(y):
            return y
        return torch.tensor(y, dtype=torch.long)

    def store(self, examples: list) -> None:
        if getattr(self, "_frozen", False):
            return
        for example in examples:
            x, y = example
            y = self._to_tensor_label(y)
            cid = _to_int(y)
            if cid in self._bank and len(self._bank[cid]) < self._capacity:
                self._bank[cid].append((x, y))

    def query(  # type: ignore[override]
        self,
        budget: int,
        *,
        exposure: Sequence[int] | None = None,
        target_per_class: Sequence[float] | None = None,
        debt: Sequence[float] | None = None,
        temperature: float = 1.0,
    ) -> list:
        if debt is not None:
            debt = [d if c in self._bank else 0.0 for c, d in enumerate(debt)]
            allocation = allocate_by_debt(debt, budget, temperature=temperature)
        elif exposure is not None and target_per_class is not None:
            debt = compute_debt(exposure, target_per_class)
            allocation = allocate_by_debt(debt, budget)
        else:
            num_classes = len(self._bank) + 1
            debt = [0.0] * num_classes
            allocation = [0] * num_classes

        self._last_debt = debt
        self._last_allocation = allocation
        return sample_by_allocation(self._bank, allocation, self._rng)

    @property
    def last_debt(self) -> list[float]:
        return list(self._last_debt)

    @property
    def last_allocation(self) -> list[int]:
        return list(self._last_allocation)

    def expand(self, num_new_classes: int) -> None:
        max_existing = max(self._bank.keys()) if self._bank else -1
        for c in range(max_existing + 1, max_existing + 1 + num_new_classes):
            if c not in self._bank:
                self._bank[c] = []

    def state_dict(self) -> dict:
        return {
            "bank": {c: list(pool) for c, pool in self._bank.items()},
            "capacity": self._capacity,
        }

    def load_state_dict(self, state: dict) -> None:
        self._bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        self._capacity = state["capacity"]
