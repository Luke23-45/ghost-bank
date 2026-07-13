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

    By default class 0 is excluded from storage and retrieval (it is
    the majority class in the synthetic benchmark).  Override with
    ``exclude_classes``.
    """

    def __init__(
        self,
        num_classes: int,
        capacity_per_class: int,
        seed: int,
        exclude_classes: Collection[int] | None = None,
    ) -> None:
        excluded = set(exclude_classes) if exclude_classes is not None else {0}
        self._bank: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._capacity = capacity_per_class
        self._rng = random.Random(seed)

        self._last_debt: list[float] = []
        self._last_allocation: list[int] = []

    def store(self, examples: list) -> None:
        for example in examples:
            _, y = example
            cid = _to_int(y)
            if cid in self._bank and len(self._bank[cid]) < self._capacity:
                self._bank[cid].append(example)

    def query(  # type: ignore[override]
        self,
        budget: int,
        *,
        exposure: Sequence[int] | None = None,
        target_per_class: Sequence[float] | None = None,
    ) -> list:
        if exposure is not None and target_per_class is not None:
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

    def state_dict(self) -> dict:
        return {
            "bank": {c: list(pool) for c, pool in self._bank.items()},
            "capacity": self._capacity,
        }

    def load_state_dict(self, state: dict) -> None:
        self._bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        self._capacity = state["capacity"]
