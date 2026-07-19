from __future__ import annotations

import random
from collections.abc import Collection

import torch

from src.bank.core.base import AbstractGhostBank, _to_int
from src.bank.core.retrieval import sample_uniform


class StaticReplayBank(AbstractGhostBank):
    """Uniform random replay buffer.

    Stores per-class example pools for all classes **except** those
    listed in ``exclude_classes``.  Pass ``exclude_classes=set()`` to
    store every class.
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

    def query(self, budget: int, **kwargs) -> list:
        return sample_uniform(self._bank, budget, self._rng)

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
