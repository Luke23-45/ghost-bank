from __future__ import annotations

import random
from collections.abc import Collection

import torch

from src.bank.core.base import AbstractGhostBank, _to_int
from src.bank.core.retrieval import sample_uniform


class StaticReplayBank(AbstractGhostBank):
    """Uniform random replay buffer.

    Stores per-class example pools for all classes **except** those
    listed in ``exclude_classes``.  By default class 0 is excluded
    (the majority class in the synthetic benchmark).
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

    def store(self, examples: list) -> None:
        for example in examples:
            _, y = example
            cid = _to_int(y)
            if cid in self._bank and len(self._bank[cid]) < self._capacity:
                self._bank[cid].append(example)

    def query(self, budget: int, **kwargs) -> list:
        return sample_uniform(self._bank, budget, self._rng)

    def state_dict(self) -> dict:
        return {
            "bank": {c: list(pool) for c, pool in self._bank.items()},
            "capacity": self._capacity,
        }

    def load_state_dict(self, state: dict) -> None:
        self._bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        self._capacity = state["capacity"]
