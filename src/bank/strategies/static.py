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
        self._seen_indices: set[int] = set()
        self._seen_counts: dict[int, int] = {}

    @staticmethod
    def _to_tensor_label(y: object) -> torch.Tensor:
        if torch.is_tensor(y):
            return y
        return torch.tensor(y, dtype=torch.long)

    def store(self, examples: list, raw_indices: torch.Tensor | None = None) -> None:
        if getattr(self, "_frozen", False):
            return
        indices = raw_indices.tolist() if raw_indices is not None else None
        for pos, example in enumerate(examples):
            x, y = example
            if indices is not None:
                sample_idx = int(indices[pos])
                if sample_idx in self._seen_indices:
                    continue
                self._seen_indices.add(sample_idx)
            y = self._to_tensor_label(y)
            cid = _to_int(y)
            if cid in self._bank:
                current_seen = self._seen_counts.get(cid, 0)
                if current_seen < self._capacity:
                    self._bank[cid].append((x, y))
                else:
                    r = self._rng.randint(0, current_seen)
                    if r < self._capacity:
                        self._bank[cid][r] = (x, y)
                self._seen_counts[cid] = current_seen + 1

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
            "seen_indices": list(self._seen_indices),
            "seen_counts": self._seen_counts.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        self._capacity = state["capacity"]
        self._seen_indices = set(int(i) for i in state.get("seen_indices", []))
        
        if "seen_counts" in state:
            self._seen_counts = {int(k): int(v) for k, v in state["seen_counts"].items()}
        else:
            self._seen_counts = {int(c): len(pool) for c, pool in self._bank.items()}
