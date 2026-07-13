from __future__ import annotations

from collections.abc import Sequence

import torch

from src.bank.core.base import _to_int


def compute_debt(accumulated: Sequence[int], target: Sequence[float]) -> list[float]:
    """Return per-class debt = max(0, target - accumulated)."""
    return [max(0.0, t - a) for a, t in zip(accumulated, target)]


class ExposureTracker:
    """Tracks how many times each class has been seen during training."""

    def __init__(self, num_classes: int) -> None:
        self._accumulated = [0] * num_classes

    def record(self, class_id: int | torch.Tensor, count: int = 1) -> None:
        self._accumulated[_to_int(class_id)] += count

    def record_batch(self, labels: Sequence[int | torch.Tensor]) -> None:
        for y in labels:
            self._accumulated[_to_int(y)] += 1

    def accumulated(self) -> list[int]:
        return list(self._accumulated)

    def debt(self, target_per_class: Sequence[float]) -> list[float]:
        return compute_debt(self._accumulated, target_per_class)

    def reset(self) -> None:
        self._accumulated = [0] * len(self._accumulated)

    def state_dict(self) -> dict:
        return {"accumulated": list(self._accumulated)}

    def load_state_dict(self, state: dict) -> None:
        self._accumulated = list(state["accumulated"])
