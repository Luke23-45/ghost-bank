from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch

from src.data.base import BaseDataset


Vector = Sequence[float]
Example = tuple[Vector, int]


def generate_gaussian_data(
    seed: int,
    imbalance_ratio: int,
    majority_train: int = 2000,
    test_per_class: int = 500,
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    centers = [(-2.0, -1.0), (2.0, 0.0), (0.0, 2.5)]
    medium_count = max(1, int(majority_train / math.sqrt(imbalance_ratio)))
    rare_count = max(1, majority_train // imbalance_ratio)
    train_counts = [majority_train, medium_count, rare_count]
    test_counts = [test_per_class] * 3

    def make_examples(counts: list[int]) -> list[Example]:
        examples: list[Example] = []
        for class_id, count in enumerate(counts):
            cx, cy = centers[class_id]
            for _ in range(count):
                x = [rng.gauss(cx, 0.9), rng.gauss(cy, 0.9)]
                examples.append((x, class_id))
        rng.shuffle(examples)
        return examples

    return make_examples(train_counts), make_examples(test_counts)


class GaussianDataset(BaseDataset):
    def __init__(self, data: list[Example]) -> None:
        super().__init__()
        self._data = data
        self._class_counts = self._compute_class_counts()

    def _compute_class_counts(self) -> list[int]:
        if not self._data:
            return []
        num = max(y for _, y in self._data) + 1
        counts = [0] * num
        for _, y in self._data:
            counts[y] += 1
        return counts

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._data[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def class_counts(self) -> list[int]:
        return list(self._class_counts)

    @property
    def num_classes(self) -> int:
        return len(self._class_counts)
