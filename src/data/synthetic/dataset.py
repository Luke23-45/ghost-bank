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
        self._xs = torch.tensor([ex[0] for ex in data], dtype=torch.float32)
        self._ys = torch.tensor([ex[1] for ex in data], dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._xs[index], self._ys[index]

    def __len__(self) -> int:
        return len(self._ys)

    @property
    def class_counts(self) -> list[int]:
        return torch.bincount(self._ys, minlength=self.num_classes).tolist()

    @property
    def num_classes(self) -> int:
        return int(self._ys.max().item()) + 1 if len(self._ys) > 0 else 0
