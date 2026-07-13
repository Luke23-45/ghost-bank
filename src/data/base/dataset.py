from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    @property
    @abstractmethod
    def class_counts(self) -> list[int]:
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    def imbalance_ratio(self) -> float:
        counts = self.class_counts
        if min(counts) == 0:
            return float("inf")
        return max(counts) / min(counts)
