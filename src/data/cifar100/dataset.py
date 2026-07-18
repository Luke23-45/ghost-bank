from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.data.base import BaseDataset


class CIFAR100TaskView(BaseDataset):
    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        class_indices: list[int],
        transform: object | None = None,
    ) -> None:
        super().__init__()
        mask = torch.isin(targets, torch.tensor(class_indices))
        self._images = images[mask]
        self._targets = targets[mask]
        self._class_indices = sorted(class_indices)
        self._transform = transform
        self._num_classes = max(self._class_indices) + 1 if self._class_indices else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self._images[index]
        label = self._targets[index].item()
        if img.dim() == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)
        if self._transform is not None:
            img = self._transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._targets)

    @property
    def class_counts(self) -> list[int]:
        counts = torch.zeros(self._num_classes, dtype=torch.long)
        for c in range(self._num_classes):
            counts[c] = (self._targets == c).sum().item()
        return counts.tolist()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_indices(self) -> list[int]:
        return list(self._class_indices)
