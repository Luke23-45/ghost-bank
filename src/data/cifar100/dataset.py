from __future__ import annotations

from typing import Sequence

import torch

from src.data.base import BaseDataset
from src.data.cifar100.transforms import (
    make_eval_transform,
    make_train_transform,
    make_train_transform_from_rng,
)


class CIFAR100TaskView(BaseDataset):
    """A view over the CIFAR-100 train or test tensors restricted to ``class_indices``.

    The dataset stores three representations:
        * ``_raw_images`` — uint8 NHWC ``[N, 32, 32, 3]`` (in-memory copy of
          the stored tensors; this is what the bank stores).
        * ``_raw_targets`` — int64 ``[N]``.
        * ``_class_indices`` — sorted list of global class IDs in this view.

    ``__getitem__`` returns ``(view_index, img_nchw, label)`` where the
    index identifies which sample of *this view* was sampled.  This makes
    it trivial for callers to fetch the corresponding **raw** image from
    ``raw_images[index]`` if they want to bypass the augmentation and
    store a raw exemplar in a replay buffer.

    The img tensor returned at __getitem__ has the train-time
    augmentation applied (or eval-time normalization, depending on the
    transform passed in).
    """

    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        class_indices: Sequence[int],
        transform: object | None = None,
    ) -> None:
        super().__init__()
        indices = sorted(int(c) for c in class_indices)
        mask = torch.isin(targets, torch.tensor(indices))
        self._raw_images = images[mask].contiguous()
        self._raw_targets = targets[mask].contiguous()
        self._class_indices = indices
        self._transform = transform
        self._num_classes = (max(indices) + 1) if indices else 0

    @property
    def raw_images(self) -> torch.Tensor:
        """NHWC uint8 ``[N, 32, 32, 3]``.  Stable across the lifetime of this view."""
        return self._raw_images

    @property
    def raw_targets(self) -> torch.Tensor:
        """int64 ``[N]`` of *global* class IDs."""
        return self._raw_targets

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        img_nhwc = self._raw_images[index]
        img_nchw = img_nhwc.permute(2, 0, 1).contiguous()
        label = self._raw_targets[index].item()
        if self._transform is not None:
            img_nchw = self._transform(img_nchw)
        return index, img_nchw, torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return self._raw_targets.shape[0]

    @property
    def class_counts(self) -> list[int]:
        counts = torch.zeros(self._num_classes, dtype=torch.long)
        for c in self._class_indices:
            counts[c] = int((self._raw_targets == c).sum().item())
        return counts.tolist()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_indices(self) -> list[int]:
        return list(self._class_indices)
