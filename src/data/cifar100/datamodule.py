from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.data.base import BaseDataModule
from src.data.cifar100.dataset import CIFAR100TaskView
from src.data.cifar100.defaults import CIFAR100Config
from src.data.cifar100.ingest import CIFAR100Ingestor
from src.data.cifar100.transforms import make_eval_transform, make_train_transform


class CIFAR100DataModule(BaseDataModule):
    """CIFAR-100 Class-Incremental Learning data module.

    Raw tensors live in ``self._train_images``, ``self._test_images`` (and
    optionally ``self._val_images``).  These are uint8 NHWC for the
    bank to consume.  The :class:`CIFAR100TaskView` instances returned
    by ``get_task_loaders`` apply the train / eval transform lazily at
    ``__getitem__`` time.
    """

    def __init__(self, config: CIFAR100Config) -> None:
        super().__init__()
        self.config = config
        self._train_images: torch.Tensor | None = None
        self._train_targets: torch.Tensor | None = None
        self._test_images: torch.Tensor | None = None
        self._test_targets: torch.Tensor | None = None
        self._val_images: torch.Tensor | None = None
        self._val_targets: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        ingestor = CIFAR100Ingestor(
            root=self.config.root,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )
        ingestor.ingest()

        processed = Path(self.config.processed_dir)
        self._train_images = torch.load(processed / "train_images.pt", weights_only=True)
        self._train_targets = torch.load(processed / "train_targets.pt", weights_only=True)
        self._test_images = torch.load(processed / "test_images.pt", weights_only=True)
        self._test_targets = torch.load(processed / "test_targets.pt", weights_only=True)

        val_img = processed / "val_images.pt"
        val_tgt = processed / "val_targets.pt"
        if val_img.is_file() and val_tgt.is_file():
            self._val_images = torch.load(val_img, weights_only=True)
            self._val_targets = torch.load(val_tgt, weights_only=True)

    @property
    def num_tasks(self) -> int:
        return self.config.num_tasks

    @property
    def classes_per_task(self) -> int:
        return self.config.classes_per_task

    def _class_range(self, task_id: int) -> list[int]:
        start = task_id * self.config.classes_per_task
        end = start + self.config.classes_per_task
        return list(range(start, end))

    def _shared_eval_transform(self) -> T.Compose:
        return make_eval_transform(self.config.mean, self.config.std)

    def _shared_train_transform(self) -> T.Compose:
        return make_train_transform(self.config.mean, self.config.std)

    def get_task_loaders(self, task_id: int) -> tuple[DataLoader, DataLoader]:
        class_range = self._class_range(task_id)
        train_view = CIFAR100TaskView(
            self._train_images, self._train_targets, class_range,
            transform=self._shared_train_transform(),
        )
        test_view = CIFAR100TaskView(
            self._test_images, self._test_targets, class_range,
            transform=self._shared_eval_transform(),
        )
        train_loader = DataLoader(
            train_view,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )
        test_loader = DataLoader(
            test_view,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return train_loader, test_loader

    def get_eval_loader(self, up_to_task_id: int) -> DataLoader:
        class_range: list[int] = []
        for t in range(up_to_task_id + 1):
            class_range.extend(self._class_range(t))
        eval_view = CIFAR100TaskView(
            self._test_images, self._test_targets, class_range,
            transform=self._shared_eval_transform(),
        )
        return DataLoader(
            eval_view,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def get_task_test_loader(self, task_id: int) -> DataLoader:
        class_range = self._class_range(task_id)
        view = CIFAR100TaskView(
            self._test_images, self._test_targets, class_range,
            transform=self._shared_eval_transform(),
        )
        return DataLoader(
            view,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def get_val_task_loader(self, task_id: int) -> DataLoader | None:
        if self._val_images is None or self._val_targets is None:
            return None
        class_range = self._class_range(task_id)
        view = CIFAR100TaskView(
            self._val_images, self._val_targets, class_range,
            transform=self._shared_eval_transform(),
        )
        return DataLoader(
            view,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    @property
    def train_dataset(self) -> CIFAR100TaskView:
        class_range = self._class_range(0)
        return CIFAR100TaskView(
            self._train_images, self._train_targets, class_range,
            transform=self._shared_train_transform(),
        )

    @property
    def test_dataset(self) -> CIFAR100TaskView:
        class_range: list[int] = []
        for t in range(self.config.num_tasks):
            class_range.extend(self._class_range(t))
        return CIFAR100TaskView(
            self._test_images, self._test_targets, class_range,
            transform=self._shared_eval_transform(),
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_task_loaders(0)[0]

    def val_dataloader(self) -> DataLoader:
        return self.get_eval_loader(0)

    def test_dataloader(self) -> DataLoader:
        return self.get_eval_loader(self.config.num_tasks - 1)
