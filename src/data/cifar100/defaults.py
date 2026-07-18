from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass, field


@dataclass
class CIFAR100Config:
    root: str = "./data/cifar100"
    seed: int = 13
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    num_tasks: int = 10
    classes_per_task: int = 10
    val_split: float = 0.0
    mean: tuple = (0.5071, 0.4867, 0.4408)
    std: tuple = (0.2675, 0.2565, 0.2761)

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root).resolve() / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root).resolve() / "processed")
