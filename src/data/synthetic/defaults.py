from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyntheticConfig:
    seed: int = 13
    imbalance_ratio: int = 100
    majority_train: int = 2000
    test_per_class: int = 500
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
