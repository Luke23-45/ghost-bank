from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLPConfig:
    input_dim: int = 2
    hidden_dim: int = 16
    num_classes: int = 3
