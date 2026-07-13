from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassBalancedConfig:
    beta: float = 0.999
    reduction: str = "mean"
