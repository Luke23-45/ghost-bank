from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FocalConfig:
    alpha: float = 0.25
    gamma: float = 2.0
    reduction: str = "mean"
