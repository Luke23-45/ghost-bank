from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResNetConfig:
    depth: int = 18
    num_classes: int = 10
    base_filters: int = 64
    dropout: float = 0.0
