from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LDAMConfig:
    cls_num_list: list[int] | None = None
    max_m: float = 0.5
    s: float = 30.0
    reduction: str = "mean"
