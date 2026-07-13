from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OutputConfig:
    base_dir: str = "output"
    enabled_formats: list[str] = field(default_factory=lambda: ["csv", "jsonl"])
