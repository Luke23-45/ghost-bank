from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProbeGuidedConfig:
    retrieval_budget: int = 64
    warmup_steps: int = 0
    memory_total: int = 2000
    memory_floor: int = 1
    gamma: float = 0.5
    beta: float = 1.0
    probe_smoothing: float = 0.0
    distillation_weight: float = 0.0
    use_prototype_classifier: bool = False
    calibrate: bool = False
    calibration_lr: float = 0.01
    calibration_epochs: int = 10
