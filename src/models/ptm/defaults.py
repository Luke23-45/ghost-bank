from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PTModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    freeze_exclude_last_stage: bool = False
    embedding_dim: int = 512
    classifier_mode: str = "linear"
    num_classes: int = 10
    adapter_type: str | None = None
    adapter_hidden_dim: int = 128
