from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLoss(ABC, nn.Module):
    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        ...
