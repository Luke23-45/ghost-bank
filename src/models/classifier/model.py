from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel


class MLPClassifier(BaseModel):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)
