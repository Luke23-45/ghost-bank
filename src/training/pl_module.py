from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.bank.core.exposure import ExposureTracker
from src.methods.base import Method
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class GhostBankLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        method: Method,
        bank: AbstractGhostBank | None = None,
        learning_rate: float = 0.05,
        num_classes: int | None = None,
        optimizer_name: str = "sgd",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("model", "method", "bank"))
        self.model = model
        self.method = method
        self.bank = bank
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.exposure_tracker: ExposureTracker | None = None
        if getattr(method, "needs_exposure_tracker", False):
            if num_classes is not None:
                self.exposure_tracker = ExposureTracker(num_classes)
            else:
                LOGGER.warning(
                    "Method %s requires exposure tracker but num_classes is None; "
                    "exposure tracking disabled.",
                    type(method).__name__,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self.method.compute_loss(batch, self, bank=self.bank)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/acc", acc, on_epoch=True, on_step=False)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, y = batch
        logits = self.model(x)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log("test/acc", acc, on_epoch=True, on_step=False)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, _ = batch
        return self.model(x).argmax(dim=-1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
