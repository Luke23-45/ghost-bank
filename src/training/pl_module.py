from __future__ import annotations

from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.bank.core.exposure import ExposureTracker
from src.bank.core.pid_controller import PIDController
from src.methods.base import Method
from src.utils.logging import get_logger
from src.utils.metrics import balanced_accuracy, macro_f1, minority_recall

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
        lr_scheduler: str | None = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        minority_classes: Sequence[int] | None = None,
        exposure_tracker: ExposureTracker | None = None,
        pid_controller: PIDController | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("model", "method", "bank", "exposure_tracker", "pid_controller"))
        self.model = model
        self.method = method
        self.bank = bank
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.minority_classes = minority_classes

        self.exposure_tracker = exposure_tracker
        if self.exposure_tracker is None and getattr(method, "needs_exposure_tracker", False):
            if num_classes is not None:
                self.exposure_tracker = ExposureTracker(num_classes)
            else:
                LOGGER.warning(
                    "Method %s requires exposure tracker but num_classes is None; "
                    "exposure tracking disabled.",
                    type(method).__name__,
                )

        self.pid_controller = pid_controller
        if self.pid_controller is None and getattr(method, "needs_pid_controller", False):
            if num_classes is not None:
                self.pid_controller = PIDController(
                    num_classes,
                    K_p=getattr(method, "K_p", 1.0),
                    K_i=getattr(method, "K_i", 0.1),
                    K_d=getattr(method, "K_d", 0.5),
                    decay=getattr(method, "pid_decay", 0.99),
                    smooth=getattr(method, "pid_smooth", 0.9),
                    temperature=getattr(method, "temperature", 1.0),
                    class_weights=getattr(method, "class_weights", None),
                )
            else:
                LOGGER.warning(
                    "Method %s requires PID controller but num_classes is None; "
                    "PID control disabled.",
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

    def on_test_start(self) -> None:
        self.test_preds: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, y = batch
        logits = self.model(x)
        preds = logits.argmax(dim=-1)
        self.test_preds.append(preds.cpu())
        self.test_labels.append(y.cpu())
        acc = (preds == y).float().mean()
        self.log("test/acc", acc, on_epoch=True, on_step=False)

    def on_test_epoch_end(self) -> None:
        if not self.test_preds or self.num_classes is None:
            self.test_preds.clear()
            self.test_labels.clear()
            return

        preds = torch.cat(self.test_preds)
        labels = torch.cat(self.test_labels)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                acc = (preds[mask] == c).float().mean()
                self.log(f"test/acc_class_{c}", acc)

        bal_acc = balanced_accuracy(labels, preds, self.num_classes)
        self.log("test/balanced_acc", bal_acc)

        f1 = macro_f1(labels, preds, self.num_classes)
        self.log("test/macro_f1", f1)

        if self.minority_classes:
            m_recall = minority_recall(labels, preds, self.minority_classes)
            self.log("test/minority_recall", m_recall)

        self.test_preds.clear()
        self.test_labels.clear()

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, _ = batch
        return self.model(x).argmax(dim=-1)

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.trainer.max_epochs if self.trainer else 100,
            )
            return {
                "optimizer": optim,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        return optim
