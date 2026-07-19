from __future__ import annotations

import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from src.bank.strategies.ed_gb import ExposureDebtGhostBank
from src.bank.strategies.static import StaticReplayBank


class DebtCurveLogger(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        bank = getattr(pl_module, "bank", None)
        if isinstance(bank, ExposureDebtGhostBank):
            for i, d in enumerate(bank.last_debt):
                pl_module.log(f"debt/class_{i}", d, on_epoch=True)
            for i, a in enumerate(bank.last_allocation):
                pl_module.log(f"alloc/class_{i}", a, on_epoch=True)


class ExposureTrackerCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        tracker = getattr(pl_module, "exposure_tracker", None)
        if tracker is not None:
            for i, e in enumerate(tracker.accumulated()):
                pl_module.log(f"exposure/class_{i}", e, on_epoch=True)


class GhostBankProgressBar(TQDMProgressBar):
    """Custom progress bar that shows ED-GB metrics in the training bar.

    Displays maximum per-class debt and total retrieved samples when
    an ExposureDebtGhostBank is attached to the module.
    """

    def __init__(self, refresh_rate: int = 1, leave: bool = True) -> None:
        super().__init__(refresh_rate=refresh_rate, leave=leave)

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict:
        items = super().get_metrics(trainer, pl_module)
        bank = getattr(pl_module, "bank", None)
        if isinstance(bank, ExposureDebtGhostBank):
            debt = bank.last_debt
            alloc = bank.last_allocation
            if debt:
                items["max_debt"] = f"{max(debt):.2f}"
            if alloc:
                items["retrieved"] = sum(alloc)
        return items

    def init_train_tqdm(self) -> object:
        bar = super().init_train_tqdm()
        bar.set_description("train")
        return bar

    def init_validation_tqdm(self) -> object:
        bar = super().init_validation_tqdm()
        bar.set_description("validate")
        return bar

    def init_test_tqdm(self) -> object:
        bar = super().init_test_tqdm()
        bar.set_description("test")
        return bar


class ConsoleEpochCallback(pl.Callback):
    """Prints a single concise line per epoch.

    Replaces the per-batch TQDM bar with a lightweight status line
    so cloud consoles stay clean while still showing progress.
    """

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self._prefix = f"{prefix} " if prefix else ""

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs or "?"
        loss = trainer.callback_metrics.get("train/loss")
        loss_str = f" | loss={loss:.4f}" if loss is not None else ""
        sys.stdout.write(
            f"\r{self._prefix}epoch {epoch + 1}/{max_epochs}{loss_str}  "
        )
        sys.stdout.flush()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
