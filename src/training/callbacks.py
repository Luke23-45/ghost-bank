from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from src.bank.strategies.ed_gb import ExposureDebtGhostBank


class DebtCurveLogger(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        bank = getattr(pl_module, "bank", None)
        if isinstance(bank, ExposureDebtGhostBank):
            for i, d in enumerate(bank.last_debt):
                pl_module.log(f"debt/class_{i}", d, on_step=True)
            for i, a in enumerate(bank.last_allocation):
                pl_module.log(f"alloc/class_{i}", a, on_step=True)


class ExposureTrackerCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        tracker = getattr(pl_module, "exposure_tracker", None)
        if tracker is not None:
            for i, e in enumerate(tracker.accumulated()):
                pl_module.log(f"exposure/class_{i}", e, on_step=True)


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
