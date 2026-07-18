from __future__ import annotations

import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from src.bank.strategies.ed_gb import ExposureDebtGhostBank
from src.bank.strategies.static import StaticReplayBank


class DistributionShiftCallback(pl.Callback):
    """Swaps labels for two classes at a given epoch.

    By default only the training-set labels are swapped.  Set
    ``shift_test_dataset=True`` to also swap test-set labels so that
    evaluation measures adaptation to the new distribution under
    proper concept drift (instead of retention of the old mapping).

    ``freeze_bank=True`` prevents any further storage into the
    replay buffer after the shift, preventing contamination from
    swapped-label data.
    """

    def __init__(
        self,
        shift_epoch: int = 5,
        swap_classes: tuple[int, int] = (0, 2),
        freeze_bank: bool = False,
        shift_test_dataset: bool = False,
    ) -> None:
        super().__init__()
        self.shift_epoch = shift_epoch
        self.swap_classes = swap_classes
        self.freeze_bank = freeze_bank
        self.shift_test_dataset = shift_test_dataset
        self._shifted = False

    @staticmethod
    def _swap_labels(dataset, c1: int, c2: int) -> None:
        mask1 = dataset._ys == c1
        mask2 = dataset._ys == c2
        dataset._ys[mask1] = c2
        dataset._ys[mask2] = c1

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch == self.shift_epoch and not self._shifted:
            ds = trainer.datamodule.train_dataset
            self._swap_labels(ds, *self.swap_classes)

            if self.shift_test_dataset:
                tds = trainer.datamodule.test_dataset
                self._swap_labels(tds, *self.swap_classes)

            if self.freeze_bank:
                bank = getattr(pl_module, "bank", None)
                if bank is not None and hasattr(bank, "freeze"):
                    bank.freeze()

            self._shifted = True

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Re-apply test label swap after PL re-creates test dataset via setup('test')."""
        if self.shift_test_dataset and self._shifted:
            tds = trainer.datamodule.test_dataset
            self._swap_labels(tds, *self.swap_classes)


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
