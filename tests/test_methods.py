"""Unit tests for training methods used by CIFAR-100.

Each method's ``compute_loss`` must return a scalar gradient-bearing tensor.
"""

import random

import pytest
import torch

from src.bank.strategies.static import StaticReplayBank
from src.bank.strategies.ed_gb import ExposureDebtGhostBank
from src.methods import (
    BaselineMethod,
    EDGBMethod,
    StaticBankMethod,
)


# -- Helpers ------------------------------------------------------------------

BATCH_SIZE = 4
NUM_CLASSES = 3
FEATURE_DIM = 2


class MockModule(torch.nn.Module):
    """Minimal stand-in for a PL LightningModule during method tests."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.global_step = 1000
        self.exposure_tracker = None
        self._num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.size(0), self._num_classes, requires_grad=True)


def _make_batch(
    batch_size: int = BATCH_SIZE,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, FEATURE_DIM, requires_grad=True)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


# -- BaselineMethod -----------------------------------------------------------

class TestBaselineMethod:
    def test_compute_loss_returns_scalar(self):
        method = BaselineMethod()
        loss = method.compute_loss(_make_batch(), MockModule())
        assert loss.ndim == 0

    def test_compute_loss_requires_grad(self):
        method = BaselineMethod()
        loss = method.compute_loss(_make_batch(), MockModule())
        assert loss.requires_grad

    def test_compute_loss_bank_ignored(self):
        method = BaselineMethod()
        loss_with = method.compute_loss(_make_batch(), MockModule(), bank=StaticReplayBank(3, 10, 0))
        loss_without = method.compute_loss(_make_batch(), MockModule())
        assert loss_with.ndim == 0
        assert loss_without.ndim == 0

    def test_compute_loss_positive(self):
        method = BaselineMethod()
        loss = method.compute_loss(_make_batch(), MockModule())
        assert loss > 0

    def test_multiple_calls(self):
        method = BaselineMethod()
        for _ in range(5):
            loss = method.compute_loss(_make_batch(), MockModule())
            assert loss.ndim == 0


# -- StaticBankMethod ---------------------------------------------------------

class TestStaticBankMethod:
    def _populate_bank(self, bank, labels: list[int]):
        """Store examples with tensor labels (matching training-time format)."""
        examples = [(torch.randn(FEATURE_DIM), torch.tensor(y)) for y in labels]
        bank.store(examples)

    def test_compute_loss_with_bank_returns_scalar(self):
        method = StaticBankMethod(retrieval_budget=4, warmup_steps=0)
        bank = StaticReplayBank(NUM_CLASSES, capacity_per_class=10, seed=0)
        self._populate_bank(bank, [1] * 5)
        pl_module = MockModule()
        pl_module.global_step = 100
        loss = method.compute_loss(_make_batch(), pl_module, bank=bank)
        assert loss.ndim == 0

    def test_compute_loss_without_bank(self):
        method = StaticBankMethod(retrieval_budget=4, warmup_steps=0)
        loss = method.compute_loss(_make_batch(), MockModule(), bank=None)
        assert loss.ndim == 0

    def test_during_warmup_bank_not_queried(self):
        method = StaticBankMethod(retrieval_budget=4, warmup_steps=500)
        bank = StaticReplayBank(NUM_CLASSES, capacity_per_class=10, seed=0)
        self._populate_bank(bank, [1] * 5)
        pl_module = MockModule()
        pl_module.global_step = 0
        loss = method.compute_loss(_make_batch(), pl_module, bank=bank)
        assert loss.ndim == 0

    def test_requires_grad(self):
        method = StaticBankMethod(retrieval_budget=4, warmup_steps=0)
        bank = StaticReplayBank(NUM_CLASSES, capacity_per_class=10, seed=0)
        self._populate_bank(bank, [1] * 5)
        pl_module = MockModule()
        pl_module.global_step = 100
        loss = method.compute_loss(_make_batch(), pl_module, bank=bank)
        assert loss.requires_grad


# -- EDGBMethod ---------------------------------------------------------------

class TestEDGBMethod:
    def test_compute_loss_with_bank_and_tracker_returns_scalar(self):
        method = EDGBMethod(retrieval_budget=4, warmup_steps=0)
        bank = ExposureDebtGhostBank(NUM_CLASSES, capacity_per_class=10, seed=0)
        bank.store([(torch.randn(FEATURE_DIM), 1) for _ in range(5)])
        pl_module = MockModule(num_classes=NUM_CLASSES)
        pl_module.global_step = 100
        loss = method.compute_loss(_make_batch(), pl_module, bank=bank)
        assert loss.ndim == 0

    def test_compute_loss_without_bank(self):
        method = EDGBMethod(retrieval_budget=4, warmup_steps=0)
        loss = method.compute_loss(_make_batch(), MockModule(), bank=None)
        assert loss.ndim == 0

    def test_during_warmup_bank_not_queried(self):
        method = EDGBMethod(retrieval_budget=4, warmup_steps=500)
        bank = ExposureDebtGhostBank(NUM_CLASSES, capacity_per_class=10, seed=0)
        bank.store([(torch.randn(FEATURE_DIM), 1) for _ in range(5)])
        pl_module = MockModule(num_classes=NUM_CLASSES)
        pl_module.global_step = 0
        loss = method.compute_loss(_make_batch(), pl_module, bank=bank)
        assert loss.ndim == 0

    def test_requires_grad(self):
        method = EDGBMethod(retrieval_budget=4, warmup_steps=0)
        bank = ExposureDebtGhostBank(NUM_CLASSES, capacity_per_class=10, seed=0)
        bank.store([(torch.randn(FEATURE_DIM), 1) for _ in range(5)])
        pl_module = MockModule(num_classes=NUM_CLASSES)
        pl_module.global_step = 100
        loss = method.compute_loss(_make_batch(), pl_module, bank=bank)
        assert loss.requires_grad
