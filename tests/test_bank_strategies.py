"""Tests for StaticReplayBank and ExposureDebtGhostBank.

Both banks store every class by default; pass ``exclude_classes`` to
skip specific classes.
"""

import random

import pytest
import torch

from src.bank.strategies.static import StaticReplayBank
from src.bank.strategies.ed_gb import ExposureDebtGhostBank


# -- Helpers ------------------------------------------------------------------

def _make_examples(
    labels: list[int],
    dim: int = 2,
) -> list[tuple[torch.Tensor, int]]:
    rng = random.Random(0)
    return [
        (torch.tensor([rng.gauss(0, 1) for _ in range(dim)], dtype=torch.float32), y)
        for y in labels
    ]


# -- StaticReplayBank ---------------------------------------------------------

class TestStaticReplayBank:
    def test_init_creates_bank_for_all_classes(self):
        bank = StaticReplayBank(num_classes=4, capacity_per_class=10, seed=42)
        assert set(bank._bank.keys()) == {0, 1, 2, 3}
        assert 0 in bank._bank

    def test_store_and_query(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        examples = _make_examples([1, 1, 2, 2, 2])
        bank.store(examples)
        result = bank.query(budget=4)
        assert len(result) == 4
        assert all(ex[1] in (1, 2) for ex in result)

    def test_store_class_zero(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        examples = _make_examples([0, 0, 0])
        bank.store(examples)
        result = bank.query(budget=3)
        assert len(result) == 3
        assert all(ex[1] == 0 for ex in result)

    def test_query_empty_bank(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        result = bank.query(budget=5)
        assert result == []

    def test_query_budget_zero(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        examples = _make_examples([1, 2])
        bank.store(examples)
        result = bank.query(budget=0)
        assert result == []

    def test_capacity_enforced(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=3, seed=42)
        examples = _make_examples([1] * 10)
        bank.store(examples)
        assert len(bank._bank[1]) == 3

    def test_capacity_per_class_multiple_classes(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=5, seed=42)
        bank.store(_make_examples([1] * 20 + [2] * 20))
        assert len(bank._bank[1]) == 5
        assert len(bank._bank[2]) == 5

    def test_deterministic_query(self):
        examples = _make_examples([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        bank1 = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=99)
        bank1.store(examples)
        bank2 = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=99)
        bank2.store(examples)
        assert bank1.query(4, rng=random.Random(10)) == bank2.query(
            4, rng=random.Random(10)
        )

    def test_state_dict_roundtrip(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 1, 2, 2, 2]))
        state = bank.state_dict()
        assert "bank" in state
        assert "capacity" in state
        assert state["capacity"] == 10

        bank2 = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=99)
        bank2.load_state_dict(state)
        assert bank2._capacity == 10
        assert len(bank2._bank[1]) == 2

    def test_multiple_store_and_query(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=20, seed=42)
        bank.store(_make_examples([1] * 5))
        bank.store(_make_examples([2] * 5))
        result = bank.query(budget=10)
        assert len(result) == 10


# -- ExposureDebtGhostBank ----------------------------------------------------

class TestExposureDebtGhostBank:
    def test_init_creates_bank_for_all_classes(self):
        bank = ExposureDebtGhostBank(num_classes=4, capacity_per_class=10, seed=42)
        assert set(bank._bank.keys()) == {0, 1, 2, 3}
        assert 0 in bank._bank

    def test_store_and_query_with_exposure(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 1, 1, 1]))
        result = bank.query(
            budget=4,
            exposure=[10, 0, 10],
            target_per_class=[10.0, 10.0, 10.0],
        )
        assert len(result) == 4
        assert all(ex[1] == 1 for ex in result)

    def test_query_without_exposure(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 2] * 5))
        result = bank.query(budget=4)
        assert len(result) == 0

    def test_query_empty_bank(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        result = bank.query(budget=5, exposure=[0, 5, 10], target_per_class=[10.0, 10.0, 10.0])
        assert result == []

    def test_query_budget_zero(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 2]))
        result = bank.query(
            budget=0,
            exposure=[0, 5, 10],
            target_per_class=[10.0, 10.0, 10.0],
        )
        assert result == []

    def test_last_debt_and_allocation_populated(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 2]))
        bank.query(
            budget=4,
            exposure=[0, 5, 10],
            target_per_class=[10.0, 10.0, 10.0],
        )
        assert len(bank.last_debt) == 3
        assert len(bank.last_allocation) == 3
        assert all(d >= 0 for d in bank.last_debt)
        assert all(a >= 0 for a in bank.last_allocation)
        assert sum(bank.last_allocation) == 4

    def test_last_debt_zero_when_no_exposure(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 2]))
        bank.query(budget=4)
        assert bank.last_debt == [0.0, 0.0, 0.0]
        assert bank.last_allocation == [0, 0, 0]

    def test_last_debt_correct_values(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 2]))
        bank.query(
            budget=10,
            exposure=[0, 20, 5],
            target_per_class=[10.0, 10.0, 10.0],
        )
        expected_debt = [10.0, 0.0, 5.0]
        for d, exp in zip(bank.last_debt, expected_debt):
            assert abs(d - exp) < 1e-9

    def test_capacity_enforced(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=3, seed=42)
        bank.store(_make_examples([1] * 10))
        assert len(bank._bank[1]) == 3

    def test_state_dict_roundtrip(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 1, 2]))
        state = bank.state_dict()
        assert "bank" in state
        assert "capacity" in state

        bank2 = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=99)
        bank2.load_state_dict(state)
        assert bank2._capacity == 10
        assert len(bank2._bank[1]) == 2
        assert len(bank2._bank[2]) == 1

    def test_exposure_drives_allocation(self):
        bank = ExposureDebtGhostBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1] * 10 + [2] * 10))
        bank.query(
            budget=10,
            exposure=[10, 10, 5],
            target_per_class=[10.0, 10.0, 10.0],
        )
        assert bank.last_allocation[2] >= bank.last_allocation[1]
