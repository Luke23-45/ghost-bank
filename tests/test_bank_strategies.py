"""Tests for the supported replay banks."""

import random

import torch

from src.bank.strategies.static import StaticReplayBank
from src.bank.strategies.herding import HerdingReplayBank


def _make_examples(
    labels: list[int],
    dim: int = 2,
) -> list[tuple[torch.Tensor, int]]:
    rng = random.Random(0)
    return [
        (torch.tensor([rng.gauss(0, 1) for _ in range(dim)], dtype=torch.float32), y)
        for y in labels
    ]


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)


class TestStaticReplayBank:
    def test_init_creates_bank_for_all_classes(self):
        bank = StaticReplayBank(num_classes=4, capacity_per_class=10, seed=42)
        assert set(bank._bank.keys()) == {0, 1, 2, 3}

    def test_store_and_query(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        bank.store(_make_examples([1, 1, 2, 2, 2]))
        result = bank.query(budget=4)
        assert len(result) == 4

    def test_store_deduplicates_raw_indices(self):
        bank = StaticReplayBank(num_classes=3, capacity_per_class=10, seed=42)
        examples = _make_examples([1, 1, 2, 2])
        raw_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        bank.store(examples, raw_indices=raw_indices)
        first_total = sum(len(pool) for pool in bank._bank.values())
        bank.store(examples, raw_indices=raw_indices)
        second_total = sum(len(pool) for pool in bank._bank.values())
        assert first_total == second_total


class TestHerdingReplayBank:
    def test_rebuild_selected_and_query(self):
        bank = HerdingReplayBank(num_classes=3, total_budget=6, seed=0)
        bank.store([
            (torch.full((3, 4, 4), 1.0), 0),
            (torch.full((3, 4, 4), 2.0), 0),
            (torch.full((3, 4, 4), 3.0), 1),
            (torch.full((3, 4, 4), 4.0), 1),
        ])

        stats = bank.rebuild_selected(
            model=DummyModel(),
            allocation=[1, 1, 0],
            eval_transform=None,
        )

        assert stats["total"] == 2
        assert len(bank.selected[0]) == 1
        assert len(bank.selected[1]) == 1
        assert bank.query(2)
