"""Tests for the supported replay banks."""

import random

import torch

from src.bank.core.allocator import allocate_uniform_fixed_total
from src.bank.strategies.static import StaticReplayBank
from src.methods.uniform_herding.herding import UniformHerdingReplayBank


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

    def test_set_quotas_prunes_pools_to_allocation(self):
        bank = StaticReplayBank(num_classes=4, capacity_per_class=100, seed=42)
        bank.store(_make_examples([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100))
        bank.set_quotas([1, 1, 1, 1])
        sizes = {c: len(pool) for c, pool in bank._bank.items()}
        assert sizes == {0: 1, 1: 1, 2: 1, 3: 1}

    def test_store_respects_quota(self):
        bank = StaticReplayBank(num_classes=2, capacity_per_class=100, seed=42)
        bank.set_quotas([2, 2])
        bank.store(_make_examples([0] * 50 + [1] * 50))
        sizes = {c: len(pool) for c, pool in bank._bank.items()}
        assert sizes == {0: 2, 1: 2}

    def test_ten_task_trajectory_keeps_total_budget(self):
        bank = StaticReplayBank(num_classes=10, capacity_per_class=200, seed=42)
        k = 2000
        idx = 0
        for task in range(10):
            if task > 0:
                bank.expand(10)
            bank.set_quotas(
                allocate_uniform_fixed_total(
                    num_classes=(task + 1) * 10,
                    total_budget=k,
                    floor=1,
                )
            )
            for c in range(10 * task, 10 * (task + 1)):
                examples = [(torch.tensor([float(i)]), c) for i in range(450)]
                bank.store(
                    examples,
                    raw_indices=torch.arange(idx, idx + 450, dtype=torch.long),
                )
                idx += 450
            total = sum(len(pool) for pool in bank._bank.values())
            assert total == k
        sizes = {c: len(pool) for c, pool in bank._bank.items()}
        assert all(size == 20 for size in sizes.values())

    def test_state_dict_roundtrip_preserves_quotas(self):
        bank = StaticReplayBank(num_classes=2, capacity_per_class=100, seed=42)
        bank.set_quotas([5, 5])
        bank.store(_make_examples([0] * 30 + [1] * 30))
        restored = StaticReplayBank(num_classes=2, capacity_per_class=100, seed=42)
        restored.load_state_dict(bank.state_dict())
        assert restored._quotas == {0: 5, 1: 5}
        assert sum(len(pool) for pool in restored._bank.values()) == 10


class TestHerdingReplayBank:
    def test_rebuild_selected(self):
        bank = UniformHerdingReplayBank(num_classes=3, total_budget=6, seed=0)
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
        assert sum(len(pool) for pool in bank.selected.values()) == 2

    def test_ten_task_trajectory_documented_memory_invariant(self):
        """Locks the docstring's invariant: the selected set equals the
        fixed budget at every rebuild boundary, the pool is bounded by
        multiplier * total_budget at boundaries, and mid-task the footprint
        exceeds that only by the full streams of classes awaiting their
        first rebuild (at most classes_per_task * stream)."""
        bank = UniformHerdingReplayBank(
            num_classes=10,
            total_budget=2000,
            seed=42,
            pool_multiplier=3,
        )
        idx = 0
        for task in range(10):
            if task > 0:
                bank.expand(10)
            bank.start_task()
            for c in range(10 * task, 10 * (task + 1)):
                examples = [
                    (torch.full((3, 8, 8), float(i % 7 + 1)), c)
                    for i in range(450)
                ]
                bank.store(
                    examples,
                    raw_indices=torch.arange(idx, idx + 450, dtype=torch.long),
                )
                idx += 450
            assert bank.bank_size() <= 3 * 2000 + 10 * 450
            stats = bank.rebuild_selected(model=DummyModel(), allocation=None)
            assert stats["total"] == 2000
            assert sum(len(v) for v in bank.selected.values()) == 2000
            assert bank.bank_size() <= 3 * 2000
        sizes = {c: len(v) for c, v in bank.selected.items()}
        assert all(size == 20 for size in sizes.values())
