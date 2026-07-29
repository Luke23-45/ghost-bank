from __future__ import annotations

from src.bank.core.allocator import (
    allocate_fixed_total,
    allocate_uniform_fixed_total,
)


def test_allocate_fixed_total_sum():
    num_classes = 10
    total_budget = 2000
    probe_scores = [float(i) / 9.0 for i in range(10)]

    alloc = allocate_fixed_total(
        num_classes, total_budget, probe_scores,
        gamma=0.5, beta=1.0, floor=1,
    )

    assert len(alloc) == num_classes
    assert sum(alloc) == total_budget


def test_allocate_fixed_total_floor():
    num_classes = 100
    total_budget = 2000
    probe_scores = [0.0] * num_classes

    alloc = allocate_fixed_total(
        num_classes, total_budget, probe_scores,
        gamma=0.5, beta=1.0, floor=1,
    )

    assert all(a >= 1 for a in alloc)
    assert sum(alloc) == total_budget


def test_allocate_fixed_total_probe_prioritises_high_score():
    num_classes = 5
    total_budget = 100
    probe_scores = [10.0, 1.0, 0.1, 0.01, 0.001]

    alloc = allocate_fixed_total(
        num_classes, total_budget, probe_scores,
        gamma=1.0, beta=10.0, floor=1,
    )

    assert alloc[0] >= alloc[1], "Highest probe class should get most"
    assert alloc[0] > alloc[-1], "Highest probe class should get more than lowest"


def test_allocate_fixed_total_uniform_when_gamma_zero():
    num_classes = 10
    total_budget = 2000
    probe_scores = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    alloc = allocate_fixed_total(
        num_classes, total_budget, probe_scores,
        gamma=0.0, beta=1.0, floor=1,
    )

    expected = total_budget // num_classes
    assert all(a == expected or a == expected + 1 for a in alloc)
    assert sum(alloc) == total_budget


def test_allocate_fixed_total_no_probe_scores():
    num_classes = 10
    total_budget = 2000

    alloc = allocate_fixed_total(
        num_classes, total_budget, probe_scores=None,
        gamma=0.5, beta=1.0, floor=1,
    )

    expected = total_budget // num_classes
    assert all(a == expected or a == expected + 1 for a in alloc)
    assert sum(alloc) == total_budget


def test_allocate_fixed_total_zero_budget():
    alloc = allocate_fixed_total(10, 0, None, gamma=0.5, beta=1.0, floor=0)
    assert alloc == [0] * 10


def test_allocate_uniform_fixed_total():
    num_classes = 10
    total_budget = 2000

    alloc = allocate_uniform_fixed_total(num_classes, total_budget, floor=1)

    assert len(alloc) == num_classes
    assert sum(alloc) == total_budget
    assert all(a >= 1 for a in alloc)


def test_allocate_fixed_total_empty():
    assert allocate_fixed_total(0, 100, None) == []


def test_allocate_fixed_total_negative_budget():
    import pytest
    with pytest.raises(ValueError):
        allocate_fixed_total(5, -1, None)
