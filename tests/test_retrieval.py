import random

from src.bank.core.retrieval import sample_uniform, sample_by_allocation


def _make_bank(
    class_pools: dict[int, list],
) -> dict[int, list]:
    return {c: list(pool) for c, pool in class_pools.items()}


# -- sample_uniform -----------------------------------------------------------

def test_uniform_budget():
    bank = _make_bank({0: [("a", 0)] * 10, 1: [("b", 1)] * 10})
    result = sample_uniform(bank, 5, random.Random(42))
    assert len(result) == 5


def test_uniform_budget_zero():
    bank = _make_bank({0: [("a", 0)] * 10})
    result = sample_uniform(bank, 0, random.Random(42))
    assert result == []


def test_uniform_empty_bank():
    result = sample_uniform({}, 5, random.Random(42))
    assert result == []


def test_uniform_all_classes_empty():
    bank = _make_bank({0: [], 1: []})
    result = sample_uniform(bank, 5, random.Random(42))
    assert result == []


def test_uniform_some_classes_empty():
    bank = _make_bank({0: [("x", 0)], 1: [("y", 1)] * 10, 2: []})
    result = sample_uniform(bank, 5, random.Random(42))
    assert len(result) == 5


def test_uniform_single_class():
    bank = _make_bank({0: [("a", 0)] * 100})
    result = sample_uniform(bank, 10, random.Random(42))
    assert len(result) == 10
    assert all(item[1] == 0 for item in result)


def test_uniform_deterministic_seed():
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    bank = _make_bank({0: [("a", 0)] * 50, 1: [("b", 1)] * 50})
    result1 = sample_uniform(bank, 20, rng1)
    result2 = sample_uniform(bank, 20, rng2)
    assert result1 == result2


def test_uniform_different_seed_differs():
    rng1 = random.Random(1)
    rng2 = random.Random(2)
    bank = _make_bank({0: [("a", 0)] * 50, 1: [("b", 1)] * 50})
    result1 = sample_uniform(bank, 20, rng1)
    result2 = sample_uniform(bank, 20, rng2)
    assert len(result1) == len(result2)


def test_uniform_all_classes_represented():
    bank = _make_bank({0: [("a", 0)] * 100, 1: [("b", 1)] * 100})
    result = sample_uniform(bank, 100, random.Random(42))
    classes_seen = {item[1] for item in result}
    assert 0 in classes_seen
    assert 1 in classes_seen


def test_uniform_does_not_mutate_bank():
    bank = _make_bank({0: [("a", 0)] * 10, 1: [("b", 1)] * 10})
    original = {c: list(pool) for c, pool in bank.items()}
    sample_uniform(bank, 5, random.Random(42))
    assert bank == original


# -- sample_by_allocation -----------------------------------------------------

def test_allocation_respects_counts():
    bank = _make_bank({0: [("a", 0)] * 50, 1: [("b", 1)] * 50})
    result = sample_by_allocation(bank, [3, 7], random.Random(42))
    assert len(result) == 10
    class_0_count = sum(1 for item in result if item[1] == 0)
    class_1_count = sum(1 for item in result if item[1] == 1)
    assert class_0_count == 3
    assert class_1_count == 7


def test_allocation_zero():
    bank = _make_bank({0: [("a", 0)] * 10, 1: [("b", 1)] * 10})
    result = sample_by_allocation(bank, [0, 0], random.Random(42))
    assert result == []


def test_allocation_single_class():
    bank = _make_bank({0: [("a", 0)] * 10})
    result = sample_by_allocation(bank, [5], random.Random(42))
    assert len(result) == 5


def test_allocation_with_replacement():
    bank = _make_bank({0: [("a", 0)] * 2, 1: [("b", 1)] * 50})
    result = sample_by_allocation(bank, [10, 5], random.Random(42))
    assert len(result) == 15
    class_0_count = sum(1 for item in result if item[1] == 0)
    assert class_0_count == 10


def test_allocation_empty_class_pool():
    bank = _make_bank({0: [], 1: [("b", 1)] * 10})
    result = sample_by_allocation(bank, [3, 5], random.Random(42))
    assert len(result) == 5
    assert all(item[1] == 1 for item in result)


def test_allocation_class_not_in_bank():
    bank = _make_bank({1: [("b", 1)] * 10})
    result = sample_by_allocation(bank, [3, 5], random.Random(42))
    assert len(result) == 5


def test_allocation_deterministic_seed():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    bank = _make_bank({0: [("a", 0)] * 50, 1: [("b", 1)] * 50})
    result1 = sample_by_allocation(bank, [3, 7], rng1)
    result2 = sample_by_allocation(bank, [3, 7], rng2)
    assert result1 == result2


def test_allocation_does_not_mutate_bank():
    bank = _make_bank({0: [("a", 0)] * 10, 1: [("b", 1)] * 10})
    original = {c: list(pool) for c, pool in bank.items()}
    sample_by_allocation(bank, [3, 7], random.Random(42))
    assert bank == original
