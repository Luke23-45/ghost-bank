import pytest

from src.bank.core.allocator import allocate_by_debt


# -- Error cases --------------------------------------------------------------

def test_negative_budget_raises():
    with pytest.raises(ValueError, match="retrieval budget must be non-negative"):
        allocate_by_debt([1.0, 2.0], -1)


# -- Zero / edge cases --------------------------------------------------------

def test_all_zero_debt():
    result = allocate_by_debt([0.0, 0.0, 0.0], 10)
    assert result == [0, 0, 0]


def test_budget_zero():
    result = allocate_by_debt([5.0, 10.0], 0)
    assert result == [0, 0]


def test_budget_zero_with_zero_debt():
    result = allocate_by_debt([0.0, 0.0], 0)
    assert result == [0, 0]


def test_single_class():
    result = allocate_by_debt([10.0], 5)
    assert result == [5]


def test_single_class_zero_debt():
    result = allocate_by_debt([0.0], 5)
    assert result == [0]


def test_single_class_budget_exceeds():
    result = allocate_by_debt([3.0], 100)
    assert result == [100]


def test_empty_debt_list():
    result = allocate_by_debt([], 10)
    assert result == []


def test_empty_debt_zero_budget():
    result = allocate_by_debt([], 0)
    assert result == []


# -- Proportionality ----------------------------------------------------------

def test_equal_debt_equal_allocation():
    result = allocate_by_debt([5.0, 5.0, 5.0], 9)
    assert result == [3, 3, 3]


def test_proportional_allocation():
    result = allocate_by_debt([1.0, 3.0], 12)
    assert result == [3, 9]


def test_proportional_one_class_zero():
    result = allocate_by_debt([0.0, 10.0, 0.0], 8)
    assert result == [0, 8, 0]


def test_allocation_sum_equals_budget():
    for _ in range(20):
        debts = [float(i * 1.3) for i in range(1, 6)]
        result = allocate_by_debt(debts, 10)
        assert sum(result) == 10


def test_allocation_sum_equals_budget_large():
    result = allocate_by_debt([3.0, 7.0, 2.0, 8.0], 100)
    assert sum(result) == 100


def test_non_decreasing_with_debt():
    debt = [1.0, 4.0, 2.0, 9.0]
    allocation = allocate_by_debt(debt, 20)
    for i in range(len(debt) - 1):
        for j in range(i + 1, len(debt)):
            if debt[i] < debt[j]:
                assert allocation[i] <= allocation[j]


# -- Largest-remainder --------------------------------------------------------

def test_largest_remainder_applied():
    debt = [1.0, 1.0, 1.0]
    result = allocate_by_debt(debt, 5)
    assert result == [2, 2, 1]


def test_allocation_no_remainder_exact():
    result = allocate_by_debt([2.0, 2.0, 2.0], 6)
    assert result == [2, 2, 2]


def test_remainder_tie_broken_by_debt():
    debt = [3.0, 1.0, 3.0, 1.0]
    result = allocate_by_debt(debt, 6)
    assert sum(result) == 6


def test_large_number_of_classes():
    debt = [float(i) for i in range(1, 101)]
    result = allocate_by_debt(debt, 50)
    assert len(result) == 100
    assert sum(result) == 50


def test_budget_smaller_than_classes():
    debt = [10.0, 10.0, 10.0, 10.0, 10.0]
    result = allocate_by_debt(debt, 2)
    assert sum(result) == 2
    assert all(r >= 0 for r in result)


def test_extreme_debt_differences():
    debt = [0.001, 1000.0]
    result = allocate_by_debt(debt, 10)
    assert result[1] >= result[0]
    assert sum(result) == 10
