import torch

from src.bank.core.base import _to_int
from src.bank.core.exposure import ExposureTracker, compute_debt


# -- _to_int ------------------------------------------------------------------

def test_to_int_with_python_int():
    assert _to_int(3) == 3


def test_to_int_with_tensor_scalar():
    assert _to_int(torch.tensor(7)) == 7


def test_to_int_with_zero():
    assert _to_int(0) == 0


# -- compute_debt -------------------------------------------------------------

def test_compute_debt_all_zero():
    assert compute_debt([0, 0, 0], [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_compute_debt_positive():
    assert compute_debt([5, 10], [20.0, 20.0]) == [15.0, 10.0]


def test_compute_debt_accumulated_exceeds_target():
    assert compute_debt([30, 5], [20.0, 10.0]) == [0.0, 5.0]


def test_compute_debt_exact_match():
    assert compute_debt([20, 10], [20.0, 10.0]) == [0.0, 0.0]


def test_compute_debt_negative_target_not_clamped():
    result = compute_debt([5], [2.0])
    assert result == [0.0]


def test_compute_debt_float_accumulated():
    result = compute_debt([5], [3.0])
    assert result == [0.0]


def test_compute_debt_large_values():
    result = compute_debt([1000, 0], [500.0, 500.0])
    assert result == [0.0, 500.0]


def test_compute_debt_empty():
    assert compute_debt([], []) == []


# -- ExposureTracker ----------------------------------------------------------

def test_init_zero():
    tracker = ExposureTracker(num_classes=3)
    assert tracker.accumulated() == [0, 0, 0]


def test_single_class():
    tracker = ExposureTracker(num_classes=1)
    assert tracker.accumulated() == [0]
    tracker.record(0)
    assert tracker.accumulated() == [1]


def test_record_int():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(1)
    assert tracker.accumulated() == [0, 1, 0]


def test_record_tensor():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(torch.tensor(2))
    assert tracker.accumulated() == [0, 0, 1]


def test_record_count_greater_than_one():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=5)
    assert tracker.accumulated() == [5, 0, 0]


def test_record_multiple_classes():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=3)
    tracker.record(1)
    tracker.record(2, count=2)
    assert tracker.accumulated() == [3, 1, 2]


def test_accumulated_returns_copy():
    tracker = ExposureTracker(num_classes=2)
    acc = tracker.accumulated()
    acc[0] = 99
    assert tracker.accumulated() == [0, 0]


def test_debt_zero_target():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=5)
    assert tracker.debt([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_debt_positive():
    tracker = ExposureTracker(num_classes=2)
    tracker.record(0, count=3)
    assert tracker.debt([10.0, 10.0]) == [7.0, 10.0]


def test_debt_accumulated_exceeds_target():
    tracker = ExposureTracker(num_classes=2)
    tracker.record(0, count=15)
    assert tracker.debt([10.0, 20.0]) == [0.0, 20.0]


def test_reset():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=10)
    tracker.record(1, count=5)
    tracker.reset()
    assert tracker.accumulated() == [0, 0, 0]


def test_reset_then_record():
    tracker = ExposureTracker(num_classes=2)
    tracker.record(0, count=10)
    tracker.reset()
    tracker.record(1, count=3)
    assert tracker.accumulated() == [0, 3]


def test_record_batch():
    tracker = ExposureTracker(num_classes=3)
    labels = [torch.tensor(0), torch.tensor(2), torch.tensor(0)]
    tracker.record_batch(labels)
    assert tracker.accumulated() == [2, 0, 1]


def test_record_batch_empty():
    tracker = ExposureTracker(num_classes=3)
    tracker.record_batch([])
    assert tracker.accumulated() == [0, 0, 0]


def test_record_batch_mixed_types():
    tracker = ExposureTracker(num_classes=2)
    labels = [0, 1]
    tracker.record_batch(labels)
    assert tracker.accumulated() == [1, 1]


def test_state_dict_roundtrip():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=5)
    tracker.record(2, count=3)
    state = tracker.state_dict()
    assert state == {"accumulated": [5, 0, 3]}

    tracker2 = ExposureTracker(num_classes=3)
    tracker2.load_state_dict(state)
    assert tracker2.accumulated() == [5, 0, 3]


def test_state_dict_after_reset():
    tracker = ExposureTracker(num_classes=2)
    tracker.record(0, count=10)
    tracker.reset()
    state = tracker.state_dict()
    assert state == {"accumulated": [0, 0]}


def test_debt_after_partial_records():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=10)
    tracker.record(1, count=5)
    result = tracker.debt([20.0, 20.0, 20.0])
    assert result[0] == 10.0
    assert result[1] == 15.0
    assert result[2] == 20.0


def test_delegate_debt_to_compute_debt():
    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=7)
    direct = compute_debt(tracker.accumulated(), [10.0, 10.0, 10.0])
    via_tracker = tracker.debt([10.0, 10.0, 10.0])
    assert direct == via_tracker
