"""Tests for accuracy-matrix serialization and cross-seed aggregation."""

import math

import numpy as np
import pytest

from studies.runner.cifar100.metrics import (
    aggregate_matrices,
    average_accuracy,
    matrix_to_csv,
)


class TestMatrixToCsv:
    def test_basic_rows(self):
        csv_text = matrix_to_csv([[0.1, 0.2], [0.3, 0.4]])
        assert csv_text == "0.1,0.2\n0.3,0.4\n"

    def test_nan_cells_serialized_as_nan(self):
        csv_text = matrix_to_csv([[0.5, float("nan")], [0.6, 0.7]])
        lines = csv_text.strip().split("\n")
        assert lines[0].endswith(",nan")
        assert lines[1] == "0.6,0.7"

    def test_accepts_numpy_array(self):
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        assert matrix_to_csv(arr) == "0.1,0.2\n0.3,0.4\n"

    def test_empty_matrix(self):
        assert matrix_to_csv([]) == "\n"


class TestAggregateMatrices:
    def test_mean_and_std_across_seeds(self):
        m1 = [[0.1, float("nan")], [0.2, 0.3]]
        m2 = [[0.3, float("nan")], [0.4, 0.5]]
        mean, std = aggregate_matrices([m1, m2])
        assert mean[0, 0] == pytest.approx(0.2)
        assert mean[1, 0] == pytest.approx(0.3)
        assert mean[1, 1] == pytest.approx(0.4)
        assert math.isnan(mean[0, 1])
        assert std[0, 0] == pytest.approx(0.1)
        assert math.isnan(std[0, 1])

    def test_all_nan_cell_stays_nan(self):
        m1 = [[float("nan"), float("nan")], [0.2, 0.3]]
        m2 = [[float("nan"), float("nan")], [0.4, 0.5]]
        mean, std = aggregate_matrices([m1, m2])
        assert math.isnan(mean[0, 0])
        assert math.isnan(std[0, 0])
        assert mean[1, 1] == pytest.approx(0.4)

    def test_aggregated_mean_matches_scalar_average_accuracy(self):
        m1 = [[0.1, float("nan")], [0.2, 0.3]]
        m2 = [[0.5, float("nan")], [0.6, 0.7]]
        mean, _ = aggregate_matrices([m1, m2])
        row_mean = float(np.nanmean(mean[-1, :]))
        expected = (average_accuracy(m1) + average_accuracy(m2)) / 2
        assert abs(row_mean - expected) < 1e-12
