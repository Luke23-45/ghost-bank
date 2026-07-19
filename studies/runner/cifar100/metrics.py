from __future__ import annotations

import numpy as np


def average_accuracy(acc_matrix: list[list[float]]) -> float:
    acc_matrix = _to_rect(acc_matrix)
    final_row = acc_matrix[-1, :]
    return float(np.nanmean(final_row))


def forgetting(acc_matrix: list[list[float]]) -> float:
    acc_matrix = _to_rect(acc_matrix)
    n_tasks = acc_matrix.shape[1]
    forget_vals = []
    for i in range(n_tasks - 1):
        col = acc_matrix[:, i]
        peak = float(np.nanmax(col))
        final = float(col[-1])
        forget_vals.append(peak - final)
    return float(np.mean(forget_vals)) if forget_vals else 0.0


def backward_transfer(acc_matrix: list[list[float]]) -> float:
    acc_matrix = _to_rect(acc_matrix)
    n_tasks = acc_matrix.shape[1]
    bwt_vals = []
    for i in range(n_tasks - 1):
        first = acc_matrix[i, i]
        final = acc_matrix[-1, i]
        bwt_vals.append(final - first)
    return float(np.mean(bwt_vals)) if bwt_vals else 0.0


def _to_rect(acc_matrix: list[list[float]]) -> np.ndarray:
    max_cols = max(len(row) for row in acc_matrix) if acc_matrix else 0
    rect = np.full((len(acc_matrix), max_cols), np.nan)
    for i, row in enumerate(acc_matrix):
        rect[i, : len(row)] = row
    return rect
