from __future__ import annotations

from collections.abc import Sequence


def allocate_by_debt(debt: Sequence[float], budget: int) -> list[int]:
    """Allocate ``budget`` items across classes proportionally to debt.

    Uses the largest-remainder method to ensure the sum of allocations
    exactly equals ``budget``.
    """
    if budget < 0:
        raise ValueError(f"retrieval budget must be non-negative, got {budget}")

    total_debt = sum(debt)
    if total_debt == 0 or budget == 0:
        return [0] * len(debt)

    raw = [budget * d / total_debt for d in debt]
    base = [int(v) for v in raw]
    remaining = budget - sum(base)

    order = sorted(
        range(len(debt)),
        key=lambda i: (raw[i] - base[i], debt[i]),
        reverse=True,
    )
    for i in order[:remaining]:
        base[i] += 1

    return base
