from __future__ import annotations

import math
from collections.abc import Sequence


def allocate_by_debt(
    debt: Sequence[float],
    budget: int,
    temperature: float = 1.0,
) -> list[int]:
    """Allocate ``budget`` items across classes proportionally to debt.

    Uses the largest-remainder method to ensure the sum of allocations
    exactly equals ``budget``.

    ``temperature`` controls how aggressively allocation concentrates
    on high-debt classes:
        temperature → ∞  : uniform allocation (all classes equal)
        temperature → 1  : proportional to debt (default)
        temperature → 0+ : hard-max (all budget to the highest-debt class)
    """
    if budget < 0:
        raise ValueError(f"retrieval budget must be non-negative, got {budget}")

    n = len(debt)
    if budget == 0:
        return [0] * n

    if temperature < 1e-6:
        # Hard-max: all budget to the class with highest debt
        max_debt = max(debt)
        if max_debt <= 0:
            return [0] * n
        argmax = max(range(n), key=lambda i: debt[i])
        alloc = [0] * n
        alloc[argmax] = budget
        return alloc

    if temperature > 1e6:
        # Uniform
        return _allocate_uniform(budget, n)

    if sum(debt) <= 0:
        return [0] * n

    # Softmax-weighted allocation
    if abs(temperature - 1.0) > 1e-6:
        scaled = [d / temperature for d in debt]
        max_s = max(scaled)
        weights = [math.exp(s - max_s) for s in scaled]
        total_w = sum(weights)
        raw = [budget * w / total_w for w in weights]
    else:
        raw = [budget * d / sum(debt) for d in debt]

    base = [int(v) for v in raw]
    remaining = budget - sum(base)

    order = sorted(
        range(n),
        key=lambda i: (raw[i] - base[i], debt[i]),
        reverse=True,
    )
    for i in order[:remaining]:
        base[i] += 1

    return base


def _allocate_uniform(budget: int, n: int) -> list[int]:
    if n == 0:
        return []
    base = [budget // n] * n
    for i in range(budget % n):
        base[i] += 1
    return base
