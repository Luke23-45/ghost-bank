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


def allocate_fixed_total(
    num_classes: int,
    total_budget: int,
    probe_scores: Sequence[float] | None = None,
    gamma: float = 0.5,
    beta: float = 1.0,
    floor: int = 1,
) -> list[int]:
    """Allocate ``total_budget`` exemplars across ``num_classes`` classes.

    The allocation weight for class ``c`` is a mixture of a uniform prior
    and a probe-driven term:

        w_c = (1 - gamma) / N  +  gamma * softmax(beta * r_c)

    where ``r_c`` is the normalised probe score for class ``c`` and
    ``N = num_classes``.

    After weighting, each class receives at least ``floor`` exemplars.
    The allocations are then clipped and renormalised so the sum exactly
    equals ``total_budget``.

    Parameters
    ----------
    num_classes : int
        Total number of seen classes.
    total_budget : int
        Total memory budget M across all classes.
    probe_scores : Sequence[float] or None
        Per-class probe-derived importance scores.  When None, allocation
        is uniform.
    gamma : float
        Mixture weight (0 = fully uniform, 1 = fully probe-driven).
    beta : float
        Sharpness of the softmax over probe scores.
    floor : int
        Minimum per-class allocation.

    Returns
    -------
    list[int]
        Per-class exemplar counts summing to ``total_budget``.
    """
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}")
    if num_classes <= 0:
        return []

    uniform_weight = 1.0 / num_classes

    if probe_scores is None or gamma <= 0.0:
        weights = [uniform_weight] * num_classes
    else:
        scores = list(probe_scores)
        n = len(scores)
        beta_scores = [s * beta for s in scores]
        max_s = max(beta_scores) if beta_scores else 0.0
        exp_s = [math.exp(s - max_s) for s in beta_scores]
        sum_exp = sum(exp_s) if exp_s else 1.0
        probe_weight = [e / sum_exp for e in exp_s]

        weights = [
            (1.0 - gamma) * uniform_weight + gamma * pw
            for pw in probe_weight
        ]

    raw = [total_budget * w for w in weights]
    clipped = [max(floor, int(v)) for v in raw]

    current_total = sum(clipped)
    if current_total == 0:
        if total_budget == 0:
            return [0] * num_classes
        per_class = max(floor, total_budget // num_classes)
        alloc = [per_class] * num_classes
        for i in range(total_budget - sum(alloc)):
            alloc[i] += 1
        return alloc

    if current_total > total_budget:
        surplus = current_total - total_budget
        for i in sorted(range(num_classes), key=lambda i: clipped[i] - raw[i]):
            if surplus <= 0:
                break
            reduction = min(clipped[i] - floor, surplus)
            clipped[i] -= reduction
            surplus -= reduction
    elif current_total < total_budget:
        deficit = total_budget - current_total
        for i in sorted(range(num_classes), key=lambda i: raw[i] - clipped[i], reverse=True):
            if deficit <= 0:
                break
            clipped[i] += 1
            deficit -= 1

    return clipped


def allocate_uniform_fixed_total(
    num_classes: int,
    total_budget: int,
    floor: int = 1,
) -> list[int]:
    """Uniform allocation under fixed total memory with per-class floor.

    Each class gets at least ``floor`` exemplars; remaining budget is
    distributed uniformly.

    This is the baseline against which probe-guided allocation is compared.
    """
    return allocate_fixed_total(
        num_classes, total_budget,
        probe_scores=None, gamma=0.0, beta=1.0, floor=floor,
    )
