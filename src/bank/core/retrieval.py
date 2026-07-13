from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def sample_by_allocation(
    bank: Mapping[int, list],
    allocation: Sequence[int],
    rng: random.Random,
) -> list:
    """Retrieve items from ``bank`` following the per-class ``allocation``.

    Logs a warning when a class with non-zero allocation has an empty pool.
    """
    retrieved: list = []
    for class_id, count in enumerate(allocation):
        if count == 0:
            continue
        pool = bank.get(class_id, [])
        if not pool:
            LOGGER.warning(
                "sample_by_allocation: class %d has empty pool (allocation=%d), skipping",
                class_id,
                count,
            )
            continue
        retrieved.extend(rng.choices(pool, k=count))
    return retrieved


def sample_uniform(
    bank: Mapping[int, list],
    budget: int,
    rng: random.Random,
) -> list:
    """Retrieve ``budget`` items uniformly at random across non-empty pools."""
    classes = [c for c, pool in bank.items() if pool]
    if not classes:
        return []

    total_pool_size = sum(len(bank[c]) for c in classes)
    if total_pool_size == 0:
        return []

    c = rng.choices(classes, k=budget)
    return [rng.choice(bank[cls]) for cls in c]
