#!/usr/bin/env python3
"""Property-based verification of bank core invariants.

These tests check mathematical properties that must hold for all valid inputs,
using many random seeds to stress-test the implementation.

Properties verified:
    1. Debt monotonicity: more accumulated exposure -> less or equal debt
    2. Allocation sum invariance: sum(allocate_by_debt(debt, budget)) == budget
    3. Zero-debt allocation: all debts zero -> all allocations zero
    4. Non-negativity: all debts >= 0, all allocations >= 0
    5. Proportionality: higher debt -> >= allocation of lower debt
    6. Remainder bound: each class allocation differs from exact share by < 1

Usage:
    python verification/verify_bank.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bank.core.exposure import compute_debt
from src.bank.core.allocator import allocate_by_debt


NUM_TRIALS = 500
MAX_CLASSES = 20
MAX_ACCUMULATED = 10_000
MAX_BUDGET = 100


def generate_random_debt(rng: random.Random, num_classes: int) -> list[float]:
    """Generate random debt values, with some zeros for edge coverage."""
    debt = []
    for _ in range(num_classes):
        if rng.random() < 0.15:
            debt.append(0.0)
        elif rng.random() < 0.3:
            debt.append(round(rng.uniform(0.1, 10.0), 2))
        else:
            debt.append(round(rng.uniform(0.1, 100.0), 2))
    return debt


def generate_random_accumulated_and_target(
    rng: random.Random, num_classes: int,
) -> tuple[list[int], list[float]]:
    """Generate random accumulated counts and target exposures."""
    accumulated = [rng.randint(0, MAX_ACCUMULATED) for _ in range(num_classes)]
    target = [round(rng.uniform(0.0, 500.0), 1) for _ in range(num_classes)]
    return accumulated, target


# -- Property 1: Debt monotonicity -------------------------------------------

def verify_debt_monotonicity() -> list[str]:
    """If accumulated increases (all else equal), debt must not increase."""
    errors: list[str] = []
    rng = random.Random(0)

    for trial in range(NUM_TRIALS):
        num_classes = rng.randint(1, MAX_CLASSES)
        target = [round(rng.uniform(1.0, 200.0), 1) for _ in range(num_classes)]
        base_accumulated = [rng.randint(0, 100) for _ in range(num_classes)]

        base_debt = compute_debt(base_accumulated, target)

        for c in range(num_classes):
            increased = list(base_accumulated)
            increased[c] += rng.randint(1, 50)
            new_debt = compute_debt(increased, target)

            if new_debt[c] > base_debt[c] + 1e-12:
                errors.append(
                    f"Trial {trial}, class {c}: debt increased from {base_debt[c]} "
                    f"to {new_debt[c]} after accumulated {base_accumulated[c]} -> {increased[c]}"
                )
                if len(errors) >= 10:
                    return errors

    return errors


# -- Property 2: Allocation sum invariance ------------------------------------

def verify_allocation_sum() -> list[str]:
    """sum(allocate_by_debt(debt, budget)) == budget when debt > 0."""
    errors: list[str] = []
    rng = random.Random(1)

    for trial in range(NUM_TRIALS):
        num_classes = rng.randint(1, MAX_CLASSES)
        debt = generate_random_debt(rng, num_classes)
        if sum(debt) == 0:
            continue
        budget = rng.randint(0, MAX_BUDGET)

        alloc = allocate_by_debt(debt, budget)

        if sum(alloc) != budget:
            errors.append(
                f"Trial {trial}: sum(alloc)={sum(alloc)} != budget={budget}, "
                f"debt={debt}"
            )
            if len(errors) >= 10:
                return errors

    return errors


# -- Property 3: Zero-debt allocation -----------------------------------------

def verify_zero_debt_allocation() -> list[str]:
    """If all debts are zero, all allocations must be zero."""
    errors: list[str] = []
    rng = random.Random(2)

    for trial in range(NUM_TRIALS):
        num_classes = rng.randint(1, MAX_CLASSES)
        budget = rng.randint(0, MAX_BUDGET)
        all_zero = [0.0] * num_classes

        for b in [0, budget]:
            alloc = allocate_by_debt(all_zero, b)
            if any(a != 0 for a in alloc):
                errors.append(
                    f"Trial {trial}, budget={b}: zero debt gave non-zero alloc {alloc}"
                )
                if len(errors) >= 5:
                    return errors

    return errors


# -- Property 4: Non-negativity ----------------------------------------------

def verify_non_negativity() -> list[str]:
    """All debts >= 0 and all allocations >= 0."""
    errors: list[str] = []
    rng = random.Random(3)

    for trial in range(NUM_TRIALS):
        num_classes = rng.randint(1, MAX_CLASSES)
        accumulated, target = generate_random_accumulated_and_target(rng, num_classes)

        debt = compute_debt(accumulated, target)
        for c, d in enumerate(debt):
            if d < -1e-12:
                errors.append(f"Trial {trial}, class {c}: negative debt {d}")
                break

        budget = rng.randint(0, MAX_BUDGET)
        alloc = allocate_by_debt(debt, budget)
        for c, a in enumerate(alloc):
            if a < 0:
                errors.append(f"Trial {trial}, class {c}: negative allocation {a}")
                break

        if errors:
            return errors

    return errors


# -- Property 5: Proportionality (debt-ordered) -----------------------------

def verify_proportionality() -> list[str]:
    """Higher debt should receive >= allocation of lower debt."""
    errors: list[str] = []
    rng = random.Random(4)

    for trial in range(NUM_TRIALS):
        num_classes = rng.randint(2, MAX_CLASSES)
        debt = generate_random_debt(rng, num_classes)
        budget = rng.randint(1, MAX_BUDGET)

        alloc = allocate_by_debt(debt, budget)

        for i in range(num_classes):
            for j in range(num_classes):
                if debt[i] > debt[j] + 1e-12 and alloc[i] < alloc[j]:
                    errors.append(
                        f"Trial {trial}: debt[{i}]={debt[i]} > debt[{j}]={debt[j]} "
                        f"but alloc[{i}]={alloc[i]} < alloc[{j}]={alloc[j]}"
                    )
                    if len(errors) >= 10:
                        return errors

    return errors


# -- Property 6: Remainder bound ---------------------------------------------

def verify_remainder_bound() -> list[str]:
    """Each class allocation differs from exact proportional share by < 1."""
    errors: list[str] = []
    rng = random.Random(5)

    for trial in range(NUM_TRIALS):
        num_classes = rng.randint(1, MAX_CLASSES)
        debt = generate_random_debt(rng, num_classes)
        total_debt = sum(debt)
        if total_debt == 0:
            continue

        budget = rng.randint(1, MAX_BUDGET)
        alloc = allocate_by_debt(debt, budget)

        for c in range(num_classes):
            exact = budget * debt[c] / total_debt
            diff = abs(alloc[c] - exact)
            if diff >= 1.0 + 1e-12:
                errors.append(
                    f"Trial {trial}, class {c}: alloc={alloc[c]}, exact={exact:.4f}, "
                    f"diff={diff:.4f} >= 1"
                )
                if len(errors) >= 10:
                    return errors

    return errors


# -- Main ---------------------------------------------------------------------

def main() -> int:
    all_errors: list[str] = []

    print("=" * 60)
    print("Property-Based Bank Verification")
    print(f"  {NUM_TRIALS} random trials per property")
    print("=" * 60)

    properties = [
        ("1. Debt monotonicity", verify_debt_monotonicity),
        ("2. Allocation sum invariance", verify_allocation_sum),
        ("3. Zero-debt -> zero allocation", verify_zero_debt_allocation),
        ("4. Non-negativity (debts & allocations)", verify_non_negativity),
        ("5. Proportionality (higher debt -> >= allocation)", verify_proportionality),
        ("6. Remainder bound (|alloc - exact| < 1)", verify_remainder_bound),
    ]

    for name, func in properties:
        print(f"\n  [{name}]")
        errors = func()
        if errors:
            print(f"    FAIL  -  {len(errors)} violation(s):")
            for err in errors[:5]:
                print(f"      • {err}")
            if len(errors) > 5:
                print(f"      … and {len(errors) - 5} more")
            all_errors.extend(errors)
        else:
            print("    PASS")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"RESULT: FAIL  -  {len(all_errors)} property violation(s)")
        return 1
    else:
        print("RESULT: PASS  -  all properties hold")
        return 0


if __name__ == "__main__":
    sys.exit(main())
