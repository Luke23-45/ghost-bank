#!/usr/bin/env python3
"""Verify that the implementation matches the formal mathematical definition.

Reference: ``docs/formal_math_definition.md``

Sections verified:
    6. Minority-Class Exposure Interpretation  -> ExposureTracker
    7. Exposure Debt                          -> compute_debt, allocate_by_debt

Usage:
    python verification/verify_formal_definition.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.bank.core.exposure import ExposureTracker, compute_debt
from src.bank.core.allocator import allocate_by_debt


# ---------------------------------------------------------------------------
# Section 6  -  Exposure Tracking
# Formal: A_c(t) = sum_{s=1}^t a_c(s)  where a_c(s) = sum_{(x,y) in B_s} 1[y=c]
# ---------------------------------------------------------------------------

def verify_exposure_tracking() -> list[str]:
    errors: list[str] = []

    tracker = ExposureTracker(num_classes=3)
    acc = tracker.accumulated()
    if acc != [0, 0, 0]:
        errors.append(f"Initial A_c(0) should be [0,0,0], got {acc}")

    tracker.record(0, count=3)
    tracker.record(1, count=2)
    acc = tracker.accumulated()
    if acc != [3, 2, 0]:
        errors.append(f"After records A_c should be [3,2,0], got {acc}")

    tracker.record(0, count=1)
    acc = tracker.accumulated()
    if acc != [4, 2, 0]:
        errors.append(f"After increment A_c should be [4,2,0], got {acc}")

    tracker.reset()
    acc = tracker.accumulated()
    if acc != [0, 0, 0]:
        errors.append(f"After reset A_c should be [0,0,0], got {acc}")

    tracker.record(2, count=7)
    acc = tracker.accumulated()
    if acc != [0, 0, 7]:
        errors.append(f"Single class record failed, got {acc}")

    return errors


# ---------------------------------------------------------------------------
# Section 7  -  Exposure Debt
# Formal: D_c(t) = max(0, T_c(t) - A_c(t))
# ---------------------------------------------------------------------------

def verify_debt_formula() -> list[str]:
    errors: list[str] = []

    A = [0, 0, 0]
    T = [10.0, 10.0, 10.0]
    D = compute_debt(A, T)
    expected = [10.0, 10.0, 10.0]
    for i, (d, e) in enumerate(zip(D, expected)):
        if abs(d - e) > 1e-12:
            errors.append(f"Debt for class {i}: expected {e}, got {d}")

    A = [10, 5, 0]
    T = [10.0, 10.0, 10.0]
    D = compute_debt(A, T)
    expected = [0.0, 5.0, 10.0]
    for i, (d, e) in enumerate(zip(D, expected)):
        if abs(d - e) > 1e-12:
            errors.append(f"Partial debt test class {i}: expected {e}, got {d}")

    A = [20, 20, 20]
    T = [10.0, 10.0, 10.0]
    D = compute_debt(A, T)
    expected = [0.0, 0.0, 0.0]
    for i, (d, e) in enumerate(zip(D, expected)):
        if abs(d - e) > 1e-12:
            errors.append(f"Exceeded debt test class {i}: expected {e}, got {d}")

    A = [5]
    T = [3.0]
    D = compute_debt(A, T)
    if D != [0.0]:
        errors.append(f"Negative debt should clamp to 0, got {D}")

    return errors


# ---------------------------------------------------------------------------
# Section 7  -  Budget Allocation
# Formal: r_c(t) = floor(R * D_c(t) / sum_j D_j(t))  + remainder distribution
# ---------------------------------------------------------------------------

def verify_allocation_formula() -> list[str]:
    errors: list[str] = []

    D = [10.0, 10.0, 10.0]
    R = 9
    alloc = allocate_by_debt(D, R)
    if sum(alloc) != R:
        errors.append(f"Allocation sum {sum(alloc)} != budget {R}")
    if alloc != [3, 3, 3]:
        errors.append(f"Equal debt should give equal allocation, got {alloc}")

    D = [1.0, 3.0]
    R = 8
    alloc = allocate_by_debt(D, R)
    if sum(alloc) != R:
        errors.append(f"Allocation sum {sum(alloc)} != budget {R}")
    if alloc[1] != 3 * alloc[0]:
        errors.append(f"Proportionality violated: {alloc} for debts {D}")

    D = [0.0, 10.0, 0.0]
    R = 5
    alloc = allocate_by_debt(D, R)
    if alloc[0] != 0 or alloc[2] != 0:
        errors.append(f"Zero debt classes should get 0, got {alloc}")
    if alloc[1] != R:
        errors.append(f"Only non-zero debt class should get all {R}, got {alloc[1]}")

    D = [7.0, 3.0]
    R = 0
    alloc = allocate_by_debt(D, R)
    if alloc != [0, 0]:
        errors.append(f"Zero budget should give zero allocation, got {alloc}")

    D = [7.0, 3.0]
    R = 10
    alloc = allocate_by_debt(D, R)
    if sum(alloc) != R:
        errors.append(f"Sum check failed: {sum(alloc)} != {R}")
    if alloc[0] + alloc[1] != R:
        errors.append(f"Total allocation != budget: {alloc}")

    return errors


# ---------------------------------------------------------------------------
# Integration: e2e trace from formal section 7
# ---------------------------------------------------------------------------

def verify_integration_trace() -> list[str]:
    errors: list[str] = []

    tracker = ExposureTracker(num_classes=3)
    tracker.record(0, count=100)
    tracker.record(1, count=30)
    tracker.record(2, count=5)

    T_t = [50.0, 50.0, 50.0]
    A_t = tracker.accumulated()
    D_t = tracker.debt(T_t)

    expected_debt = [0.0, 20.0, 45.0]
    for i, (d, e) in enumerate(zip(D_t, expected_debt)):
        if abs(d - e) > 1e-12:
            errors.append(f"Integration debt class {i}: expected {e}, got {d}")

    R = 10
    alloc = allocate_by_debt(D_t, R)
    if sum(alloc) != R:
        errors.append(f"Integration allocation sum {sum(alloc)} != {R}")

    for i, a in enumerate(alloc):
        if D_t[i] == 0.0 and a != 0:
            errors.append(f"Class {i} has 0 debt but got {a} allocation")
            break
    else:
        if alloc[1] > 0 and alloc[2] < alloc[1]:
            errors.append(f"Higher debt class should get >= lower debt class: {alloc}")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    all_errors: list[str] = []

    print("=" * 60)
    print("Formal Definition Verification  -  Ghost Bank")
    print("=" * 60)

    sections = [
        ("6. Exposure Tracking", verify_exposure_tracking),
        ("7. Debt Formula (D_c = max(0, T - A))", verify_debt_formula),
        ("7. Budget Allocation (r_c = floor(R * D_c / sum(D)))", verify_allocation_formula),
        ("Integration: full trace", verify_integration_trace),
    ]

    for name, func in sections:
        print(f"\n  [{name}]")
        errors = func()
        if errors:
            print(f"    FAIL  -  {len(errors)} error(s):")
            for err in errors:
                print(f"      • {err}")
            all_errors.extend(errors)
        else:
            print("    PASS")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"RESULT: FAIL  -  {len(all_errors)} verification(s) failed")
        return 1
    else:
        print("RESULT: PASS  -  all formal definitions verified")
        return 0


if __name__ == "__main__":
    sys.exit(main())
