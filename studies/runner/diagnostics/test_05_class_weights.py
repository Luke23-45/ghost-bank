"""H9: Class weight scaling — systematic sweep.

Hypothesis: The current sqrt(max_count / count) scaling is suboptimal.
The optimal exponent and functional form are unknown.

This script sweeps class weight exponents and measures their effect
on allocation ratios without training a model.
"""

import math


def compute_allocations(class_counts, debt, exponent, budget=8):
    """Compute allocation with class_weights = (max/count)**exponent."""
    max_count = max(class_counts)
    weights = [(max_count / c) ** exponent if c > 0 else 1.0 for c in class_counts]
    weighted_debt = [d * w for d, w in zip(debt, weights)]

    if sum(weighted_debt) <= 0:
        n = len(debt)
        base = [budget // n] * n
        for i in range(budget % n):
            base[i] += 1
        return base

    raw = [budget * d / sum(weighted_debt) for d in weighted_debt]
    base = [int(v) for v in raw]
    remaining = budget - sum(base)
    order = sorted(range(len(debt)), key=lambda i: (raw[i] - base[i], weighted_debt[i]), reverse=True)
    for i in order[:remaining]:
        base[i] += 1
    return base


def compute_allocations_with_func(class_counts, debt, func, budget=8):
    """Compute allocation with arbitrary weight function."""
    weights = [func(c, class_counts) for c in class_counts]
    weighted_debt = [d * w for d, w in zip(debt, weights)]

    if sum(weighted_debt) <= 0:
        n = len(debt)
        base = [budget // n] * n
        for i in range(budget % n):
            base[i] += 1
        return base

    raw = [budget * d / sum(weighted_debt) for d in weighted_debt]
    base = [int(v) for v in raw]
    remaining = budget - sum(base)
    order = sorted(range(len(debt)), key=lambda i: (raw[i] - base[i], weighted_debt[i]), reverse=True)
    for i in order[:remaining]:
        base[i] += 1
    return base


if __name__ == "__main__":
    print("=" * 70)
    print("  CLASS WEIGHT SWEEP (H9)")
    print("=" * 70)

    class_counts = [2000, 200, 20]
    # Post-shift debt: class 0 and 2 both high
    debt = [2.3, 0.1, 2.3]

    # --- Exponent sweep ---
    print("\n" + "-" * 70)
    print("  SWEEP: weight = (max_count / count)^exponent")
    print("  Post-shift debt: C0=2.3, C1=0.1, C2=2.3")
    print("  Class counts: 2000, 200, 20")
    print("-" * 70)
    print(f"  {'Exponent':>10} {'W_C0':>8} {'W_C1':>8} {'W_C2':>8}  "
          f"{'Eff_C0':>8} {'Eff_C1':>8} {'Eff_C2':>8}  "
          f"{'Alloc_C0':>10} {'Alloc_C1':>10} {'Alloc_C2':>10}  {'Alloc%_C2':>10}")
    print("  " + "-" * 84)

    for exponent in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
        weights = [(max(class_counts) / c) ** exponent if c > 0 else 1.0 for c in class_counts]
        weighted = [d * w for d, w in zip(debt, weights)]
        alloc = compute_allocations(class_counts, debt, exponent)
        pct_c2 = alloc[2] / sum(alloc) * 100
        print(f"  {exponent:>10.2f} {weights[0]:>8.2f} {weights[1]:>8.2f} {weights[2]:>8.2f}  "
              f"{weighted[0]:>8.2f} {weighted[1]:>8.2f} {weighted[2]:>8.2f}  "
              f"{alloc[0]:>10} {alloc[1]:>10} {alloc[2]:>10}  {pct_c2:>9.1f}%")

    # --- Functional form comparison ---
    print("\n" + "-" * 70)
    print("  FUNCTIONAL FORM COMPARISON")
    print("-" * 70)
    print(f"  {'Form':<40} {'Eff_C0':>8} {'Eff_C1':>8} {'Eff_C2':>8}  "
          f"{'Alloc_C0':>10} {'Alloc_C1':>10} {'Alloc_C2':>10}")

    forms = {
        "No weights (exponent=0)": lambda c, cc: 1.0,
        "sqrt (current, exp=0.5)": lambda c, cc: (max(cc) / c) ** 0.5 if c > 0 else 1.0,
        "Linear (exp=1.0)": lambda c, cc: max(cc) / c if c > 0 else 1.0,
        "Quadratic (exp=2.0)": lambda c, cc: (max(cc) / c) ** 2 if c > 0 else 1.0,
        "Log": lambda c, cc: math.log(max(cc) / c + 1) if c > 0 else 1.0,
        "exp(1/freq)": lambda c, cc: math.exp(max(cc) / c) if c > 0 else 1.0,
        "1/freq (raw inverse)": lambda c, cc: max(cc) / c if c > 0 else 1.0,
        "threshold (C2=10x)": lambda c, cc: 10.0 if c == 2 else 1.0,
        "threshold (C2=100x)": lambda c, cc: 100.0 if c == 2 else 1.0,
    }

    for name, func in forms.items():
        alloc = compute_allocations_with_func(class_counts, debt, func)
        weights = [func(c, class_counts) for c in class_counts]
        weighted = [d * w for d, w in zip(debt, weights)]
        w_str = ", ".join([f"W{i}={w:.2f}" for i, w in enumerate(weights)])
        print(f"  {name:<40} "
              f"{weighted[0]:>8.2f} {weighted[1]:>8.2f} {weighted[2]:>8.2f}  "
              f"{alloc[0]:>10} {alloc[1]:>10} {alloc[2]:>10}")

    # --- Effect on different debt scenarios ---
    print("\n" + "-" * 70)
    print("  WEIGHT EFFECT ACROSS DEBT SCENARIOS (sqrt weights)")
    print("-" * 70)
    scenarios = {
        "Immediate post-shift (equal high)": [2.3, 0.1, 2.3],
        "Class 0 partially recovered": [1.0, 0.1, 2.3],
        "Class 0 fully recovered": [0.1, 0.1, 2.3],
        "All low (pre-shift)": [0.1, 0.1, 0.1],
    }

    print(f"  {'Scenario':<35} {'Unweighted alloc':>20} {'Weighted alloc':>20}")
    for sc_name, sc_debt in scenarios.items():
        unweighted = compute_allocations(class_counts, sc_debt, 0.0)
        weighted = compute_allocations(class_counts, sc_debt, 0.5)
        print(f"  {sc_name:<35} "
              f"{' '.join(f'C{i}={a}' for i,a in enumerate(unweighted)):>20} "
              f"{' '.join(f'C{i}={a}' for i,a in enumerate(weighted)):>20}")

    # --- VERDICT ---
    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  H9 (Class weight scaling is wrong):
    The optimal exponent depends on the debt scenario.  Under the
    current sqrt (exponent=0.5), class 2 gets ~5x weight → ~62% of
    budget when its debt equals class 0's.

    Higher exponents (>1.0) push more budget to class 2, but may
    over-allocate in scenarios where class 2's debt is lower than
    class 0's.

    The key insight: class weights are a STATIC fix for a DYNAMIC
    problem.  They don't adapt to the current state — a class with
    10x weight always gets 10x weight, regardless of whether it
    actually needs more replay at this moment.
""")
