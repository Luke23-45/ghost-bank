"""H5: Proportional allocation is wrong for asymmetric difficulty.

Hypothesis: After label swap (0↔2), both classes 0 and 2 have equally
high loss (both are incorrectly predicted).  Proportional allocation
gives them equal budget.  But class 0 has 2000 training samples/epoch
and adapts quickly, while class 2 has 20 samples/epoch and needs MORE
replay.  An allocation mechanism that considers difficulty relative to
available training data would beat proportional.

This script compares allocation mechanisms on the same debt vector:
  1. Proportional (current)
  2. Rank-based (sorted, then weighted ranks)
  3. Softmax (with temperature sweep)
  4. Threshold (only classes with debt > threshold get budget)
  5. Inverse-frequency-weighted (debt * class_weight)
"""

import math

import numpy as np


def allocate_proportional(debt, budget=8):
    if sum(debt) <= 0:
        n = len(debt)
        base = [budget // n] * n
        for i in range(budget % n):
            base[i] += 1
        return base
    raw = [budget * d / sum(debt) for d in debt]
    base = [int(v) for v in raw]
    remaining = budget - sum(base)
    order = sorted(range(len(debt)), key=lambda i: (raw[i] - base[i], debt[i]), reverse=True)
    for i in order[:remaining]:
        base[i] += 1
    return base


def allocate_rank_based(debt, budget=8):
    """Allocate proportional to rank (highest debt = most budget)."""
    if sum(debt) <= 0:
        return allocate_proportional(debt, budget)
    n = len(debt)
    # Rank: highest debt gets rank n, lowest gets rank 1
    sorted_indices = sorted(range(n), key=lambda i: debt[i])
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank
    return allocate_proportional(ranks, budget)


def allocate_softmax(debt, budget=8, temperature=1.0):
    if sum(debt) <= 0:
        n = len(debt)
        base = [budget // n] * n
        for i in range(budget % n):
            base[i] += 1
        return base
    scaled = [d / temperature for d in debt]
    max_s = max(scaled)
    weights = [math.exp(s - max_s) for s in scaled]
    total_w = sum(weights)
    raw = [budget * w / total_w for w in weights]
    base = [int(v) for v in raw]
    remaining = budget - sum(base)
    order = sorted(range(len(debt)), key=lambda i: (raw[i] - base[i], debt[i]), reverse=True)
    for i in order[:remaining]:
        base[i] += 1
    return base


def allocate_threshold(debt, budget=8, threshold_ratio=0.1):
    """Only allocate to classes with debt > max_debt * threshold_ratio."""
    if sum(debt) <= 0:
        return [0] * len(debt)
    max_d = max(debt)
    threshold = max_d * threshold_ratio
    eligible = [i for i, d in enumerate(debt) if d >= threshold]
    if not eligible:
        return [0] * len(debt)
    # Allocate proportionally among eligible
    eligible_debt = [debt[i] for i in eligible]
    total = sum(eligible_debt)
    raw = [0] * len(debt)
    fracs = [budget * d / total for d in eligible_debt]
    base = [int(v) for v in fracs]
    remaining = budget - sum(base)
    for i, idx in enumerate(eligible):
        raw[idx] = base[i]
    # Assign remainder
    for i in sorted(eligible, key=lambda i: debt[i] - raw[i], reverse=True)[:remaining]:
        raw[i] += 1
    return raw


def allocate_inverse_freq(debt, budget=8, class_counts=None):
    """Allocate proportional to debt * (1/freq**power)."""
    if class_counts is None:
        return allocate_proportional(debt, budget)
    max_count = max(class_counts)
    weights = [(max_count / c) ** 0.5 if c > 0 else 1.0 for c in class_counts]
    weighted_debt = [d * w for d, w in zip(debt, weights)]
    return allocate_proportional(weighted_debt, budget)


def simulate_allocation_scenarios():
    """Generate debt vectors for different post-shift scenarios and compare allocators."""

    scenarios = {
        "Post-shift (immediate, step 350)": [2.3, 0.1, 2.3],
        "Post-shift (partial recovery, class 0 adapting)": [1.5, 0.1, 2.3],
        "Post-shift (late recovery)": [0.3, 0.1, 1.8],
        "Steady state (pre-shift)": [0.1, 0.1, 0.1],
        "All equal high loss": [2.0, 2.0, 2.0],
        "One class struggling": [0.1, 0.1, 2.3],
    }

    allocators = {
        "Proportional": lambda d: allocate_proportional(d),
        "Rank-based": lambda d: allocate_rank_based(d),
        "Softmax T=0.1": lambda d: allocate_softmax(d, temperature=0.1),
        "Softmax T=0.5": lambda d: allocate_softmax(d, temperature=0.5),
        "Softmax T=2.0": lambda d: allocate_softmax(d, temperature=2.0),
        "Threshold 20%": lambda d: allocate_threshold(d, threshold_ratio=0.2),
        "Threshold 5%": lambda d: allocate_threshold(d, threshold_ratio=0.05),
        "InvFreq (class_weights)": lambda d: allocate_inverse_freq(
            d, class_counts=[2000, 200, 20]),
    }

    print(f"  {'Scenario':<40} ", end="")
    for name in allocators:
        print(f"{name[:15]:>15}", end="")
    print()

    class_counts = [2000, 200, 20]
    n_allocators = len(allocators)

    for sc_name, debt in scenarios.items():
        print(f"  {'-'* (40 + 15 * n_allocators)}")

        # Show debt vector
        debt_str = ", ".join([f"C{i}={d:.1f}" for i, d in enumerate(debt)])
        print(f"  {sc_name:<40} Debt: {debt_str}")
        class_info = ", ".join([f"C{i}={class_counts[i]}" for i in range(3)])
        print(f"  {'':<40} Samples: {class_info}")

        for name in allocators:
            alloc = allocators[name](debt)
            alloc_str = ", ".join([f"C{i}={a}" for i, a in enumerate(alloc)])
            print(f"  {name:<40} {alloc_str:>15}")

    # === DESIRED ALLOCATION ANALYSIS ===
    print("\n" + "=" * 70)
    print("  DESIRED ALLOCATION: What should PID-CR do?")
    print("=" * 70)
    print("""
    After label swap (epoch 5):
      - Class 0: 2000 samples/epoch, adapts fast (few epochs)
      - Class 2: 20 samples/epoch, adapts slow (many epochs)

    The controller should allocate MORE to class 2 because it needs
    more replay per-sample to recover.  But with proportional allocation,
    both classes with equal loss get EQUAL replay — and per-sample,
    class 0 actually gets MORE replay relative to its abundance.

    Allocation SHOULD consider:
      1. Current loss/debt (how hard is this class right now?)
      2. Available training data (how many fresh samples per epoch?)
      3. How quickly this class adapts given new data

    This suggests the control signal should be:
      effective_debt = debt * (1 / update_frequency)
    or
      effective_debt = debt * (total_samples / class_samples)
    """)


def effective_debt_analysis():
    """Compare different formulations of effective debt."""
    debt = [2.3, 0.1, 2.3]  # Post-shift
    class_counts = [2000, 200, 20]
    total = sum(class_counts)
    update_freq = [c / total for c in class_counts]

    print("\n" + "-" * 70)
    print("  EFFECTIVE DEBT FORMULATIONS")
    print("-" * 70)
    print(f"  {'Formulation':<40} {'C0':>8} {'C1':>8} {'C2':>8} {'Alloc C0':>10} {'Alloc C1':>10} {'Alloc C2':>10}")
    print("  " + "-" * 86)

    formulations = {
        "Raw debt (current)": debt,
        "Debt / freq": [d / f if f > 0 else d for d, f in zip(debt, update_freq)],
        "Debt * (total/class_count)": [d * total / c for d, c in zip(debt, class_counts)],
        "Debt * sqrt(total/class_count)": [d * math.sqrt(total / c) for d, c in zip(debt, class_counts)],
        "Debt / sqrt(freq)": [d / math.sqrt(f) if f > 0 else d for d, f in zip(debt, update_freq)],
        "Debt * (1 if c==2 else 1)": [d * (10.0 if i == 2 else 1.0) for i, d in enumerate(debt)],
    }

    for name, effective in formulations.items():
        alloc = allocate_proportional(effective)
        print(f"  {name:<40} "
              f"{effective[0]:>8.2f} {effective[1]:>8.2f} {effective[2]:>8.2f} "
              f"{alloc[0]:>10} {alloc[1]:>10} {alloc[2]:>10}")


if __name__ == "__main__":
    print("=" * 70)
    print("  ALLOCATION MECHANISM DIAGNOSTIC (H5)")
    print("=" * 70)
    simulate_allocation_scenarios()
    effective_debt_analysis()

    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  H5 (Proportional allocation is wrong):
    Under proportional allocation, equal debt → equal budget.
    But classes with fewer training samples need MORE replay budget
    per unit of debt because they receive less gradient signal from
    fresh data.

    The fix: scale debt by inverse class frequency (or update frequency)
    before allocation.  This is what class_weights attempt, but the
    current sqrt(max_count / count) may be too weak or too strong.

    Alternative: A rank-based allocator that explicitly prioritizes
    high-debt classes, giving the top class 50% of budget regardless
    of exact debt values.
""")
