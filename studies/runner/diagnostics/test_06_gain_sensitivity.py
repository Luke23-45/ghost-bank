"""H10: PID gain hyperparameter sensitivity.

Hypothesis: The default gains (K_p=1.0, K_i=0.1, K_d=0.5) are not
optimal, and tuning them could significantly change PID-CR's behavior.

This test sweeps gains systematically and measures:
  - Response time (steps to reach 80% of max debt after shift)
  - Overshoot (peak debt relative to steady-state)
  - Steady-state allocation (class 2 share of budget)
  - Integral persistence (how long class 0's integral lingers)
"""

import math
import itertools

import numpy as np


def simulate_pid(batches, K_p=1.0, K_i=0.1, K_d=0.5, decay=0.99, smooth=0.9):
    """Simplified PID simulation, returns debts per step."""
    n = 3
    integral = [0.0] * n
    prev_loss = [0.0] * n
    smoothed_loss = [0.0] * n
    all_debts = []

    for b in batches:
        classes = b["classes"]
        shifted = b["shifted"]

        per_class_loss = []
        for c in range(n):
            if c in classes:
                if shifted:
                    per_class_loss.append(2.3)  # High loss after shift
                else:
                    per_class_loss.append(0.1)  # Low loss before shift
            else:
                per_class_loss.append(None)

        raw_debt = []
        for c in range(n):
            L = per_class_loss[c]
            if L is not None:
                smoothed_loss[c] = smooth * smoothed_loss[c] + (1.0 - smooth) * L
                integral[c] = decay * integral[c] + (1.0 - decay) * smoothed_loss[c]

            p = K_p * smoothed_loss[c]
            i_term = K_i * integral[c]
            d = K_d * (smoothed_loss[c] - prev_loss[c])

            debt_val = max(0.0, p + i_term + d)
            raw_debt.append(debt_val)

            if L is not None:
                prev_loss[c] = smoothed_loss[c]

        all_debts.append(raw_debt)
    return all_debts


def generate_batches(num_batches=700, shift_batch=350):
    """Generate batch stream with class presence info."""
    import random
    rng = random.Random(13)
    class_counts = [2000, 200, 20]
    total = sum(class_counts)

    epoch_items = []
    for c, count in enumerate(class_counts):
        epoch_items.extend([c] * count)

    batches = []
    for idx in range(num_batches):
        if idx % (total // 32 + 1) == 0:
            rng.shuffle(epoch_items)
        start = (idx * 32) % total
        end = start + 32
        if end <= total:
            batch = epoch_items[start:end]
        else:
            batch = epoch_items[start:] + epoch_items[:end - total]

        shifted = idx >= shift_batch
        if shifted:
            batch = [2 if c == 0 else (0 if c == 2 else c) for c in batch]

        batches.append({"idx": idx, "shifted": shifted, "classes": set(batch)})
    return batches


def compute_metrics(debts, shift_batch=350, num_batches=700):
    """Compute diagnostic metrics from debt trajectory."""
    # Response time: steps after shift for class 2 debt to exceed class 0 debt
    c0_debts = [d[0] for d in debts]
    c2_debts = [d[2] for d in debts]

    response_time = None
    for i in range(shift_batch, min(len(debts), shift_batch + 200)):
        if c2_debts[i] > c0_debts[i]:
            response_time = i - shift_batch
            break

    # Class 2 allocation share in steady state (last 100 steps)
    steady_start = max(shift_batch + 200, len(debts) - 100)
    steady_debts = debts[steady_start:]
    if steady_debts:
        c2_share = sum(d[2] for d in steady_debts) / (
            sum(sum(d) for d in steady_debts) + 1e-10
        )
    else:
        c2_share = 0.0

    # Integral persistence: how many steps for class 0 integral to halve after shift
    # (approximated from half-life formula)

    # Class 2 peak debt
    c2_peak = max(c2_debts) if c2_debts else 0.0
    c0_peak = max(c0_debts) if c0_debts else 0.0

    return {
        "response_time": response_time,
        "c2_share_steady": c2_share,
        "c2_peak": c2_peak,
        "c0_peak": c0_peak,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  GAIN SENSITIVITY DIAGNOSTIC (H10)")
    print("=" * 70)

    batches = generate_batches()

    # Default gains
    print("\n" + "-" * 70)
    print("  DEFAULT GAINS: K_p=1.0, K_i=0.1, K_d=0.5, decay=0.99, smooth=0.9")
    print("-" * 70)
    debts = simulate_pid(batches)
    metrics = compute_metrics(debts)
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # --- K_p sweep ---
    print("\n" + "-" * 70)
    print("  K_p SWEEP (K_i=0.1, K_d=0.5)")
    print("-" * 70)
    print(f"  {'K_p':>8} {'Resp Time (steps)':>20} {'C2 Share (steady)':>20} {'C2 Peak':>10} {'C0 Peak':>10}")
    print("  " + "-" * 68)
    for K_p in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        d = simulate_pid(batches, K_p=K_p)
        m = compute_metrics(d)
        print(f"  {K_p:>8.1f} {m['response_time'] if m['response_time'] else 'N/A':>20} "
              f"{m['c2_share_steady']:>19.4f} {m['c2_peak']:>10.4f} {m['c0_peak']:>10.4f}")

    # --- K_i sweep ---
    print("\n" + "-" * 70)
    print("  K_i SWEEP (K_p=1.0, K_d=0.5)")
    print("-" * 70)
    print(f"  {'K_i':>8} {'Resp Time (steps)':>20} {'C2 Share (steady)':>20} {'C2 Peak':>10} {'C0 Peak':>10}")
    print("  " + "-" * 68)
    for K_i in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        d = simulate_pid(batches, K_i=K_i)
        m = compute_metrics(d)
        print(f"  {K_i:>8.2f} {m['response_time'] if m['response_time'] else 'N/A':>20} "
              f"{m['c2_share_steady']:>19.4f} {m['c2_peak']:>10.4f} {m['c0_peak']:>10.4f}")

    # --- K_d sweep ---
    print("\n" + "-" * 70)
    print("  K_d SWEEP (K_p=1.0, K_i=0.1)")
    print("-" * 70)
    print(f"  {'K_d':>8} {'Resp Time (steps)':>20} {'C2 Share (steady)':>20} {'C2 Peak':>10} {'C0 Peak':>10}")
    print("  " + "-" * 68)
    for K_d in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        d = simulate_pid(batches, K_d=K_d)
        m = compute_metrics(d)
        print(f"  {K_d:>8.1f} {m['response_time'] if m['response_time'] else 'N/A':>20} "
              f"{m['c2_share_steady']:>19.4f} {m['c2_peak']:>10.4f} {m['c0_peak']:>10.4f}")

    # --- Decay sweep ---
    print("\n" + "-" * 70)
    print("  DECAY SWEEP (K_p=1.0, K_i=0.1, K_d=0.5)")
    print("-" * 70)
    print(f"  {'Decay':>8} {'Resp Time (steps)':>20} {'C2 Share (steady)':>20} {'C2 Peak':>10} {'C0 Peak':>10}")
    print("  " + "-" * 68)
    for decay in [0.9, 0.95, 0.99, 0.995, 0.999]:
        d = simulate_pid(batches, decay=decay)
        m = compute_metrics(d)
        print(f"  {decay:>8.3f} {m['response_time'] if m['response_time'] else 'N/A':>20} "
              f"{m['c2_share_steady']:>19.4f} {m['c2_peak']:>10.4f} {m['c0_peak']:>10.4f}")

    # --- Smooth sweep ---
    print("\n" + "-" * 70)
    print("  SMOOTH SWEEP (K_p=1.0, K_i=0.1, K_d=0.5, decay=0.99)")
    print("-" * 70)
    print(f"  {'Smooth':>8} {'Resp Time (steps)':>20} {'C2 Share (steady)':>20} {'C2 Peak':>10} {'C0 Peak':>10}")
    print("  " + "-" * 68)
    for smooth in [0.5, 0.7, 0.9, 0.95, 0.99]:
        d = simulate_pid(batches, smooth=smooth)
        m = compute_metrics(d)
        print(f"  {smooth:>8.2f} {m['response_time'] if m['response_time'] else 'N/A':>20} "
              f"{m['c2_share_steady']:>19.4f} {m['c2_peak']:>10.4f} {m['c0_peak']:>10.4f}")

    # --- FULL GRID top results ---
    print("\n" + "-" * 70)
    print("  FULL GRID: Top 10 configurations by C2 share")
    print("-" * 70)

    K_p_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    K_i_vals = [0.0, 0.05, 0.1, 0.2, 0.5]
    K_d_vals = [0.0, 0.1, 0.5, 1.0, 2.0]

    all_configs = []
    for K_p, K_i, K_d in itertools.product(K_p_vals, K_i_vals, K_d_vals):
        d = simulate_pid(batches, K_p=K_p, K_i=K_i, K_d=K_d)
        m = compute_metrics(d)
        all_configs.append((K_p, K_i, K_d, m))

    # Sort by C2 share (higher is better for minority)
    all_configs.sort(key=lambda x: x[3]["c2_share_steady"], reverse=True)

    print(f"  {'K_p':>6} {'K_i':>6} {'K_d':>6} {'Resp':>6} {'C2 Share':>10} {'C2 Peak':>10} {'C0 Peak':>10}")
    print("  " + "-" * 54)
    for K_p, K_i, K_d, m in all_configs[:10]:
        rt = m["response_time"] if m["response_time"] else -1
        print(f"  {K_p:>6.1f} {K_i:>6.2f} {K_d:>6.1f} {rt:>6} {m['c2_share_steady']:>9.2%} "
              f"{m['c2_peak']:>10.4f} {m['c0_peak']:>10.4f}")
    print()
    print(f"  Worst 5:")
    print(f"  {'K_p':>6} {'K_i':>6} {'K_d':>6} {'Resp':>6} {'C2 Share':>10}")
    for K_p, K_i, K_d, m in all_configs[-5:]:
        rt = m["response_time"] if m["response_time"] else -1
        print(f"  {K_p:>6.1f} {K_i:>6.2f} {K_d:>6.1f} {rt:>6} {m['c2_share_steady']:>9.2%}")

    # --- VERDICT ---
    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  H10 (Gain sensitivity):
    If C2 share varies significantly across gain configurations, then
    tuning matters.  But the key question is whether ANY configuration
    gives PID-CR a CLEAR advantage over static_bank (which gives class 2
    ~33% of budget).

    If no gain configuration gives class 2 >50% share, then the
    bottleneck is NOT the gains — it's the fundamental architecture
    (loss signal, allocation mechanism, or bank quality).
""")
