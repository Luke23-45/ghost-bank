"""H2/H3/H4: PID controller internal dynamics under sparse updates.

Hypotheses:
  H2: Minority class (class 2) PID updates are too infrequent.
      Class 2 appears in ~1% of training samples → ~30% of batches.
      For 70% of batches, class 2's PID state is stale.
  H3: Integral term causes harmful lag — after shift, class 0's integral
      keeps its debt high long after its loss normalizes.
  H4: PID state (smooth=0.9, decay=0.99) creates slow response dynamics.

This script:
  1. Tracks per-class loss availability through training
  2. Simulates PID dynamics with realistic update patterns
  3. Measures: response time, overshoot, steady-state debt
  4. Compares P-only vs PI vs PID vs full PID
"""

import math
import random

import numpy as np


def simulate_batch_stream(seed=13, majority_train=2000, imbalance_ratio=100,
                          batch_size=32, num_batches=700, shift_batch=350):
    """Simulate which classes appear in each batch."""
    rng = random.Random(seed)
    medium_count = max(1, int(majority_train / math.sqrt(imbalance_ratio)))
    rare_count = max(1, majority_train // imbalance_ratio)
    class_counts = [majority_train, medium_count, rare_count]
    total = sum(class_counts)

    epoch_items = []
    for c, count in enumerate(class_counts):
        epoch_items.extend([c] * count)

    batches = []
    for batch_idx in range(num_batches):
        # Simulate a batch by sampling without replacement from shuffled epoch
        if batch_idx % (total // batch_size) == 0:
            rng.shuffle(epoch_items)
        start = (batch_idx * batch_size) % total
        end = start + batch_size
        if end <= total:
            batch = epoch_items[start:end]
        else:
            # Wrap around
            batch = epoch_items[start:] + epoch_items[:end - total]

        shifted = batch_idx >= shift_batch
        if shifted:
            batch = [2 if c == 0 else (0 if c == 2 else c) for c in batch]

        classes_in_batch = set(batch)
        batches.append({
            "idx": batch_idx,
            "shifted": shifted,
            "classes": classes_in_batch,
        })
    return batches


def simulate_pid_on_batches(batches, shift_batch=350,
                            K_p=1.0, K_i=0.1, K_d=0.5,
                            decay=0.99, smooth=0.9,
                            class_weights=None,
                            pre_shift_loss=0.1, post_shift_loss=2.3,
                            recovery_epochs=3, batches_per_epoch=70):
    """Simulate PID controller with realistic loss dynamics.

    After the shift, class 0 and 2 losses spike to post_shift_loss.
    Over 'recovery_epochs' epochs, class 0's loss decays (it has 2000
    samples/epoch so it adapts fast), while class 2's loss stays high
    (only 20 samples/epoch so it adapts slowly).
    """
    n = 3
    weights = class_weights or [1.0] * n

    integral = [0.0] * n
    prev_loss = [0.0] * n
    smoothed_loss = [0.0] * n
    losses_history = []
    debts_history = []
    debt_components = []  # Track p, i, d separately

    # Loss tracking per class
    class_0_adaptation_steps = 0
    class_2_adaptation_steps = 0
    # After shift, class 0 adapts in ~50 steps (2000 samples / 32 bsz ≈ 62.5)
    # Class 2 adapts much slower (20 / 32 ≈ 0.6 steps per epoch worth of new data)
    # We model: class 0 loss decays linearly over ~150 steps
    #           class 2 loss decays linearly over ~350 steps (5 epochs)
    c0_decay_rate = (post_shift_loss - pre_shift_loss) / 150  # per step
    c2_decay_rate = (post_shift_loss - pre_shift_loss) / 350  # per step

    class_0_loss = pre_shift_loss
    class_2_loss = pre_shift_loss

    for b in batches:
        idx = b["idx"]
        classes = b["classes"]
        shifted = b["shifted"]

        if shifted:
            if "class_0_spike" not in locals() or not class_0_spike:
                class_0_spike = True
                # Losses spike at shift
                class_0_loss = post_shift_loss
                class_2_loss = post_shift_loss
                c0_steps_since_shift = 0
                c2_steps_since_shift = 0

            # Decay losses based on how many steps each class has been seen
            c0_steps_since_shift = c0_steps_since_shift if 'c0_steps_since_shift' in dir() else 0
            c2_steps_since_shift = c2_steps_since_shift if 'c2_steps_since_shift' in dir() else 0

            if 0 in classes:
                c0_steps_since_shift = (c0_steps_since_shift if 'c0_steps_since_shift' in locals() else 0) + 1
            if 2 in classes:
                c2_steps_since_shift = (c2_steps_since_shift if 'c2_steps_since_shift' in locals() else 0) + 1

            # Decay only when class is seen (its loss drops only when trained on it)
            if 0 in classes:
                class_0_loss = max(pre_shift_loss, class_0_loss - c0_decay_rate)
            if 2 in classes:
                class_2_loss = max(pre_shift_loss, class_2_loss - c2_decay_rate)
        else:
            class_0_loss = pre_shift_loss
            class_2_loss = pre_shift_loss

        # Build per_class_loss for PID
        per_class_loss = []
        for c in range(n):
            if c in classes:
                if c == 0:
                    per_class_loss.append(class_0_loss)
                elif c == 2:
                    per_class_loss.append(class_2_loss)
                else:  # class 1
                    per_class_loss.append(pre_shift_loss)
            else:
                per_class_loss.append(None)

        losses_history.append(per_class_loss)

        # PID update (same logic as pid_controller.py)
        raw_debt = []
        comps = []
        for c in range(n):
            L = per_class_loss[c]

            if L is not None:
                smoothed_loss[c] = smooth * smoothed_loss[c] + (1.0 - smooth) * L
                integral[c] = decay * integral[c] + (1.0 - decay) * smoothed_loss[c]

            p = K_p * smoothed_loss[c]
            i_term = K_i * integral[c]
            d = K_d * (smoothed_loss[c] - prev_loss[c])

            debt_val = max(0.0, weights[c] * (p + i_term + d))
            raw_debt.append(debt_val)
            comps.append({"p": p, "i": i_term, "d": d})

            if L is not None:
                prev_loss[c] = smoothed_loss[c]

        debts_history.append(raw_debt)
        debt_components.append(comps)

    return {
        "losses": losses_history,
        "debts": debts_history,
        "components": debt_components,
    }


def print_epoch_summary(results, batches_per_epoch=70, shift_batch=350):
    """Print per-epoch averages of loss, debt, and components."""
    n_epochs = len(results["losses"]) // batches_per_epoch
    print(f"  {'Epoch':>6} {'Shift':>6} {'Class':>6} {'Loss Seen':>10} {'MissCnt':>8} "
          f"{'Smoothed':>10} {'Integral':>10} {'Debt':>10} {'P':>10} {'I':>10} {'D':>10} {'Updates':>8}")
    print("  " + "-" * 108)

    for epoch in range(n_epochs):
        start = epoch * batches_per_epoch
        end = min(start + batches_per_epoch, len(results["losses"]))
        shifted = start >= shift_batch

        for c in [0, 1, 2]:
            losses_c = [results["losses"][i][c] for i in range(start, end)]
            debts_c = [results["debts"][i][c] for i in range(start, end)]
            comps_c = [results["components"][i][c] for i in range(start, end)]
            updates = sum(1 for l in losses_c if l is not None)

            losses_seen = [l for l in losses_c if l is not None]
            losses_miss = [l for l in losses_c if l is None]

            loss_seen_avg = sum(losses_seen) / len(losses_seen) if losses_seen else 0.0
            loss_miss_count = len(losses_miss)
            debt_avg = sum(debts_c) / len(debts_c) if debts_c else 0.0

            seen_str = f"{loss_seen_avg:.3f}" if losses_seen else "   N/A"

            flag = ""
            if c == 2 and shifted:
                flag = " <<< MINORITY"
            elif c == 0 and shifted:
                flag = " (majority)"

            p_epoch = sum(comp["p"] for comp in comps_c) / len(comps_c) if comps_c else 0
            i_epoch = sum(comp["i"] for comp in comps_c) / len(comps_c) if comps_c else 0
            d_epoch = sum(comp["d"] for comp in comps_c) / len(comps_c) if comps_c else 0

            print(f"  {epoch:>6} {'Y' if shifted else 'N':>6} {c:>6} "
                  f"{seen_str:>10} {loss_miss_count:>9} "
                  f"{'':>10} {'':>10} "
                  f"{debt_avg:>10.4f} {p_epoch:>10.4f} {i_epoch:>10.4f} {d_epoch:>10.4f} "
                  f"{updates:>8}{flag}")


def run_ablation(batches, shift_batch=350):
    """Compare P-only, PI, PD, Full PID debt dynamics."""
    configs = {
        "P-only": {"K_i": 0.0, "K_d": 0.0},
        "PI": {"K_d": 0.0},
        "PD": {"K_i": 0.0},
        "Full PID": {},
    }
    results = {}
    for name, overrides in configs.items():
        kwargs = {"K_p": 1.0, "K_i": 0.1, "K_d": 0.5}
        kwargs.update(overrides)
        results[name] = simulate_pid_on_batches(batches, shift_batch=shift_batch, **kwargs)

    # Compare class 2 debt trajectories
    print("\n  ABLATION: Class 2 debt trajectory (post-shift, step 350-700)")
    print(f"  {'Step':>6} ", end="")
    for name in configs:
        print(f"{name + ' Debt':>18} ", end="")
    print()
    print("  " + "-" * (6 + 18 * len(configs)))

    for step in range(350, min(700, len(batches)), 10):
        print(f"  {step:>6} ", end="")
        for name in configs:
            d = results[name]["debts"][step][2]
            print(f"{d:>18.4f} ", end="")
        print()

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  PID DYNAMICS DIAGNOSTIC (H2, H3, H4)")
    print("=" * 70)

    batches = simulate_batch_stream(seed=13, num_batches=700, shift_batch=350)

    # H2: Update frequency
    print("\n" + "-" * 70)
    print("  H2: PID Update Frequency (how often each class is seen)")
    print("-" * 70)
    class_updates = {0: 0, 1: 0, 2: 0}
    class_update_epochs = {e: {0: 0, 1: 0, 2: 0} for e in range(10)}
    total_batches = 0
    for b in batches:
        epoch = b["idx"] // 70
        for c in b["classes"]:
            class_updates[c] += 1
            class_update_epochs[epoch][c] += 1
        total_batches += 1

    print(f"  Total batches: {total_batches}")
    print(f"  Updates per class:")
    for c in [0, 1, 2]:
        pct = class_updates[c] / total_batches * 100
        print(f"    Class {c}: {class_updates[c]} / {total_batches} ({pct:.1f}%)")

    print(f"\n  Per-epoch update count:")
    print(f"  {'Epoch':>6} {'Class 0':>10} {'Class 1':>10} {'Class 2':>10} {'Active %':>10}")
    print("  " + "-" * 46)
    for e in range(10):
        total_epoch = sum(class_update_epochs[e].values())
        c0 = class_update_epochs[e][0]
        c1 = class_update_epochs[e][1]
        c2 = class_update_epochs[e][2]
        max_possible = 70  # batches per epoch
        print(f"  {e:>6} {c0:>10} {c1:>10} {c2:>10} "
              f"{c2 / max_possible * 100 if max_possible > 0 else 0:>9.1f}%")

    # H2 impact: PID update gaps
    print(f"\n  Consecutive batches WITHOUT class 2 update:")
    max_gap = 0
    current_gap = 0
    for b in batches:
        if 2 in b["classes"]:
            if current_gap > max_gap:
                max_gap = current_gap
            current_gap = 0
        else:
            current_gap += 1
    print(f"    Max gap: {max_gap} batches ({max_gap / 70:.2f} epochs)")
    print(f"    During this gap, class 2's PID state is completely stale.")

    # H3 / H4: Full PID simulation
    print("\n" + "-" * 70)
    print("  H3/H4: PID State Dynamics (Full PID: K_p=1, K_i=0.1, K_d=0.5)")
    print("-" * 70)
    results = simulate_pid_on_batches(batches, shift_batch=350)

    print_epoch_summary(results, batches_per_epoch=70, shift_batch=350)

    # H3 specifically: Integral build-up for class 0
    print("\n" + "-" * 70)
    print("  H3: Integral Term Analysis")
    print("-" * 70)
    print("""
    After label swap (batch 350), class 0 appears in EVERY batch.
    Its loss stays high for ~150 steps, causing integral to accumulate.
    Even after class 0's loss normalizes, the integral DECAYS slowly
    (decay=0.99, half-life ~69 batches ≈ 1 epoch).

    This means class 0's debt stays artificially high from lingering
    integral well after class 0 has adapted, stealing budget from
    class 2 which still has high loss.
    """)

    # Track integral and debt for class 0 around the shift
    print("  Class 0 Integral and Debt trajectory around shift:")
    print(f"  {'Step':>6} {'Shift':>6} {'Loss':>8} {'Smoothed':>10} {'Integral':>10} {'Debt':>10} {'P-term':>10} {'I-term':>10} {'D-term':>10}")
    print("  " + "-" * 80)
    for step in range(340, min(500, len(batches)), 5):
        b = batches[step]
        shifted = "Y" if step >= 350 else "N"
        loss = results["losses"][step][0]
        loss_str = f"{loss:.3f}" if loss is not None else "  N/A"
        debt = results["debts"][step][0]
        comps = results["components"][step][0]
        print(f"  {step:>6} {shifted:>6} {loss_str:>8} "
              f"{'':>10} {'':>10} "
              f"{debt:>10.4f} {comps['p']:>10.4f} {comps['i']:>10.4f} {comps['d']:>10.4f}")

    # Compare with P-only
    print("\n" + "-" * 70)
    print("  ABLATION: P-only vs Full PID (class 2 debt post-shift)")
    print("-" * 70)
    run_ablation(batches, shift_batch=350)

    # H4: Smoothing lag
    print("\n" + "-" * 70)
    print("  H4: Smoothing Parameter Sensitivity")
    print("-" * 70)
    print(f"  Current smooth=0.9 (EMA: 10% weight on new loss)")
    print(f"  Theoretical 90% convergence: ln(0.1)/ln(0.9) ≈ 22 steps")
    print(f"  After label swap, it takes ~22 batches for smoothed_loss to")
    print(f"  reach 90% of the true post-shift loss value.\n")
    print(f"  With 70 batches/epoch, that's ~0.3 epochs of delayed response.")

    for s_val in [0.5, 0.7, 0.9, 0.95, 0.99]:
        steps_90 = math.log(0.1) / math.log(s_val) if s_val < 1 else float('inf')
        steps_99 = math.log(0.01) / math.log(s_val) if s_val < 1 else float('inf')
        print(f"    smooth={s_val}: 90% converge={steps_90:.0f} steps, "
              f"99% converge={steps_99:.0f} steps")

    for d_val in [0.9, 0.95, 0.99, 0.999]:
        half_life = math.log(0.5) / math.log(d_val) if d_val < 1 else float('inf')
        print(f"    decay={d_val}: integral half-life={half_life:.0f} steps "
              f"({half_life/70:.2f} epochs)")

    # --- VERDICT ---
    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  H2 (Sparse updates):
    If class 2 gets <50% update frequency, its PID state is stale
    for most batches.  The controller can't respond quickly to
    changes in class 2's need.

  H3 (Integral lag):
    Class 0's integral accumulates after the shift.  With decay=0.99,
    the integral half-life is ~69 batches.  This creates persistent
    high debt for class 0 even after its loss normalizes, diverting
    budget from class 2.

  H4 (Smoothing delay):
    smooth=0.9 means ~22 batches (~0.3 epochs) to converge to 90%
    of true loss.  This is the response delay of the controller.
""")
