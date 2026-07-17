"""H6/H8/H11: Bank contamination & fill dynamics after label shift.

Hypotheses:
  H6: The bank itself is the bottleneck — retrieved items have wrong labels.
  H8: The bank fills BEFORE the shift (epoch 5), so no new data enters.
  H11: PID-CR allocates MORE to contaminated classes, hurting performance.

This script SIMULATES the bank dynamics without training a model.
Key improvement over v1: correctly tracks ORIGINAL class identity vs
STORED label to detect contamination.
"""

import math
import random

import numpy as np


def simulate_data_stream(seed=13, majority_train=2000, imbalance_ratio=100,
                         batch_size=32, num_epochs=10, shift_epoch=5):
    """Generate batch labels (original_id, stored_label) for each epoch."""
    rng = random.Random(seed)
    medium_count = max(1, int(majority_train / math.sqrt(imbalance_ratio)))
    rare_count = max(1, majority_train // imbalance_ratio)
    class_counts = [majority_train, medium_count, rare_count]

    epoch_items = []
    for c, count in enumerate(class_counts):
        epoch_items.extend([c] * count)

    for epoch in range(num_epochs):
        shifted = epoch >= shift_epoch
        items = list(epoch_items)
        rng.shuffle(items)

        batches = []
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            if shifted:
                # After shift: label toggles (0<->2)
                batch = [(c, 2 if c == 0 else (0 if c == 2 else c)) for c in batch_items]
            else:
                batch = [(c, c) for c in batch_items]
            batches.append(batch)
        yield epoch, shifted, batches


class Bank:
    def __init__(self, capacity_per_class=200, num_classes=3):
        self.capacity = capacity_per_class
        # pool[label] = list of (original_identity, label_at_storage)
        self.pools = {c: [] for c in range(num_classes)}

    def store(self, original_id, stored_label):
        """Store an item in the bank under its stored_label."""
        pool = self.pools[stored_label]
        if len(pool) < self.capacity:
            pool.append((original_id, stored_label))

    def get_contamination(self, class_id, epoch_shifted):
        """Fraction of items in pool that have WRONG labels.

        An item has a wrong label if:
          original_identity != stored_label AND epoch >= shift_epoch
        """
        pool = self.pools[class_id]
        if not pool:
            return 0.0
        bad = sum(1 for orig, stored in pool if epoch_shifted and orig != stored)
        return bad / len(pool)

    def pool_size(self, class_id):
        return len(self.pools[class_id])


def pid_debt_simulation(class_losses, K_p=1.0, K_i=0.1, K_d=0.5,
                        decay=0.99, smooth=0.9):
    """Simulate PID controller and return debts at each step."""
    n = 3
    integral = [0.0] * n
    prev_loss = [0.0] * n
    smoothed_loss = [0.0] * n
    debts = []

    for losses in class_losses:
        raw_debt = []
        for c in range(n):
            L = losses[c]
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
        debts.append(raw_debt)
    return debts


def compute_pid_allocation(debts, budget=8):
    """Proportional allocation from debt."""
    if sum(debts) <= 0:
        n = len(debts)
        base = [budget // n] * n
        for i in range(budget % n):
            base[i] += 1
        return base
    raw = [budget * d / sum(debts) for d in debts]
    base = [int(v) for v in raw]
    remaining = budget - sum(base)
    order = sorted(range(len(debts)), key=lambda i: (raw[i] - base[i], debts[i]), reverse=True)
    for i in order[:remaining]:
        base[i] += 1
    return base


def simulate_bank_dynamics():
    """Full simulation of bank dynamics under shift."""
    rng = random.Random(13)
    bank = Bank(capacity_per_class=200, num_classes=3)

    batch_size = 32
    post_shift_loss = [2.3, 0.1, 2.3]   # C0 high, C1 low, C2 high
    pre_shift_loss = [0.1, 0.1, 0.1]

    class_loss_history = []
    bank_history = []
    epoch_alloc_pid = {}
    epoch_alloc_static = {}

    for epoch, shifted, batches in simulate_data_stream(
        seed=13, shift_epoch=5, num_epochs=10
    ):
        epoch_losses = []
        for batch in batches:
            # Store items with their TRUE original identity
            for orig_id, stored_label in batch:
                bank.store(orig_id, stored_label)

            # Record per-class loss for PID
            classes_in_batch = {stored for _, stored in batch}
            if shifted:
                loss = [post_shift_loss[c] if c in classes_in_batch else None for c in range(3)]
            else:
                loss = [pre_shift_loss[c] if c in classes_in_batch else None for c in range(3)]
            epoch_losses.append(loss)

        # Record bank state at end of epoch
        pools = {c: bank.pool_size(c) for c in range(3)}
        contamination = {c: bank.get_contamination(c, epoch >= 5) for c in range(3)}
        bank_history.append({
            "epoch": epoch, "pools": pools, "contamination": contamination, "shifted": shifted,
        })

        class_loss_history.extend(epoch_losses)

    # Run PID on loss history
    debts = pid_debt_simulation(class_loss_history)

    # Compute per-epoch allocations
    batch_idx = 0
    for epoch, shifted, batches in simulate_data_stream(seed=13, shift_epoch=5, num_epochs=10):
        epoch_pid = []
        epoch_static = []
        for _ in batches:
            if batch_idx < len(debts):
                ap = compute_pid_allocation(debts[batch_idx])
                epoch_pid.append(ap)
                as_ = [3, 3, 2]  # static: 8/3 ≈ 3,3,2
                epoch_static.append(as_)
            batch_idx += 1
        if epoch_pid:
            epoch_alloc_pid[epoch] = [sum(v[i] for v in epoch_pid) / len(epoch_pid) for i in range(3)]
            epoch_alloc_static[epoch] = [sum(v[i] for v in epoch_static) / len(epoch_static) for i in range(3)]

    # Compute retrieval contamination per epoch
    ret_con_pid = {}
    ret_con_static = {}

    for epoch in range(10):
        temp_bank = Bank(capacity_per_class=200, num_classes=3)
        for epoch2, shifted2, batches2 in simulate_data_stream(seed=13, shift_epoch=5, num_epochs=epoch + 1):
            if epoch2 > epoch:
                break
            for batch in batches2:
                for orig_id, stored_label in batch:
                    temp_bank.store(orig_id, stored_label)

        epoch_shifted = epoch >= 5

        if epoch in epoch_alloc_pid:
            alloc = epoch_alloc_pid[epoch]
            retrieved = [0, 0, 0]
            contaminated = [0, 0, 0]
            for c in range(3):
                count = int(round(alloc[c]))
                pool = temp_bank.pools[c]
                if pool:
                    sampled = rng.choices(pool, k=min(count, len(pool)))
                    retrieved[c] = len(sampled)
                    for orig, stored in sampled:
                        if epoch_shifted and orig != stored:
                            contaminated[c] += 1
            tr = sum(retrieved)
            tc = sum(contaminated)
            ret_con_pid[epoch] = {"retrieved": tr, "contaminated": tc,
                                  "ratio": tc / tr if tr > 0 else 0.0,
                                 "per_class": {c: {"retrieved": retrieved[c],
                                                   "contaminated": contaminated[c],
                                                   "ratio": contaminated[c] / retrieved[c] if retrieved[c] > 0 else 0.0}
                                              for c in range(3)}}

        if epoch in epoch_alloc_static:
            alloc = epoch_alloc_static[epoch]
            retrieved = [0, 0, 0]
            contaminated = [0, 0, 0]
            for c in range(3):
                pool = temp_bank.pools[c]
                if pool:
                    sampled = rng.choices(pool, k=min(int(round(alloc[c])), len(pool)))
                    retrieved[c] = len(sampled)
                    for orig, stored in sampled:
                        if epoch_shifted and orig != stored:
                            contaminated[c] += 1
            tr = sum(retrieved)
            tc = sum(contaminated)
            ret_con_static[epoch] = {"retrieved": tr, "contaminated": tc,
                                     "ratio": tc / tr if tr > 0 else 0.0,
                                     "per_class": {c: {"retrieved": retrieved[c],
                                                       "contaminated": contaminated[c],
                                                       "ratio": contaminated[c] / retrieved[c] if retrieved[c] > 0 else 0.0}
                                                  for c in range(3)}}

    return {
        "bank_history": bank_history,
        "alloc_pid": epoch_alloc_pid,
        "alloc_static": epoch_alloc_static,
        "ret_con_pid": ret_con_pid,
        "ret_con_static": ret_con_static,
        "debts": debts,
    }


if __name__ == "__main__":
    result = simulate_bank_dynamics()

    print("=" * 70)
    print("  H8: Bank Fill Timing (capacity=200 per class)")
    print("=" * 70)
    prior = {0: 0, 1: 0, 2: 0}
    for h in result["bank_history"]:
        e, pools = h["epoch"], h["pools"]
        for c in [0, 1, 2]:
            if prior[c] < 200 and pools[c] >= 200:
                print(f"    Class {c} filled at epoch {e}")
        prior = pools.copy()

    print(f"\n  Pool sizes at shift point (epoch 5):")
    for h in result["bank_history"]:
        if h["epoch"] == 5:
            for c in [0, 1, 2]:
                sz = h["pools"][c]
                print(f"    Class {c}: {sz}")
            break

    print("\n" + "=" * 70)
    print("  H11: Pool Contamination After Shift (% items with wrong label)")
    print("=" * 70)
    print(f"  {'Epoch':>6} {'Shift':>6} {'C0 size':>8} {'C0 contam':>10} "
          f"{'C1 size':>8} {'C1 contam':>10} {'C2 size':>8} {'C2 contam':>10}")
    print("  " + "-" * 66)
    for h in result["bank_history"]:
        e, s, p, con = h["epoch"], "Y" if h["shifted"] else "N", h["pools"], h["contamination"]
        print(f"  {e:>6} {s:>6} {p[0]:>8} {con[0]:>9.1%} "
              f"{p[1]:>8} {con[1]:>9.1%} {p[2]:>8} {con[2]:>9.1%}")

    # Detailed breakdown of contamination
    print("\n  DETAIL: Class 2 pool contents by epoch:")
    for h in result["bank_history"]:
        if h["shifted"]:
            print(f"    Epoch {h['epoch']}: size={h['pools'][2]}, "
                  f"contamination={h['contamination'][2]:.1%}")

    print("\n" + "=" * 70)
    print("  H6: Retrieval Contamination")
    print("=" * 70)
    print(f"  {'Epoch':>6} {'PID retr':>10} {'PID bad':>10} {'PID bad%':>10} "
          f"{'Static retr':>12} {'Static bad':>12} {'Static bad%':>12}")
    print("  " + "-" * 72)
    for e in range(10):
        pid = result["ret_con_pid"].get(e, {})
        sta = result["ret_con_static"].get(e, {})
        print(f"  {e:>6} {pid.get('retrieved', 0):>10} {pid.get('contaminated', 0):>10} "
              f"{pid.get('ratio', 0):>9.1%} "
              f"{sta.get('retrieved', 0):>12} {sta.get('contaminated', 0):>12} "
              f"{sta.get('ratio', 0):>11.1%}")

    # Per-class breakdown at epoch 9
    print("\n  Per-class retrieval contamination at epoch 9:")
    for method, data in [("PID-CR", result["ret_con_pid"]), ("Static", result["ret_con_static"])]:
        ep9 = data.get(9, {})
        print(f"\n    {method}:")
        for c in range(3):
            pc = ep9.get("per_class", {}).get(c, {})
            if pc.get("retrieved", 0) > 0:
                print(f"      Class {c}: retrieved={pc['retrieved']}, "
                      f"bad={pc['contaminated']}, bad%={pc['ratio']:.1%}")

    print("\n" + "=" * 70)
    print("  ALLOCATION: PID-CR vs Static (budget=8, avg/epoch)")
    print("=" * 70)
    print(f"  {'Epoch':>6} {'PID C0':>8} {'PID C1':>8} {'PID C2':>8}  "
          f"{'Static C0':>10} {'Static C1':>10} {'Static C2':>10}")
    for e in range(10):
        pa = result["alloc_pid"].get(e, [0, 0, 0])
        sa = result["alloc_static"].get(e, [3, 3, 2])
        print(f"  {e:>6} {pa[0]:>8.2f} {pa[1]:>8.2f} {pa[2]:>8.2f}  "
              f"{sa[0]:>10} {sa[1]:>10} {sa[2]:>10}")

    # VERDICT
    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    max_con_pid = max((result["ret_con_pid"].get(e, {}).get("ratio", 0) for e in range(10)), default=0)
    max_con_sta = max((result["ret_con_static"].get(e, {}).get("ratio", 0) for e in range(10)), default=0)

    verdict_h6 = "CONFIRMED" if max_con_pid > 0 else "REFUTED"
    verdict_h8 = "CONFIRMED" if any(h["pools"][c] >= 200 for h in result["bank_history"] if h["epoch"] < 5 for c in [0, 1]) else "REFUTED"
    verdict_h11 = "CONFIRMED" if max_con_pid > max_con_sta else ("REFUTED" if max_con_pid == max_con_sta == 0 else "PARTIALLY")

    print(f"\n  H6 (Bank bottleneck): {verdict_h6}")
    print(f"    Max PID-CR retrieval contamination: {max_con_pid:.1%}")
    print(f"    Max Static retrieval contamination: {max_con_sta:.1%}")
    print(f"\n  H8 (Bank fills before shift): {verdict_h8}")
    print(f"\n  H11 (PID allocates more to contaminated class): {verdict_h11}")
