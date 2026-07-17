"""Diagnostic: extract debt and allocation trajectories from logged CSV data.

Reads the per-seed metrics.csv for pid_gb under shift and analyzes
how debt/allocation behave before and after the label swap at epoch 5.

Usage: python diagnostic_debt_trajectory.py
"""

import csv
import sys
from pathlib import Path

# Locate the pid_gb shift experiment output
shift_dir = Path("output/shift_experiment")
runs = sorted(shift_dir.iterdir())
if not runs:
    print("ERROR: no shift experiment output found")
    sys.exit(1)

# The latest run is pid_gb (from timestamps: ed_gb=151145, static=151133, pid_gb=151209)
pid_gb_run = shift_dir / "20260713_151209"
seed_csv = pid_gb_run / "seed_13" / "metrics.csv"

if not seed_csv.exists():
    print(f"ERROR: {seed_csv} not found")
    sys.exit(1)

# Read CSV and extract debt/alloc columns
rows = []
with open(seed_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Read {len(rows)} rows from {seed_csv}")
print(f"Columns: {list(rows[0].keys())}")

# Check if debt columns have data
def safe_float(v, default=0.0):
    try:
        return float(v) if v != "" else default
    except (ValueError, TypeError):
        return default

non_zero_debt_0 = sum(1 for r in rows if safe_float(r.get("debt/class_0", 0)) > 0)
non_zero_debt_1 = sum(1 for r in rows if safe_float(r.get("debt/class_1", 0)) > 0)
non_zero_debt_2 = sum(1 for r in rows if safe_float(r.get("debt/class_2", 0)) > 0)
print(f"\nRows with non-zero debt: class_0={non_zero_debt_0}, class_1={non_zero_debt_1}, class_2={non_zero_debt_2}")

# Find the shift epoch boundary
shift_epoch = 5
shift_step = None
for r in rows:
    if int(r["epoch"]) == shift_epoch and int(r["step"]) == 0:
        shift_step = int(r["step"])
        break

print(f"\nShift at epoch {shift_epoch}")

# ---- Summary stats before vs after shift ----
def summarize(rows_subset, label):
    if not rows_subset:
        print(f"\n  {label}: no rows")
        return
    debts = [[], [], []]
    allocs = [[], [], []]
    for r in rows_subset:
        for c in range(3):
            d = safe_float(r.get(f"debt/class_{c}", 0))
            a = safe_float(r.get(f"alloc/class_{c}", 0))
            debts[c].append(d)
            allocs[c].append(a)

    print(f"\n  {label} ({len(rows_subset)} steps):")
    for c in range(3):
        avg_d = sum(debts[c]) / len(debts[c]) if debts[c] else 0
        avg_a = sum(allocs[c]) / len(allocs[c]) if allocs[c] else 0
        max_d = max(debts[c]) if debts[c] else 0
        print(f"    class {c}: debt avg={avg_d:.4f} max={max_d:.4f}  alloc avg={avg_a:.2f}")

# Split by epoch boundary
before = [r for r in rows if int(r["epoch"]) < shift_epoch]
after = [r for r in rows if int(r["epoch"]) >= shift_epoch]

print("=" * 60)
summarize(before, "BEFORE SHIFT (epochs < 5)")
summarize(after, "AFTER SHIFT (epochs >= 5)")

# ---- Check the first few steps after shift ----
print("\n" + "=" * 60)
print("First 10 steps after shift boundary:")
count = 0
for r in rows:
    if int(r["epoch"]) >= shift_epoch and count < 10:
        step = r["step"]
        ep = r["epoch"]
        d0, d1, d2 = safe_float(r["debt/class_0"]), safe_float(r["debt/class_1"]), safe_float(r["debt/class_2"])
        a0, a1, a2 = safe_float(r["alloc/class_0"]), safe_float(r["alloc/class_1"]), safe_float(r["alloc/class_2"])
        print(f"  epoch={ep} step={step:>4}  debt=[{d0:.4f} {d1:.4f} {d2:.4f}]  alloc=[{a0:.0f} {a1:.0f} {a2:.0f}]")
        count += 1

# ---- Check if debt actually changes for class 0 after shift ----
print("\n" + "=" * 60)
# Class 0: avg debt in last 50 steps before shift vs first 50 steps after shift
last_before = [r for r in before][-50:]
first_after = [r for r in after][:50]

if last_before and first_after:
    d0_before = [safe_float(r["debt/class_0"]) for r in last_before]
    d0_after = [safe_float(r["debt/class_0"]) for r in first_after]
    avg_before = sum(d0_before) / len(d0_before)
    avg_after = sum(d0_after) / len(d0_after)
    print(f"Class 0 debt: last 50 pre-shift avg={avg_before:.4f}  first 50 post-shift avg={avg_after:.4f}")

    d2_before = [safe_float(r["debt/class_2"]) for r in last_before]
    d2_after = [safe_float(r["debt/class_2"]) for r in first_after]
    avg_before_2 = sum(d2_before) / len(d2_before)
    avg_after_2 = sum(d2_after) / len(d2_after)
    print(f"Class 2 debt: last 50 pre-shift avg={avg_before_2:.4f}  first 50 post-shift avg={avg_after_2:.4f}")
