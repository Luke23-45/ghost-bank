"""Ablation harness: dedicated per-family runners for the deep feature bank study.

Baseline/control methods (iCaRL, static bank, no-replay, linear legacy)
are run directly via ``run_all.py``; this package covers only the
ablations layered on top of the locked reference.

Layout
------
shared/       cross-family orchestration (rows, protocol, executor, report)
component.py  family=component: a1..a4 (KD, head eval, linear head, random bank)
sensitivity.py family=sensitivity: s1..s4 (memory budget, retrieval budget)
dry_run_all.py compose + validate every row (no GPU)
"""