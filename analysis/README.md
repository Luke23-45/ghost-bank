# Probe-Guided Analysis

This directory contains short, targeted scripts for diagnosing why probe-guided replay is failing.

## What these scripts test

### H1: Probe signal is weak
- Question: do probe scores correlate with eventual forgetting?
- Expected healthy signal: positive rank correlation and separation between high- and low-forgetting classes.

### H2: Allocation degenerates
- Question: does probe-guided allocation stay meaningfully different from uniform replay?
- Expected healthy signal: allocations vary across classes and do not collapse to near-uniform or floor-only quotas.

### H3: Memory budget is violated
- Question: does the exemplar bank stay within the intended fixed budget?
- Expected healthy signal: total stored exemplars stay near `memory_total`, never explode across epochs.

### H4: NME is consistent with linear evaluation
- Question: is the NME head usable, or is it collapsing relative to the linear head?
- Expected healthy signal: NME should be competitive with, or at least not far below, the linear head.

### H5: Probe-guided replay is actually better than uniform replay
- Question: does the method beat the uniform baseline under the same short-run conditions?
- Expected healthy signal: better final accuracy and/or lower forgetting than uniform replay.

## Scripts

- `probe_guided_audit.py`
  - Runs a short GPU-backed CIFAR-100 continual-learning audit.
  - Reports probe statistics, allocation shape, bank growth, and final linear/NME metrics.

- `compare_methods.py`
  - Runs probe-guided, uniform replay, and frozen baseline under the same short settings.
  - Useful for checking whether failure is method-specific or a broader training issue.

- `herding_bic_pilot.py`
  - Ground-up rescue pilot that pairs fixed-memory replay with herding exemplar selection and a BiC-style bias-correction fit.
  - Compares `probe_guided_herding_bic`, `uniform_herding_bic`, and `frozen_baseline`.
  - Best used as the next decision-point experiment before any core-method rewrite.

- `replay_ablation.py`
  - Focused ablation that removes BiC and isolates the two remaining axes:
    - allocation source: `uniform` vs `probe-guided`
    - selection rule: `random` vs `herding`
  - Default `compare` mode runs `uniform_random`, `probe_random`, `uniform_herding`, and `probe_herding`.
  - This is the main script to use when deciding whether probe-guided allocation is actually helping or merely adding noise.

## Recommended short run

Use a small number of tasks and epochs first:

```bash
python analysis/probe_guided_audit.py --tasks 3 --epochs 5 --seed 13
python analysis/compare_methods.py --tasks 3 --epochs 5 --seed 13
python analysis/herding_bic_pilot.py --method compare --tasks 3 --epochs 5 --seed 13
python analysis/replay_ablation.py --method compare --tasks 3 --epochs 5 --seed 13
```

If those results look plausible, increase to 5 tasks before doing a full run.
