# Replay Analysis

This directory contains short, targeted scripts for diagnosing replay behavior and final method selection.

## What these scripts test

### H1: Memory budget is violated
- Question: does the exemplar bank stay within the intended fixed budget?
- Expected healthy signal: total stored exemplars stay near `memory_total`, never explode across epochs.

### H2: NME is consistent with linear evaluation
- Question: is the NME head usable, or is it collapsing relative to the linear head?
- Expected healthy signal: NME should be competitive with, or at least not far below, the linear head.

### H3: Herding is stable
- Question: does herding produce a stable exemplar set across seeds?
- Expected healthy signal: final accuracy and forgetting remain consistent across short sweeps.

## Scripts

- `herding_bic_pilot.py`
  - Ground-up rescue pilot that pairs fixed-memory replay with herding exemplar selection and a BiC-style bias-correction fit.
  - Compares `probe_guided_herding_bic`, `uniform_herding_bic`, and `frozen_baseline`.
  - Best used as the next decision-point experiment before any core-method rewrite.

- `replay_ablation.py`
  - Focused ablation that removes BiC and isolates the two remaining axes:
    - allocation source: `uniform` vs `probe-guided`
    - selection rule: `random` vs `herding`
  - Default `compare` mode runs `uniform_random`, `probe_random`, `uniform_herding`, and `probe_herding`.
  - Also includes `probe_blend_random` and `probe_blend_herding`, which constrain probe-guided allocation to stay near the uniform baseline.
  - This is the main script to use when deciding whether probe-guided allocation is actually helping or merely adding noise.

- `confirmation_sweep.py`
  - Runs the final comparison between `uniform_herding` and `probe_blend_herding` across multiple seeds.
  - Aggregates mean, std, min, and max for raw accuracy, NME, forgetting, and probe correlation.
  - Use this when you are ready to decide whether `probe_blend_herding` is stable enough to keep as the paper candidate.

## Recommended short run

Use a small number of tasks and epochs first:

```bash
python analysis/herding_bic_pilot.py --method compare --tasks 3 --epochs 5 --seed 13
python analysis/replay_ablation.py --method compare --tasks 3 --epochs 5 --seed 13
python analysis/replay_ablation.py --method probe_blend_herding --tasks 3 --epochs 5 --seed 13
python analysis/confirmation_sweep.py --method compare --seeds 13 17 23 --tasks 3 --epochs 5
```

If those results look plausible, increase to 5 tasks before doing a full run.
