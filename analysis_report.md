# Analysis Report

## Scope

This report summarizes the analysis work done to understand why the original probe-guided replay direction was failing, what alternative replay structure works better, and what the final direction should be for the paper.

The work was driven by the logs and scripts under [`analysis/`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/analysis) and the run history recorded in [`logs.md`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/logs.md) plus the detailed run notes in [`docs/g1/log1.md`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/docs/g1/log1.md) and [`docs/g1/log2.md`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/docs/g1/log2.md).

## Problem Statement

The initial probe-guided method was not performing well enough to serve as a paper result. The core questions were:

1. Is the probe signal itself meaningful?
2. Is the allocation rule the real problem?
3. Is the gain coming from replay structure rather than probe guidance?
4. Can the method be made stable enough to support a paper claim?

The analysis pipeline was built to answer those questions experimentally instead of guessing from theory.

## Scripts Run

The following scripts were used in sequence:

1. [`analysis/probe_guided_audit.py`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/analysis/probe_guided_audit.py)
   - Short diagnostic run for the original probe-guided replay.
   - Used to inspect probe statistics, allocation shape, memory growth, and linear/NME results.

2. [`analysis/compare_methods.py`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/analysis/compare_methods.py)
   - Compared probe-guided replay, uniform replay, and frozen baseline.
   - Used as a broad sanity check before deeper changes.

3. [`analysis/herding_bic_pilot.py`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/analysis/herding_bic_pilot.py)
   - Ground-up pilot that paired replay with herding and BiC-style correction.
   - This helped show that BiC was unstable and not a good centerpiece for the method.

4. [`analysis/replay_ablation.py`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/analysis/replay_ablation.py)
   - Ablation that separated allocation source from selection rule.
   - Variants included:
     - `uniform_random`
     - `probe_random`
     - `uniform_herding`
     - `probe_herding`
     - `probe_blend_random`
     - `probe_blend_herding`

5. [`analysis/confirmation_sweep.py`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/analysis/confirmation_sweep.py)
   - Final confirmation sweep on the two finalists:
     - `uniform_herding`
     - `probe_blend_herding`
   - Ran across seeds `13`, `17`, and `23`.

## Findings By Stage

### 1. Original probe-guided method

The initial probe-guided runs were weak and inconsistent. In the early logs, the method lost badly relative to the baseline, and the bank behavior indicated a structural issue rather than a tuning issue.

Important observations from the earlier run history:

- The memory bank originally grew too much before the deduplication and pruning fixes.
- The probe-guided method remained worse than uniform replay even after memory accounting was fixed.
- BiC-style correction often produced unstable coefficients and degraded accuracy.

This was the first signal that probe-guided quota assignment was probably the wrong lever.

### 2. Herding + BiC pilot

The herding + BiC pilot showed that herding-based exemplar selection was more promising than the original probe-guided quota logic, but the BiC component was not reliable.

From [`docs/g1/log1.md`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/docs/g1/log1.md):

- `probe_guided_herding_bic`
  - raw accuracy: `0.1343`
  - BiC accuracy: `0.0120`
  - NME accuracy: `0.1493`
- `uniform_herding_bic`
  - raw accuracy: `0.0683`
  - BiC accuracy: `0.0130`
  - NME accuracy: `0.1517`
- `frozen_baseline`
  - raw accuracy: `0.0470`
  - BiC accuracy: `0.0347`
  - NME accuracy: `0.0333`

The BiC branch was clearly not helping. The negative `alpha` values were a strong warning sign that the correction layer was overfitting or compensating in the wrong direction.

### 3. Replay ablation

The replay ablation removed BiC and isolated the two remaining axes:

- allocation source: uniform vs probe-guided
- selection rule: random vs herding

The first ablation pass on seed `13` showed:

- `uniform_random`
  - raw accuracy: `0.2983`
  - NME: `0.3987`
  - probe Spearman: `0.1092`
- `probe_random`
  - raw accuracy: `0.3280`
  - NME: `0.3733`
  - probe Spearman: `0.0954`
- `uniform_herding`
  - raw accuracy: `0.3247`
  - NME: `0.4247`
  - probe Spearman: `0.1448`
- `probe_herding`
  - raw accuracy: `0.3200`
  - NME: `0.4233`
  - probe Spearman: `0.1083`
- `probe_blend_random`
  - raw accuracy: `0.3193`
  - NME: `0.3867`
  - probe Spearman: `0.0994`
- `probe_blend_herding`
  - raw accuracy: `0.3303`
  - NME: `0.4250`
  - probe Spearman: `0.1724`

This was the first sign that a constrained probe prior could be competitive, but one seed was not enough.

### 4. Confirmation sweep

The confirmation sweep compared only the two finalists across three seeds:

- `uniform_herding`
- `probe_blend_herding`

The aggregated results from [`logs.md`](/C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/logs.md) were:

#### `uniform_herding`

- raw accuracy:
  - mean `0.3262`
  - std `0.0063`
  - min `0.3173`
  - max `0.3310`
- NME accuracy:
  - mean `0.4283`
  - std `0.0033`
  - min `0.4247`
  - max `0.4327`
- probe Spearman:
  - mean `0.0862`
  - std `0.0377`
- raw forgetting:
  - mean `0.0552`
  - std `0.0164`
- raw backward transfer:
  - mean `0.3900`
  - std `0.0181`

#### `probe_blend_herding`

- raw accuracy:
  - mean `0.3160`
  - std `0.0024`
  - min `0.3130`
  - max `0.3190`
- NME accuracy:
  - mean `0.4286`
  - std `0.0030`
  - min `0.4257`
  - max `0.4327`
- probe Spearman:
  - mean `0.0762`
  - std `0.0422`
- raw forgetting:
  - mean `0.0648`
  - std `0.0296`
- raw backward transfer:
  - mean `0.3723`

## Interpretation

The confirmation sweep settled the question.

### What worked

- Herding was consistently the strongest structural improvement.
- `uniform_herding` was the best stable baseline across the multi-seed confirmation sweep.
- NME stayed strong under herding.

### What did not work

- Hard probe-guided allocation was unstable and generally weak.
- The BiC-style correction was not reliable and should not be the paper centerpiece.
- The constrained probe prior (`probe_blend_herding`) did not beat `uniform_herding` on the final multi-seed confirmation sweep.

### Why the result matters

The probe scores do contain some signal, but that signal is too weak to serve as the main memory-allocation rule. The strongest evidence now points to:

- probe scores as a diagnostic or auxiliary signal
- herding as the real contributor to improvement
- uniform fixed-memory herding as the most defensible final method

## Conclusion

The final decision from the analysis work is:

1. Do not build the paper around hard probe-guided quota allocation.
2. Do not build the paper around BiC.
3. Use `uniform_herding` as the main baseline / final method.
4. Treat probe-guided allocation as a negative result or at most a weak auxiliary idea.

In short: the ablation and confirmation sweep showed that herding is the stable gain, while probe-guided allocation does not hold up as the main contribution.

## Recommended Next Step

The next work should be paper-facing, not more method churn:

- freeze the final experimental setup around `uniform_herding`
- write the results section around the ablation evidence
- include the negative finding that probe-guided quota allocation was not stable
- present probe scores as a diagnostic signal rather than the main mechanism

If another experiment is run, it should be a final verification pass on `uniform_herding`, not another probe variant search.
