# Final Research Execution Plan

## Research Claim

The project will test one claim:

> Exposure-Debt Ghost Bank improves rare-class learning because it measures class underexposure during training and uses that measurement to control minority-memory retrieval.

The contribution is not "a bank." The contribution is the exposure-debt controller.

## Repository Structure

Use this structure:

`verification/`

Contains numerical checks for the formal definition. This is where we verify that the exposure, debt, and retrieval equations are correct, stable, deterministic, and internally consistent.

`empirical/`

Contains experiments and evidence. This includes synthetic controlled experiments, real benchmark experiments, raw outputs, and analysis.

`docs/`

Contains the research definition, literature framing, method specification, and experiment protocol.

## Stage 1: Formal Verification

Goal:

Verify the mathematical machinery before claiming any empirical result.

The verification script must check:

1. Exposure accumulation is correct.
2. Exposure debt is always non-negative.
3. Classes with zero debt receive zero debt-based retrieval.
4. Retrieval allocation is bounded by the retrieval budget.
5. Under imbalanced sampling, minority classes accumulate more debt than majority classes.
6. Increasing retrieval budget increases compensation but does not change the definition of debt.

This stage does not prove the method works. It proves the formal definition is implemented correctly.

## Stage 2: Controlled Synthetic Experiment

Goal:

Test the mechanism in a setting where we control the imbalance.

Use synthetic Gaussian classification first because it lets us isolate the claim:

`Does debt-controlled retrieval outperform static retrieval when the data-generating process is known?`

Methods:

1. Baseline: standard minibatch training.
2. Static bank: same bank capacity and retrieval budget, but retrieval is not controlled by debt.
3. Exposure-Debt Ghost Bank: same bank capacity and retrieval budget, with debt-controlled retrieval.

Controlled variables:

- same synthetic data seed
- same train/test split
- same classifier
- same optimizer
- same number of steps
- same minibatch size
- same bank capacity
- same retrieval budget

Primary result:

`Exposure-Debt Ghost Bank > Static Bank > Baseline`

on minority recall or balanced accuracy.

If this does not happen on controlled synthetic data, do not move to real datasets yet.

## Stage 3: Real Benchmark Experiment

Goal:

Test whether the method survives standard community benchmarks.

Use real datasets only after the synthetic mechanism check passes.

Recommended first real benchmarks:

1. CIFAR-10-LT for simple long-tailed image classification.
2. CIFAR-100-LT for harder long-tailed image classification.
3. A tabular or fraud-style rare-event dataset only if the paper targets rare-event classification rather than vision.

The real benchmark stage should compare against:

1. Cross-entropy.
2. Class-balanced loss.
3. Focal loss.
4. LDAM-DRW.
5. Logit adjustment.
6. Static minority bank.
7. Exposure-Debt Ghost Bank.

Published bank methods should be baselines only when the task and implementation are comparable. If the original method is for a different setting, cite it in related work and compare against a matched static bank instead.

## Decision Rule

Continue the project only if:

`Exposure-Debt Ghost Bank > Static Bank`

under matched bank capacity, matched retrieval budget, and matched optimizer steps.

Do not claim novelty if the only improvement is over plain cross-entropy.

## What To Track

Track only what supports the claim:

1. `A_c(t)`: accumulated exposure per class.
2. `T_c(t)`: target exposure per class.
3. `D_c(t)`: exposure debt per class.
4. `r_c(t)`: retrieved bank examples per class.
5. per-class recall.
6. balanced accuracy.
7. macro-F1.
8. overall accuracy.

Do not add extra diagnostics unless a failure needs explanation.

## Final Standard

The experiment is valid if it answers:

`Does measuring exposure debt and using it for retrieval improve rare-class learning compared with static memory?`

That is the core of the research.
