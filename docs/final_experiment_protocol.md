# Final Experiment Protocol

## Purpose

The experiment must be built around the idea, not around a convenient dataset. The idea is:

> Minority classes fail partly because they receive insufficient effective optimization exposure. Exposure-Debt Ghost Bank should help only if measuring this underexposure and using it to control memory retrieval improves minority learning beyond static memory and strong imbalance baselines.

Therefore, the experiment must test exposure control directly.

## Core Claims to Test

The paper should test three claims.

Claim 1:

`exposure debt is measurable and correlates with minority-class failure`

Claim 2:

`debt-controlled retrieval improves minority performance beyond static memory retrieval`

Claim 3:

`the improvement is not explained only by extra compute, extra samples, or ordinary oversampling`

If these three claims are not supported, the research idea is not strong enough for a method paper.

## Experimental Design

Use a staged design.

Stage 1: Controlled Imbalance

Use a standard dataset and construct controlled imbalance ratios. This gives causal control over the class-frequency variable.

Recommended settings:

- imbalance ratio `10`
- imbalance ratio `50`
- imbalance ratio `100`
- imbalance ratio `200`

The purpose is not to claim realism. The purpose is to isolate whether exposure-debt control behaves as expected when imbalance severity is known.

Stage 2: Natural Long-Tailed Data

Use a natural long-tailed benchmark after the controlled experiment works.

The purpose is to show that the method is not only exploiting an artificial setup.

Stage 3: Stress Tests

Test conditions where a reviewer would expect the method to fail:

- very small bank
- noisy minority labels
- high minority intra-class diversity
- no warmup
- equal compute budget

These stress tests protect the paper from the criticism that the method works only in a narrow easy setting.

## Baselines

The baselines should be divided into three groups.

Group A: Standard Imbalance Baselines

1. Cross-entropy.
2. Class-balanced loss.
3. Focal loss.
4. LDAM-DRW.
5. Logit adjustment.
6. Decoupled classifier retraining, if the task is image classification.

Group B: Memory Baselines

1. Static minority replay bank.
2. Static inverse-frequency replay bank.
3. Static prototype bank, if the final method uses feature prototypes.
4. Published memory-bank method if it matches the task and can be reproduced fairly.

Group C: Proposed Method

1. Exposure-Debt Ghost Bank with count-based exposure.
2. Exposure-Debt Ghost Bank with loss-weighted exposure.
3. Exposure-Debt Ghost Bank with different target schedules.

The critical comparison is:

`static memory bank vs Exposure-Debt Ghost Bank`

The published memory-bank papers should be treated as related baselines when the experimental setting matches. If a published method was designed for a different task, it should be discussed in related work and only implemented if adaptation is fair.

## Controlled Variables

Keep the following fixed across methods:

- dataset split
- architecture
- optimizer
- learning-rate schedule
- augmentation
- batch size
- number of optimizer steps
- random seeds
- evaluation protocol
- total retrieval budget, when comparing memory methods

The memory comparisons must be especially strict. A static bank and Exposure-Debt Ghost Bank should receive the same bank capacity and same maximum retrieval budget. Otherwise, reviewers can argue that the proposed method wins only because it sees more data.

## Fairness Protocols

Use two fairness protocols.

Protocol 1: Equal Optimizer Steps

All methods receive the same number of optimizer updates. This tests final performance under a standard training budget.

Protocol 2: Equal Effective Sample Budget

Memory methods are constrained so that the total number of bank-retrieved examples is matched. This tests whether exposure-debt control is better than static retrieval when both see the same number of extra examples.

The method should be reported under both protocols.

## Parameters to Track

Track these during training.

Exposure Parameters:

- `a_c(t)`: step-level exposure for class `c`
- `A_c(t)`: accumulated exposure for class `c`
- `T_c(t)`: target exposure for class `c`
- `D_c(t)`: exposure debt for class `c`
- `r_c(t)`: retrieval count assigned to class `c`

Bank Parameters:

- bank size per class
- insertion count per class
- replacement count per class
- average item age
- retrieved item count per class
- duplicate retrieval rate
- hard-example ratio
- prototype drift, if prototypes are used

Optimization Parameters:

- per-class training loss
- per-class validation loss
- per-class recall
- per-class gradient norm if feasible
- minority-to-majority gradient contribution ratio if feasible
- classifier weight norm per class

Outcome Metrics:

- macro-F1
- balanced accuracy
- minority recall
- minority AUCPR for binary or one-vs-rest rare-class settings
- many-shot, medium-shot, few-shot accuracy for long-tailed recognition
- overall accuracy as a secondary metric

The exposure-debt curves are not optional. They are the evidence that the method is doing what the paper claims.

## Main Ablation Table

The main table should include:

1. Cross-entropy.
2. Best non-memory imbalance baseline.
3. Static minority replay bank.
4. Static inverse-frequency bank.
5. Exposure-Debt Ghost Bank.
6. Exposure-Debt Ghost Bank without warmup.
7. Exposure-Debt Ghost Bank with random retrieval.
8. Exposure-Debt Ghost Bank with debt tracking disabled.

The decisive row is:

`Exposure-Debt Ghost Bank with debt tracking disabled`

If disabling debt tracking does not hurt performance, the proposed mechanism is not validated.

## Primary Figures

The paper should include these figures:

1. Per-class exposure debt over training.
2. Per-class retrieval allocation over training.
3. Minority recall vs exposure debt.
4. Performance vs imbalance ratio.
5. Static bank vs debt-controlled bank under equal retrieval budget.

These figures directly connect the mechanism to the result.

## Success Standard

The minimum success pattern is:

`Exposure-Debt Ghost Bank > Static Bank > Strong Baseline`

on minority-sensitive metrics, while overall accuracy remains within an acceptable tolerance.

The stronger success pattern is:

`Exposure-Debt Ghost Bank > Static Bank`

under equal optimizer steps and equal effective sample budget.

That result would show that the controller matters, not just the presence of a bank.

## Failure Interpretation

If Exposure-Debt Ghost Bank only beats cross-entropy, the method is not publishable as a strong contribution.

If static memory performs the same as Exposure-Debt Ghost Bank, the exposure-debt controller is not justified.

If the method improves minority recall but destroys majority performance, the objective is not balanced enough.

If the method works only with large banks, the contribution may be too expensive or too close to replay.

## Final Experimental Principle

Do not ask:

`Does our method improve on some dataset?`

Ask:

`Does measuring exposure debt and using it to control memory retrieval explain and improve minority-class learning beyond strong baselines and static memory?`

That is the experiment the paper must run.
