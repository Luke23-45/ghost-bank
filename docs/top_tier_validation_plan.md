# Top-Tier Validation Plan

## Central Question

The paper must answer one question clearly:

> Does exposure-debt controlled memory improve minority-class learning beyond strong imbalance baselines and beyond static memory replay?

If the answer is not yes, the method is not ready for a top-tier method paper.

The experiment must be claim-led. The goal is not to find a dataset where the method looks good; the goal is to verify whether exposure debt is a useful control signal for minority-memory retrieval.

## Required Baselines

The minimum baseline set should include:

1. Standard cross-entropy.
2. Class-balanced loss.
3. Focal loss.
4. LDAM-DRW.
5. Logit adjustment.
6. Decoupled classifier retraining if the setting is image classification.
7. Static minority replay bank.
8. Static prototype or feature bank if the final method uses features.
9. Exposure-Debt Ghost Bank.

Published memory-bank methods should be included as baselines only when they are appropriate for the same task and can be reproduced fairly. Otherwise, they belong in related work and the fair experimental baseline is a carefully implemented static bank with matched capacity and retrieval budget.

The most important comparison is:

`static minority bank vs exposure-debt controlled ghost bank`

This isolates whether the proposed controller matters.

The second most important comparison is:

`exposure-debt controlled ghost bank vs exposure-debt ghost bank with debt tracking disabled`

This isolates whether the measured debt signal itself is responsible for the gain.

## Required Metrics

Use metrics that expose minority performance:

- macro-F1
- balanced accuracy
- minority recall
- minority AUCPR for binary or one-vs-rest rare-class settings
- many-shot, medium-shot, and few-shot accuracy for long-tailed image benchmarks

Overall accuracy should be reported, but it should not be the main success metric.

## Required Ablations

The method needs the following ablations:

1. No bank.
2. Static bank with uniform tail retrieval.
3. Static bank with inverse-frequency retrieval.
4. Exposure-debt controlled bank.
5. Count-based exposure debt.
6. Loss-weighted exposure debt.
7. Different bank sizes.
8. Warmup vs no warmup.
9. Raw-sample bank vs feature/prototype bank, if both are implemented.
10. Debt tracking enabled vs disabled.
11. Equal optimizer steps vs equal effective sample budget.

## Reviewer Failure Cases

The paper is weak if:

- the method only beats plain cross-entropy
- the static bank performs the same as the exposure-debt bank
- gains appear only after increasing compute or data exposure unfairly
- the method improves minority recall by destroying majority or overall performance
- the bank stores mislabeled hard examples and overfits to noise
- the paper claims novelty from memory alone

## Strong Evidence Pattern

The strongest result pattern would be:

1. Strong baselines improve over plain cross-entropy.
2. Static bank improves slightly or inconsistently.
3. Exposure-Debt Ghost Bank improves minority metrics consistently.
4. Overall accuracy remains within a small tolerance.
5. Exposure-debt curves explain when the bank activates and which classes benefit.

This pattern would support the claim that closed-loop exposure control is the contribution.

## First Experimental Setting

Start with controlled imbalance:

1. Use a standard benchmark.
2. Artificially create imbalance ratios such as `10`, `50`, `100`, and `200`.
3. Run all baselines with the same model and training budget.
4. Report mean and standard deviation across multiple seeds.

Then move to a natural long-tailed benchmark.

## Minimum Acceptance Standard

Before writing the paper as a method contribution, the project should show:

`Exposure-Debt Ghost Bank > Static Bank > Strong Baseline`

on at least one severe imbalance setting, with repeatable gains across seeds.

If the result is only:

`Exposure-Debt Ghost Bank > Cross-Entropy`

then the contribution is not strong enough.
