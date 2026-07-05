# Finalized Research Idea

## Title Candidate

Ghost Bank: Exposure-Debt Controlled Memory for Extreme Class Imbalance

## Core Idea

The contribution should not be framed as "a memory bank for rare classes." That overlaps too much with replay, prototype memory, and prior long-tailed recognition methods.

The stronger idea is:

> Ghost Bank is a closed-loop minority-memory mechanism that tracks each class's effective training exposure and retrieves stored minority information only when that class has accumulated exposure debt.

This gives the method a sharper identity. The bank is not a static oversampling buffer. It is controlled by a training-time signal that measures whether each class has received enough useful optimization attention.

## Research Gap

Existing methods usually address imbalance through one of the following:

- loss reweighting or reshaping
- resampling
- logit adjustment
- classifier decoupling
- memory or prototype enrichment

These approaches are useful, but many are static or frequency-driven. They often use class counts as a proxy for imbalance. The proposed gap is that class count alone does not tell us whether a class has received sufficient effective optimization exposure during training.

Ghost Bank targets this gap by using a dynamic exposure ledger. The model tracks a per-class exposure signal during training and uses the bank to compensate for classes whose recent or cumulative exposure is below target.

## Proposed Contribution

The proposed contribution is a method called Exposure-Debt Ghost Bank.

It has three components:

1. Exposure ledger: tracks how much effective training signal each class has received.
2. Minority memory bank: stores selected rare-class examples, features, prototypes, or anchors.
3. Debt-controlled retrieval: retrieves bank entries according to measured exposure debt, not only according to class frequency.

The key novelty is the control policy:

`class is retrieved when its measured exposure is below its target exposure`

rather than:

`class is retrieved because it is globally rare`

## Why This Is Different From Existing Memory Banks

A standard memory bank stores examples or features and reuses them.

Ghost Bank should be defined more narrowly:

- It stores minority-class information.
- It maintains a per-class exposure ledger.
- It computes exposure debt during training.
- It allocates retrieval budget according to that debt.
- It can be activated after a warmup stage to avoid damaging representation learning.

This separates the method from plain replay, plain oversampling, and static prototype banks.

## Method Summary

At each training step:

1. Sample a normal minibatch.
2. Update the exposure ledger using the classes present in the minibatch.
3. Compute the exposure debt for each class.
4. Retrieve ghost-bank entries for classes with high exposure debt.
5. Train on the original minibatch plus a bank loss.
6. Update the bank using selected current examples, features, or prototypes.

## Publishability Condition

This idea is worth pursuing for a top-tier venue only if experiments show at least one of the following:

1. The exposure-debt controller improves minority-class metrics over strong baselines.
2. The same bank without exposure-debt control performs worse.
3. The method remains effective under extreme imbalance ratios where static reweighting or resampling is unstable.
4. The exposure ledger explains training behavior, such as why certain tail classes fail even when their nominal sample count is not the lowest.

The required ablation is:

`baseline < static bank < exposure-debt ghost bank`

If exposure-debt control does not improve over a static bank, the method is not strong enough as a method paper.

## Final Paper Claim

The final claim should be conservative:

> We introduce an exposure-debt controlled memory mechanism for extreme class imbalance. Instead of relying only on class frequency, the method tracks how much effective training exposure each class receives and dynamically retrieves stored minority information when a class becomes underexposed. Experiments evaluate whether this closed-loop memory control improves minority-class performance beyond strong long-tailed learning baselines.

This is the strongest defensible version of the idea at the current stage.
