# Problem Statement

We study supervised learning under severe class imbalance, where one or more classes appear far less often than the majority classes. In this regime, standard empirical risk minimization tends to optimize the frequent classes more effectively than the rare ones, because the minority classes contribute fewer training updates in expectation and are therefore underrepresented in the optimization signal.

The literature on imbalanced learning shows several established responses to this problem: loss reweighting, focal-style loss reshaping, class-balanced weighting, careful minibatch and optimization tuning, and memory- or prototype-based tail enrichment. The present project asks whether Exposure-Debt Ghost Bank, defined as a persistent auxiliary memory controlled by a per-class exposure ledger, can provide an additional and measurable benefit beyond those existing ideas.

The research problem is therefore:

1. Train a classifier on a severely imbalanced dataset.
2. Compare a strong baseline training pipeline against static memory variants and the same pipeline augmented with Exposure-Debt Ghost Bank.
3. Measure whether exposure-debt controlled retrieval improves minority-class performance without unacceptable loss in overall performance.

The paper should frame the issue as minority-class underexposure and tail-class instability, not as a vague claim that the majority gradient "overrides" the minority gradient. The mathematically relevant point is that the expected minority contribution to the training objective is too sparse under ordinary sampling.

This project treats the bank itself as insufficient for novelty. The intended contribution is the exposure-debt controller: a mechanism that tracks per-class training exposure and retrieves minority information when a class falls below its target exposure schedule.
