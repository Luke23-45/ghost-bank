# Plan: `manuscript/sections/results/results.tex`

## Objective

Present the evidence in method-paper order: complete method comparison first,
supporting within-method analyses second, resource sensitivity third, and
diagnostics last. Results should report what was measured and reserve causal
interpretation for the Discussion.

## 1. Main Method Comparison

Open with Uniform Herding and state that the comparison uses a common nominal
active replay budget of `M=2,000` and retrieval budget `b=64`. Immediately
disclose that Uniform Herding additionally retains candidate examples for
refresh, while iCaRL and the static bank retain their nominal active stores.
Also disclose the method-specific readout/training protocols from the appendix.

Report the exact regenerated T1 values:

- Uniform Herding: `44.99 +/- 1.00` average accuracy, `13.85 +/- 0.44`
  forgetting.
- iCaRL: `42.36 +/- 0.87` average accuracy, `19.46 +/- 0.10` forgetting.
- Static bank: `28.60 +/- 1.35` average accuracy, `55.86 +/- 1.42`
  forgetting.

Interpret this as an end-to-end comparison of complete protocols. Say that
Uniform Herding obtains higher mean accuracy and lower mean forgetting in this
setting; do not say that refresh alone caused the difference.

The table caption must use `active replay budget` rather than implying equal
total storage. Do not add a claim about statistical significance to T1 unless a
prespecified between-method analysis exists in the regenerated data.

## 2. Supporting Analyses Within Uniform Herding

Rename `Component Attribution` to `Supporting Analyses of Uniform Herding`.
Report the T2 comparisons in this order:

1. NME versus head-logit evaluation.
2. Herding versus random selection.
3. KD versus no KD.
4. Cosine-margin versus linear head.

Use the exact regenerated deltas. Describe them as matched within-method
comparisons under the locked protocol. The NME and selection results support
the usefulness of the corresponding choices in this implementation; the KD
result shows a larger effect on forgetting than on mean accuracy; the head
comparison is smaller over the tested configuration.

If paired-test labels remain in the table, state that they use three matched
seeds and treat them as supporting evidence. Do not make p-values the abstract,
conclusion, or central novelty claim.

## 3. Active-Budget and Retrieval Sensitivity

Rename `memory sensitivity` language to `active-budget sensitivity` where
needed. Varying `M` also changes Uniform Herding's candidate-pool cap because
the implementation uses `rho=3`; therefore, do not present T3 as an isolated
test of active storage alone. Report it as resource sensitivity over the tested
configuration.

Report the exact T3 deltas: memory 500 (`-8.17` accuracy, `+5.77`
forgetting), memory 4000 (`+2.66`, `-1.18`), retrieval 32 (`-1.71`, `+1.13`),
and retrieval 128 (`-0.18`, `+0.81`). State that these are tested-range
sensitivity patterns, not a scaling law or saturation result.

## 4. Diagnostics

Keep the accuracy-forgetting plane and task-age analysis. Use `Uniform Herding`
in prose and captions. Preserve the exact zero-forgetting convention for the
final task. Do not infer a trajectory or a causal mechanism from a descriptive
figure alone.

## Labels and Captions

Replace `reference` labels with `Uniform Herding`; keep asset filenames only if
renaming them is necessary for consistency, because filename changes create
avoidable path risks. Do not alter regenerated numeric values in this writing
patch.
