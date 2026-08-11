# Research Basis for the Rewrite

## Paper Type

The manuscript should read as a method paper with empirical validation. The
method is Uniform Herding; the ablations explain the behavior of the proposed
configuration, and the resource sweeps characterize its operating range. The
paper is not framed as a component-analysis paper and does not need to repeat a
novelty disclaimer in every section.

## Argument Flow

Use one continuous argument:

`bounded replay problem -> Uniform Herding design -> precise protocol -> main
method comparison -> supporting within-method evidence -> limitations and
storage trade-offs -> conclusion`

This is the same professional structure used by strong empirical method
papers. The strength comes from a clear question, faithful baselines, exact
method specification, and evidence matched to each claim, not from repeatedly
labeling the work as incremental or non-breakthrough.

## External Technical Anchor

The original iCaRL paper is the primary source for the comparison template:
herding-based prioritized exemplars, a fixed total exemplar budget, and
nearest-mean-of-exemplars prediction. Use it to define prior work, not to claim
that the current implementation is identical in every training detail.

Primary source: https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html

## Evidence Boundary

- The implementation verifies the refresh and storage behavior.
- T1 supports an end-to-end comparison of complete implementations.
- T2 supports within-Uniform-Herding comparisons.
- T3 supports sensitivity statements over the tested active budgets and
  retrieval counts.
- The current evidence does not support a refresh-only causal claim, a total
  storage-matched claim, a universal ranking, or an optimal candidate multiplier.

## Writing Consequence

State the method directly, quantify the main comparison, disclose the active
versus candidate storage distinction, and use the Discussion for the limits of
interpretation. Do not make the title or abstract carry the conclusion that the
work is incremental.
