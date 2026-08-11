# Plan: `manuscript/sections/conclusion/conclusion.tex`

## Objective

End with the method, the measured evidence, and the precise boundary of the
claim. Do not end with a generic component-analysis summary or a novelty
disclaimer.

## Required Content

1. Name Uniform Herding as the proposed method.
2. State its defining mechanism: uniform allocation of an active selected
   exemplar set, refresh of every observed class in the current representation,
   and a bounded candidate pool used for refresh.
3. State the T1 outcome against faithful iCaRL and the static bank using the
   exact regenerated values and the evaluated active/retrieval budgets.
4. Summarize the supporting evidence: NME, herding, KD retention, and
   active-budget/retrieval sensitivity.
5. State the storage distinction: the active set is `M`, but the candidate pool
   adds implementation-dependent retained examples and is not included in the
   nominal active-budget comparison.
6. State the boundary correctly: the iCaRL gap is an end-to-end method result,
   not proof that refresh alone causes the improvement.
7. Give the matched refresh-isolation experiment as the next empirical step.

Avoid `not a breakthrough`, `not a new algorithm`, `not a general ranking`, and
similar defensive constructions. Use factual scope and storage language
instead.
