# Plan: `manuscript/sections/introduction/introduction.tex`

## Objective

Use the standard method-paper progression: problem, concrete design gap,
proposed method, evaluation protocol, findings, and scope. The introduction
must be about replay under representation change, not about defending the
paper's level of novelty.

## Paragraph Structure

1. Define class-incremental classification and the bounded active replay
   constraint. Explain why exemplars selected in one representation may become
   less suitable as the representation changes.
2. Identify the design gap: exemplar refresh, training objective, classifier,
   readout, and retrieval are commonly bundled. State that this makes it hard
   to interpret a replay method's behavior under a common active budget.
3. Introduce Uniform Herding. It uses uniform active class allocation and
   greedy herding, but refreshes all observed classes after each task in the
   current representation. Its candidate pool is distinct from the active
   replay set and is bounded only at task boundaries by the implementation's
   multiplier.
4. Contrast the faithful iCaRL baseline precisely: arrival-time herding,
   prioritized prefix truncation, NME prediction, and the iCaRL-style training
   objective. State that the main comparison is between complete protocols, not
   a refresh-only intervention.
5. Preview the evidence in order: main comparison, within-Uniform-Herding
   ablations, active-budget and retrieval sensitivity, and task-age diagnostics.
   Give only claims supported by T1--T3 and the appendix data.
6. State the scope and storage trade-off briefly: the method fixes the active
   replay budget but retains additional candidates for refresh.

## Contribution List

Use positive, method-centered contributions:

- We define Uniform Herding, a representation-refresh replay method with
  uniform active class allocation and a bounded candidate pool.
- We provide an end-to-end comparison with faithful iCaRL and static-bank
  baselines at a common nominal active budget and retrieval count, while
  reporting the differences in readout, objective, and retained candidates.
- We characterize the proposed configuration with targeted within-method
  comparisons and active-budget/retrieval sweeps.

Do not call the third item the principal novelty. Do not claim that the iCaRL
difference isolates refresh, and do not call the proposed method a reference
configuration.

## Terminology Patches

- Replace `reference configuration` with `Uniform Herding` or `proposed
  method` when the context requires the role.
- Replace `this paper is a controlled empirical study, not a claim...` with the
  positive method introduction and a factual scope sentence.
- Use `active exemplar budget` for `M`.
- Use `candidate pool` or `candidate storage` for the additional retained
  examples; do not call it active replay memory.
