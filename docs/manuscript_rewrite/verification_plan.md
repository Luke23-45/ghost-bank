# Plan: Final Verification Before Any Manuscript Build

This file describes checks to run after the writing patches are implemented.
The current task does not build the manuscript.

## Implementation Fidelity

- Confirm `iCaRLReplayBank` herds new classes once, truncates old ranked lists,
  recomputes current-space means, and discards transient pools.
- Confirm `UniformHerdingReplayBank` refreshes every class, stores selected plus
  bounded leftover candidates, uses the default `pool_multiplier=3`, and
  provides an active selected set with the reported budget at boundaries.
- Confirm the runner dispatch maps `uniform_herding`, `icarl`, and `static` to
  the intended bank and method implementations.
- Re-run the targeted herding and bank tests and require them to pass. The
  current baseline observed in this worktree is 16 passing tests across the
  identity, bank, and strategy suites; this is a baseline observation, not a
  substitute for the post-edit verification.

## Claims and Numbers

- Every T1, T2, T3, and appendix number matches the regenerated source exactly.
- The main comparison is described as complete-protocol evidence, never as a
  one-factor refresh-policy test.
- `M` is consistently called the active selected exemplar budget.
- The `rho=3` candidate-pool bound and transient first-rebuild behavior appear
  in Method and Appendix, and the storage trade-off appears in Discussion.
- Claims about NME, KD, selection, head geometry, active budget, and retrieval
  are explicitly limited to the evaluated protocol and tested range.
- No claim says that `rho=3` is optimal or that more total storage is free.
- Do not state that `rho` was supplied by the experiment configuration unless
  the runner/configuration path is changed and the results are regenerated.

## Mathematical and Protocol Consistency

- Quota notation, class indexing, remainder allocation, selected-set means,
  candidate sets, and boundary timing agree with the implementation.
- The method states when current-task candidates are transient and when old
  candidate pools are bounded.
- The loss equation matches the actual CE/KL implementation, including the
  margin-unbinding operation used for KD if retained in the source.
- iCaRL's BCE-style objective and NME readout are not silently replaced by the
  Uniform Herding objective.
- The T1 table and caption distinguish common active/retrieval budgets from
  unequal total candidate storage and method-specific readouts.

## Framing and Terminology

- The title names Uniform Herding and representation-refresh replay.
- `reference` is absent from narrative prose and manuscript-facing labels when
  it refers to Uniform Herding.
- `component analysis` appears only as supporting analysis, never as the main
  contribution.
- No apology, novelty disclaimer, universal ranking, or refresh-only causal
  claim remains.
- `Class-incremental learning` remains where it defines the task or literature,
  not as the paper's novelty framing.

## File and Venue Checks

- Review all changed `.tex` text for unresolved labels and cross-references.
- Confirm no regenerated numeric file or figure asset changed in the writing
  patch.
- Confirm the generic source order remains main content, bibliography,
  appendix until a venue-specific format patch is selected.
- If NeurIPS is selected, apply `venue_plan.md` separately and verify the
  official style, page limit, references/appendix/checklist order, and required
  checklist before submission.
