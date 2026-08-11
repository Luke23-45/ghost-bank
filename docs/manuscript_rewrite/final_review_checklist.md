# Final Manuscript Review Checklist

Use these questions for the final reading pass. A sentence should remain only
if it answers a reader's question, defines the method, reports evidence, or
limits an interpretation.

## Global Flow

- Does the title identify the research object rather than its verdict?
- Does the abstract state the problem, method, protocol, main evidence, and
  scope without reproducing the Introduction sentence by sentence?
- Does each section prepare the question answered by the next section?
- Is every claim supported by an equation, citation, implementation fact, or
  regenerated result?
- Does each paragraph have one job, with no repeated conclusion in different
  wording?
- Are active budget, candidate storage, retrieval budget, and total storage
  kept distinct everywhere?

## Introduction

- What problem does the method address, and why does representation change
  matter for replay?
- Is the proposed method introduced before its ablations?
- Does the reader understand the difference between selected exemplars and
  candidate storage before reaching the Method section?
- Does the comparison with iCaRL state what is and is not controlled?
- Do the contributions describe the method and evidence rather than sell an
  ablation as the novelty?

## Related Work

- Is each cited method described only for the role supported by its primary
  source?
- Does the section distinguish prior herding and NME from the proposed refresh
  policy?
- Are differences in training objective and readout stated without implying a
  refresh-only comparison?
- Does the section end by positioning this paper rather than repeating the
  Introduction or Method?

## Method

- Could a reader implement Uniform Herding from this section without guessing
  when candidates are collected, refreshed, selected, or discarded?
- Are `M`, `q_c`, the selected set, candidate pool, and transient stream defined
  before they are used?
- Does the herding equation operate on the actual candidate set?
- Does the text state that replay uses selected exemplars, not candidates?
- Does the loss equation match the CE/KL implementation, including the
  margin-unbinding step used for KD?
- Are iCaRL and the static bank specified as complete baselines, with their
  own objective and readout protocols?
- Does the section disclose the `rho=3` candidate-storage overhead and its
  boundary-only guarantee?

## Results

- Does the first subsection answer whether Uniform Herding performs better in
  the evaluated comparison?
- Are all numbers copied from the regenerated tables without recomputation or
  rounding changes?
- Are T2 comparisons described as within-method evidence rather than universal
  causal effects?
- Does T3 distinguish active-budget sensitivity from total-storage sensitivity?
- Are figures used to expose measured behavior rather than introduce new
  claims?
- Is the final-task zero-forgetting convention explained once and precisely?

## Discussion

- Does the Discussion interpret the Results instead of restating every table?
- Are mechanistic statements limited to comparisons that actually hold other
  choices fixed?
- Is the candidate-storage and refresh-computation trade-off explicit?
- Are dataset, order, architecture, seed, range, and matched-control limits
  stated without defensive repetition?
- Does the proposed next experiment change only refresh after fixing the other
  variables?

## Conclusion

- Does the conclusion name the method and its mechanism in one sentence?
- Does it summarize the principal measured result without introducing a new
  claim?
- Does it preserve the active-versus-candidate storage distinction?
- Does it state the refresh-causality boundary and the next experiment briefly?

## Appendix and Artifacts

- Can a reader audit protocol, seeds, metrics, quotas, compute, and figures?
- Are supporting tables and captions free of stale `reference` labels?
- Are generated numeric values unchanged by writing edits?
- Are all labels and references defined exactly once?
- Is the venue-specific checklist handled separately from content framing?
