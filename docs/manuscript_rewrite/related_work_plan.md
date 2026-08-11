# Plan: `manuscript/sections/related_work/related_work.tex`

## Objective

Position Uniform Herding precisely. The section should establish the iCaRL
template, identify the refresh-policy and storage distinction, and explain why
the comparison is informative without claiming that the paper invented
herding, NME, distillation, or a new metric.

## Required Edits

1. Keep a concise overview of regularization, distillation, replay, and
   exemplar selection.
2. Present iCaRL as the canonical comparison point: herding for new classes,
   prioritized exemplars, quota reduction by prefix truncation, replay, and
   nearest-mean-of-exemplars prediction.
3. State that Uniform Herding shares the greedy herding primitive and uniform
   active allocation but refreshes every observed class in the current feature
   space. Explain that it retains a bounded candidate pool to make this refresh
   possible.
4. Distinguish the methods' training objectives and readout protocols. Uniform
   Herding uses CE plus temperature-scaled softmax KL and NME by default; the
   faithful iCaRL implementation uses its BCE-style target objective and NME;
   the static bank uses its native head-logit prediction.
5. Connect the classifier/readout literature to the paper's controlled
   within-method comparisons, without claiming that one readout is universally
   correct.
6. Close by stating the actual contribution positively: a replay method and an
   empirical evaluation of its behavior under the stated protocol.

## Language Constraints

- Remove `our contribution is not...` constructions and replace them with a
  positive description of what is evaluated.
- Do not call Uniform Herding a `reference`.
- Do not imply that the main comparison proves the refresh policy alone causes
  the observed difference.
- Do not imply that the candidate pool gives the method the same total storage
  as iCaRL or the static bank.
- Do not overstate novelty beyond the documented refresh-policy method.
