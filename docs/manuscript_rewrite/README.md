# Manuscript Rewrite Plan

## Purpose

This directory is the final content plan for a writing-only rewrite. The paper
will present Uniform Herding as the proposed replay method. The ablations and
resource sweeps are supporting evidence about that method; they are not the
paper's stated novelty.

No manuscript source, experiment output, generated figure, table value, or
implementation is changed by this plan review. Later writing patches must use
the current implementation and regenerated material as their evidence.

## Evidence Hierarchy

Use the following order when resolving a claim:

1. Current implementation for algorithmic behavior.
2. Regenerated files under `manuscript/data/` for reported numbers.
3. Current manuscript for existing notation and document integration.
4. Cited primary literature for prior methods and terminology.

If these sources disagree, stop the writing patch and resolve the discrepancy;
do not silently choose the version that produces a cleaner narrative.

## Verified Method Distinction

- `iCaRLReplayBank` herds a class once when it first appears, stores the
  resulting prioritized order, and keeps a prefix when the class quota later
  decreases.
- `UniformHerdingReplayBank` uses the same greedy herding primitive but
  reselects every observed class at each task boundary in the current feature
  representation.
- Uniform Herding has an active selected set with budget `M`. Replay draws come
  from this set.
- Uniform Herding also retains a bounded candidate pool. With the current
  implementation default `pool_multiplier = 3`, its boundary footprint is at
  most `3M` when all class quotas are positive and the candidate pools have
  reached their boundary caps. Newly observed classes can retain their full
  current-task stream before their first rebuild, so the `3M` statement is a
  boundary statement, not a mid-task total-storage guarantee.
- The regenerated main comparison is between complete protocols. Uniform
  Herding and iCaRL also differ in training objective, and the static bank uses
  its native head-logit readout. The comparison therefore does not identify the
  refresh policy as the sole cause of the observed gap.

## Central Claim

Uniform Herding is a representation-refresh replay method with uniform active
class allocation and a bounded candidate pool. Under the evaluated CIFAR-100
protocol, it obtains higher mean final accuracy and lower mean forgetting than
the faithful iCaRL and static-bank baselines at the reported active budget and
retrieval count. Within-method analyses show the behavior associated with
NME readout, herding selection, distillation, and active-budget changes.

The paper must not claim universal superiority, optimality of `pool_multiplier
= 3`, or that the iCaRL gap is caused by refresh alone.

## Plan Files

- `research_basis.md`: evidence boundary and paper-level narrative rationale.
- `venue_plan.md`: conditional submission-format requirements; no venue is
  assumed by the content rewrite.
- `main_plan.md`: title, abstract, integration, and global terminology.
- `introduction_plan.md`: problem, method proposal, comparison, and claims.
- `related_work_plan.md`: precise positioning against iCaRL and replay work.
- `method_plan.md`: formal method, active/candidate storage, and baselines.
- `results_plan.md`: main comparison, supporting analyses, and diagnostics.
- `discussion_plan.md`: interpretation, storage trade-offs, and limitations.
- `conclusion_plan.md`: final method-level claims and next experiment.
- `appendix_plan.md`: reproducibility and storage-accounting additions.
- `data_labels_plan.md`: label-only handling for generated render inputs.
- `verification_plan.md`: final evidence and consistency checks.
- `final_review_checklist.md`: reader-perspective questions for the final pass.

## Non-Negotiable Rules

- Do not make component analysis the primary contribution.
- Do not call Uniform Herding a `reference` method.
- Do not use `fixed memory` or `fixed-budget` to mean total retained storage
  unless the candidate-pool overhead is explicitly included.
- Use `active exemplar budget` for `M`.
- Do not attribute a method gap to one component without a matched one-factor
  experiment.
- Do not turn three-seed paired tests into broad statistical claims.
- Keep class-incremental learning as the task setting, not as a novelty
  disclaimer or the title's main subject.
- Do not change manuscript source files until all plan files are accepted.
