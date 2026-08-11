# Plan: `manuscript/appendix/appendix.tex`

## Objective

Make the appendix auditable for Uniform Herding, its storage semantics, and the
reported evidence. The appendix supplies details; it must not introduce new
interpretations that the main text does not support.

## Required Additions

- Replace all `reference` narrative and labels with `Uniform Herding`.
- Add separate protocol fields for active selected budget `M`, candidate-pool
  multiplier `rho=3`, boundary candidate capacity, and replay source.
- State that the boundary candidate footprint is at most `rho M` in the
  reported setting, while a newly observed class can retain its full transient
  stream before its first rebuild.
- State that replay samples come only from the selected active set.
- Clarify that iCaRL's persistent bank is capped at its nominal active budget,
  herds each class once, truncates by prefix, and discards the current-task pool
  at the boundary.
- Clarify that the static bank uses random replacement and native head-logit
  prediction in T1.
- Keep all regenerated A1--A6 numbers unchanged unless a source-of-truth audit
  proves a discrepancy.
- Update figure captions from `reference` to `Uniform Herding`; keep existing
  asset filenames unless a coordinated path change is necessary.
- Keep per-seed, task-level, quota, and compute tables as supplementary
  evidence, and state that the three seeds are training seeds sharing one class
  partition.

## Storage Accounting Rule

Do not present the active budget and candidate footprint as one number. If an
observed total candidate size is not present in the regenerated data, report the
implementation bound and transient behavior only; do not invent an empirical
storage measurement.
