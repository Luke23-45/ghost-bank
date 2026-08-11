# Plan: Generated Data Labels and Captions

## Scope

The regenerated numeric material under `manuscript/data/` is the numerical
source of truth. A writing patch must not hand-edit numbers, standard
deviations, deltas, or figure pixels.

## Label Policy

Replace manuscript-facing `Uniform herding (Reference)` and `Reference` labels
with `Uniform Herding`. Expand abbreviated variant labels only when needed for
clarity:

- `Ref. without KD` -> `Uniform Herding without KD`;
- `Ref. head-logit eval` -> `Uniform Herding, head-logit evaluation`;
- `Ref. linear head` -> `Uniform Herding, linear head`;
- `Ref. random selection` -> `Uniform Herding, random selection`.

Keep `iCaRL` and `Static bank` as the baseline names. Do not add `(Proposed)` to
every table row; the prose and contribution statement already establish the
method's role.

## Generated-File Rule

First determine whether the final manuscript consumes a generated `.tex` table
or reproduces the table manually. If a generator owns the labels, update the
generator and regenerate rather than hand-editing generated files. If the files
are only audit artifacts, leave them unchanged and patch the manuscript source
captions/rows. In either case, the numeric audit must cover:

- `T1_master_results.md` for the complete-method comparison;
- `T2_component_ablations.md` for within-method comparisons;
- `T3_resource_sensitivity.md` for active-budget and retrieval sweeps;
- `A1_protocol.md`, `A4_per_seed_metrics.md`, and `A6_bank_sizes.md` for
  protocol, seed, and quota checks.

The final text must not use `reference` to describe Uniform Herding, including
in captions for figures whose filenames still contain that word.
