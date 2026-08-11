# Plan: `manuscript/sections/method/method.tex`

## Objective

Make the method section a faithful mathematical specification of Uniform
Herding, its active/candidate storage semantics, the training and prediction
rules, the baselines, and the evaluation protocol. This section must be the
technical source of truth for the rest of the paper.

## Section Patches

### Heading and Task Setup

Rename `Method and Experimental Setup` to `Uniform Herding and Experimental
Setup`. Retain the current CIFAR-100 task definition, split, held-out probe and
validation sets, and task indexing. Introduce the proposed method before the
baselines.

### Model and Training

Retain the verified ResNet-18, cosine-margin head, imprinting, optimizer,
augmentation, epoch, precision, and seed details. Replace `reference` with
`Uniform Herding` wherever it identifies the proposed configuration.

Document the actual Uniform Herding loss: cross-entropy plus the
temperature-scaled softmax KL term on old classes. State the teacher timing,
`lambda=1`, and temperature `2` for the main configuration. Document the
target-dependent margin removal used before the KD comparison if it is present
in the final source code path; do not replace this implementation detail with a
generic KD equation that describes a different objective.

### Active Allocation

Define `M` as the active selected exemplar budget and `C_t` as the number of
observed classes. Preserve the uniform floor-and-remainder allocation. State
that in the reported CIFAR-100 runs every class has enough candidates, so the
selected sets sum to `M` at each completed boundary. For a general candidate
set, state the implementation's actual rule: a class can select no more than
its quota or the number of available candidates.

### Candidate Storage and Refresh

Use separate notation for selected exemplars, persistent candidates, and the
current-task stream, for example `E_c^t`, `B_c^t`, and `P_{c,cur}^t`.

- Old classes enter a task with a bounded persistent candidate pool.
- Examples of classes in the current task are accumulated until the boundary;
  a class awaiting its first rebuild can retain its full current-task stream.
- At the boundary, herding is applied to the candidates available for each
  class in the current representation.
- The selected set is the herding output. The persistent pool is then rebuilt
  from that output plus bounded leftover candidates.
- The implementation default is `rho = pool_multiplier = 3`. The current
  runner does not expose this argument as a separate experiment override, so
  describe it as the implementation default rather than as a swept or tuned
  configuration value. At a completed
  boundary in the reported setting, the candidate pool is bounded by `rho q_c`
  per class and by `rho M` in total. This is not a guarantee for the transient
  current-task stream before its first rebuild.
- `M` is the active replay footprint; `rho M` is the boundary candidate
  footprint. The paper must not describe the method as having total storage
  `M`.

Retain the greedy herding equation, but apply it to the actual candidate set.
Make clear that Uniform Herding refreshes every observed class at every task
boundary, whereas iCaRL does not re-herd old classes.

Do not describe the candidate pool as a statistically uniform sample of the
entire historical stream. The implementation uses class-local bounded
reservoir insertion for classes whose cap is known, plus a full transient
stream before the first rebuild; the paper should call this a bounded candidate
pool.

### Replay and Prediction

State that replay sampling draws from the active selected set, not from the
candidate pool, and uses the configured retrieval budget with replacement.
Define NME from the selected-exemplar means. State the head-logit alternative
only as the designated ablation and baseline protocol.

### Faithful Baselines and Supporting Analyses

Add a dedicated iCaRL paragraph:

- herd each new class once from its full current-task pool;
- store the prioritized order;
- truncate old classes to the new quota by keeping the prefix;
- recompute NME means in the current feature space over retained exemplars;
- discard the transient current-task pool at the boundary;
- train with the iCaRL BCE-style target objective.

Describe the static bank separately: random replacement within class pools,
the same nominal active budget and retrieval count, and native head-logit
prediction.

State explicitly that T1 compares complete methods. It matches the nominal
active replay budget and retrieval count, but it does not match total retained
storage, training objective, or readout across all rows. T1 is therefore not a
one-factor refresh-policy experiment.

Rename `Baselines and Ablations` to `Baselines and Supporting Analyses`. Treat
T2 as one-change comparisons within Uniform Herding and T3 as active-budget and
retrieval sensitivity, not as proof of independent causal contributions.
