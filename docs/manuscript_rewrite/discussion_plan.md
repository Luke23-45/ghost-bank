# Plan: `manuscript/sections/discussion/discussion.tex`

## Objective

Interpret Uniform Herding as the proposed method without turning the Discussion
into a second Results section or a defense of the paper's novelty level.

## Structure

### Main Finding

Open by stating that Uniform Herding obtains higher mean final accuracy and
lower mean forgetting than the faithful iCaRL and static-bank baselines in the
evaluated protocol. Qualify this immediately: T1 is a complete-method
comparison, and the methods differ in refresh policy, training objective,
readout, and retained candidate storage. The result is not a refresh-only
causal estimate.

### Evidence About the Proposed Configuration

Interpret T2 in this order:

1. NME is strongly favored over head-logit evaluation within the Uniform
   Herding configuration.
2. Herding is favored over the random-selection variant under the same
   within-method protocol.
3. KD primarily improves retention in the reported objective.
4. The cosine-margin versus linear-head difference is smaller in this tested
   configuration.

Use `supports`, `is consistent with`, and `within this protocol` where the
evidence is comparative rather than causal. Do not call the ablations the
paper's central novelty.

### Resource and Storage Implications

Explain that the active-budget sweep changes both active selected exemplars and
the candidate capacity tied to `rho=3`. State that the method trades additional
candidate storage and refresh computation for an active replay set of size
`M`. Do not describe the result as a free improvement at equal total memory.
The retrieval sweep changes the number of replay items drawn per update while
holding the active budget fixed.

### Limitations

Include all of the following:

- one dataset, one ten-task partition, one architecture, and three training
  seeds;
- no sensitivity to class order, task granularity, or architecture;
- limited active-budget and retrieval ranges;
- `rho=3` is the implementation default and was not independently swept;
- transient current-task candidate storage can exceed the boundary `rho M`
  bound before the first rebuild of a class;
- T1 does not isolate refresh from objective, readout, and storage differences;
- no matched refresh-only experiment uses the same objective, head, readout,
  active budget, and candidate multiplier for both refresh policies;
- paired analyses use only three matched seeds.

State the next experiment precisely: retain one common training objective, head,
readout, active budget, candidate multiplier, data order, and seed set, then
swap only arrival-time prefix truncation for refresh-all-class candidate
reselection. This is the experiment needed to make a refresh-specific claim.
