# Plan: `manuscript/main.tex`

## Objective

Make the front matter identify the method and the research question. The title
must describe the proposed replay mechanism, not a conclusion about whether the
work is incremental or a component-analysis exercise.

## Title Patch

Replace:

`Disentangling Exemplar Replay in Class-Incremental Learning`

with:

`Uniform Herding: Exemplar Replay with Representation Refresh`

This title names the method and its distinguishing mechanism. It does not
claim that herding itself is new, and it does not imply that `M` is the entire
storage footprint.

## Abstract Patch

Write one compact paragraph in this order:

1. State the replay problem: prior examples must be represented under a
   bounded active replay budget while the feature representation changes.
2. Introduce Uniform Herding as a method that allocates the active set across
   observed classes and refreshes each class's selected exemplars in the
   current representation using a bounded candidate pool.
3. State the evaluation: CIFAR-100, ten class-incremental tasks, ResNet-18,
   active budget `M=2,000`, retrieval budget `b=64`, and three seeds.
4. Report the main values exactly: Uniform Herding has `44.00 +/- 0.51`
   average accuracy and `17.22 +/- 0.43` forgetting; iCaRL has
   `42.33 +/- 1.20` and `24.87 +/- 1.11`.
5. Summarize the supporting evidence without turning it into the headline:
   NME is substantially better than head-logit evaluation in this protocol,
   herding is better than the random-selection variant, KD mainly changes
   forgetting, and the active-budget sweep has a larger effect than the tested
   retrieval sweep.
6. End with the scope: these are results for the evaluated protocol, not a
   universal ranking or a refresh-only causal test.

The abstract should say `active budget`, not simply `memory`, because Uniform
Herding also retains candidate examples. It need not list every T2/T3 number.
Do not include a novelty apology or a generic statement that the work is not a
new algorithm.

## Integration Checks

- Preserve the current section input order for the content rewrite.
- Keep the bibliography after the main content and the appendix after the
  bibliography in the generic source until the venue-format patch is selected.
- Do not modify packages, margins, or document class in the content patch.
- Confirm every abstract number against the regenerated T1--T3 tables.
- Add no keywords unless the selected venue requires them.
