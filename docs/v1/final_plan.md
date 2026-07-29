# Final Research Plan

## 0. Goal

Build a class-incremental learning method that can plausibly beat the current
from-scratch iCaRL-style baseline by combining:

1. a strong pre-trained backbone,
2. a held-out forgetting probe that is never replayed,
3. a fixed-total-memory replay policy that allocates scarce memory by measured
   class drift,
4. a post-hoc classifier calibration step.

The research claim is not "PID control is good". The claim is:

> A clean forgetting probe, when measured on held-out exemplars and coupled
> to a scarcity-aware replay policy, can improve class-incremental learning
> under fixed memory.

This is a narrower and more defensible direction than the previous
ghost-bank/PID framing, and it is aligned with the current PTM-based CIL
literature.

---

## 1. Why this direction

The earlier codebase explored many variants around replay, distillation, and
PID feedback. That line of work ran into two problems:

1. the probe was contaminated when it measured the same exemplars used for
   replay,
2. the "no replay" variants were confounded with frozen classifier rows.

The research question should therefore move one level up:

- first validate the signal,
- then use that signal to control replay,
- and do it on a strong representation backbone.

This matches the current literature:

- iCaRL established exemplar replay and NME classification as a strong early
  baseline for class-incremental learning [Rebuffi et al., 2017](https://arxiv.org/abs/1611.07725).
- Surveys emphasize that memory budget alignment is critical for fair CIL
  comparison [Zhou et al., 2023](https://arxiv.org/abs/2302.03648) and
  [Zhou et al., 2024](https://arxiv.org/abs/2401.16386).
- Pre-trained models changed the problem: a frozen PTM plus a simple prototype
  classifier can already be very strong [Zhou et al., 2023/2024](https://arxiv.org/abs/2303.07338),
  [Zhang et al., 2023](https://arxiv.org/abs/2303.05118).
- PTM-based CIL and replay-based CIL both still leave room for improved
  replay dynamics and gradient balancing [Zhou et al., 2023](https://arxiv.org/abs/2308.01698),
  [Yin et al., 2024](https://arxiv.org/html/2408.08084v1),
  [Zhou et al., 2024](https://arxiv.org/html/2405.15157v1).

---

## 2. Final Hypothesis

### Primary hypothesis

Let `P_c` be a held-out probe set for class `c` that never enters the replay
buffer. Let `s_c(t)` be the probe loss of class `c` after task `t`, measured
on the current model.

If the forgetting signal is real, then:

1. `s_c(t)` should correlate with future forgetting on class `c`,
2. a replay policy that allocates more capacity to classes with larger
   `s_c(t)` should outperform uniform replay,
3. this improvement should remain visible under a fixed total memory budget.

### Null hypothesis

If the signal is not useful, then:

1. the probe loss will not predict future forgetting,
2. probe-guided allocation will not improve over uniform replay,
3. any gain will vanish once memory is made scarce and evaluation is done on
   held-out test data.

This null is important. If it survives, we should write it as a negative
result rather than forcing a positive story.

---

## 3. Method Direction

### Name

Working name: **Probe-Guided Replay with Pre-trained Backbone**.

The method is intentionally not framed as PID. The previous PID control
framing introduced unnecessary instability and made the results harder to
interpret. The new method should be a bounded allocation policy driven by a
probe signal.

### Backbone

Use a **pre-trained visual backbone** rather than training from scratch.
The first implementation should stay close to the existing codebase but still
move into the PTM regime:

- primary option: ImageNet-pretrained ResNet-18 or ResNet-50,
- secondary option: CLIP/ViT or another vision-language backbone if the
  first branch is successful and compute allows.

This choice is supported by the PTM-CIL literature:

- PTM embeddings can be strong enough that a simple prototype classifier is
  competitive by itself [Zhou et al., 2023/2024](https://arxiv.org/abs/2303.07338).
- The key challenge becomes adaptivity, not raw feature learning
  [Zhou et al., 2024](https://arxiv.org/abs/2401.16386),
  [Zhang et al., 2023](https://arxiv.org/abs/2303.05118).

### Adaptation Strategy

Use one of the following, in order:

1. **Frozen backbone + prototype classifier** as the strongest sanity baseline.
2. **Partial fine-tuning** of the last residual stage, or a lightweight
   adapter/prompt module, if frozen features underfit the task stream.
3. **Classifier alignment / calibration** after each task, using exemplars
   from memory.

The adaptation path should be selected by the smallest model that gives
non-trivial incremental gains, not by architectural complexity.

### Replay Policy

Use a fixed total memory budget `M` and maintain a per-class memory quota
`m_c(t)` over seen classes `S_t`.

The policy should obey:

`sum_{c in S_t} m_c(t) = M`

with a non-zero floor for every class:

`m_c(t) >= m_min`

Replay allocation is computed from a mixture of:

- a uniform prior, to guarantee coverage,
- a probe-derived importance score, to emphasize drifting classes.

Example allocation form:

`w_c(t) = (1 - gamma) / |S_t| + gamma * softmax(beta * r_c(t))`

where:

- `r_c(t)` is the normalized forgetting score of class `c`,
- `gamma in [0,1]` controls how strongly the probe overrides uniform replay,
- `beta > 0` sharpens the ranking.

Then:

`m_c(t) = max(m_min, floor(M * w_c(t)))`

and the allocations are renormalized if needed to keep the sum fixed.

This is deliberately simpler than PID, because the question here is allocation
quality, not controller dynamics.

### Probe Signal

For each class `c`, reserve a fixed held-out probe set `P_c` before any
training begins. These images:

- are never stored in replay memory,
- are never used for herding,
- are never used in classifier calibration,
- are never used in SGD.

For task `t`, define the probe loss:

`s_c(t) = (1 / |P_c|) * sum_{x in P_c} CE(f_t(x), c)`

For PTM-based methods with cosine or prototype classifiers, the score can be
computed as:

`s_c(t) = (1 / |P_c|) * sum_{x in P_c} [1 - cos(phi_t(x), mu_c(t))]`

if the classifier is prototype-based, where `mu_c(t)` is the class prototype.

### Memory Update

At the end of each task:

1. update the probe score for all seen classes,
2. compute replay weights,
3. select exemplars for each class by herding or prototype closeness,
4. shrink or redistribute per-class memory to match the fixed total budget.

The implementation should support both:

- **herding in feature space**,
- **simple nearest-to-prototype selection**.

The first is closer to iCaRL.
The second is cheaper and may be enough in a PTM setting.

---

## 4. Loss Function

For task `t`, train on a mixture of:

- current-task samples,
- replayed samples from the memory bank,
- optionally, a distillation term against the previous snapshot.

A practical objective is:

`L_t = L_new + lambda_rep * L_replay + lambda_kd * L_distill`

where:

- `L_new` is CE on the current task classes,
- `L_replay` is CE or BCE on replayed old classes,
- `L_distill` is logit distillation from the previous model snapshot.

The exact loss should match the backbone choice:

- if using a SimpleCIL-like prototype baseline, use prototype updates plus
  a calibrated classifier,
- if using a replay-trained classifier, use CE/BCE with replay and a
  lightweight distillation term.

The important point is that the probe determines *which* classes are replayed
more often, not the entire learning objective.

---

## 5. Why This Is Better Than the Old PID Route

The older PID plan had three weaknesses:

1. the sensor was not clean,
2. the controller could starve classes,
3. the head/backbone interaction was too confounded.

The new plan removes those failure modes:

- held-out probe eliminates self-measurement,
- fixed-total memory creates real scarcity,
- uniform-floor allocation avoids starvation,
- PTM backbone makes the representation problem easier and more relevant to
  current CIL work.

This is consistent with recent findings:

- `SimpleCIL` can already be very strong when the backbone is pretrained
  [Zhou et al., 2023/2024](https://arxiv.org/abs/2303.07338).
- `SLCA` shows that careful adaptation plus classifier alignment can give
  large gains on PTM-based CIL [Zhang et al., 2023](https://arxiv.org/abs/2303.05118).
- `BDR` shows that controlling the destructive effect of current-task updates
  in replay-based CIL can improve reconstruction of old knowledge
  [Zhou et al., 2023](https://arxiv.org/abs/2308.01698).
- `WBR` and similar replay papers show that the balance between old and new
  samples is a first-order issue, not a minor detail
  [Yin et al., 2024](https://arxiv.org/html/2408.08084v1).

---

## 6. Experimental Protocol

### Primary benchmark

Use CIFAR-100 in class-incremental format:

- 10 tasks,
- 10 classes per task,
- standard class order at first, then a shuffled-order ablation later,
- fixed total memory `M = 2000`.

This matches the standard memory regime emphasized in CIL surveys and keeps
the comparison fair.

### Data splits

For each class:

- `P_c`: held-out probe images, reserved before training,
- `V_c`: validation images for tuning,
- `T_c`: training images used for SGD.

Recommended split on the 500 CIFAR-100 training images per class:

- 30 probe,
- 20 validation,
- 450 training.

### Evaluation

Report:

1. average incremental accuracy,
2. final average accuracy after the last task,
3. average forgetting,
4. backward transfer,
5. per-class accuracy,
6. compute cost and memory cost.

Use the official CIFAR-100 test split for final reporting.

### Seeds

At least 3 seeds for the final comparison table.
Do not publish single-seed results as the main claim.

### Main baselines

At minimum compare against:

- iCaRL [Rebuffi et al., 2017](https://arxiv.org/abs/1611.07725),
- a uniform replay baseline with the same memory budget,
- a strong PTM baseline such as SimpleCIL-style prototype classification
  [Zhou et al., 2023/2024](https://arxiv.org/abs/2303.07338),
- a PTM adaptation baseline such as SLCA
  [Zhang et al., 2023](https://arxiv.org/abs/2303.05118),
- one modern replay baseline such as BDR or an equivalent recent replay
  method [Zhou et al., 2023](https://arxiv.org/abs/2308.01698).

If compute allows, add a CLIP-based branch:

- [Class-Incremental Learning with Pre-Trained Vision-Language Models](https://arxiv.org/abs/2310.20348)

but this is optional for the first implementation pass.

---

## 7. Ablation Matrix

The ablation matrix must isolate the effect of each design choice:

1. **Backbone**
   - from-scratch ResNet
   - ImageNet-pretrained ResNet
   - PTM + lightweight adapter

2. **Probe**
   - no probe, uniform replay
   - replay-buffer probe only
   - held-out probe only
   - held-out probe + EMA smoothing

3. **Allocation**
   - uniform replay
   - linear probe weighting
   - softmax probe weighting
   - floor-clipped weighting

4. **Classifier**
   - plain linear head
   - prototype classifier
   - calibrated classifier after each task

5. **Memory regime**
   - fixed total memory
   - fixed per-class memory

The priority is to prove that the held-out probe is the useful part.

---

## 8. Success Criteria

The project is successful if all of the following hold:

1. the held-out probe correlates with later forgetting better than the replay
   buffer probe,
2. probe-guided allocation beats uniform replay under fixed memory,
3. the PTM-based version beats iCaRL and the strongest internal baseline,
4. the result is stable over at least 3 seeds,
5. the gains survive a shuffled class-order test.

If the method only wins on the default class order or only on one seed, it is
not ready for a paper claim.

---

## 9. Implementation Order

### Phase 1: scientific cleanup

1. freeze the benchmark protocol,
2. define held-out probe splits,
3. define fixed-total memory and shrinking per-class allocation,
4. reproduce a clean uniform replay baseline.

### Phase 2: PTM baseline

1. add an ImageNet-pretrained backbone,
2. implement frozen-prototype and simple adaptation baselines,
3. verify that the PTM baseline is stronger than the current from-scratch
   ResNet stack.

### Phase 3: probe-guided replay

1. compute held-out probe scores,
2. map scores to memory allocation,
3. train with allocation-aware replay,
4. evaluate against uniform replay.

### Phase 4: calibration and polishing

1. add post-hoc classifier calibration,
2. tune one small set of hyperparameters on validation only,
3. run 3 seeds,
4. write the paper-grade result table.

---

## 10. Risks

### Risk 1: PTM baseline is already too strong

If the frozen PTM baseline is already competitive with everything else, the
research contribution shifts from "beat iCaRL" to "improve PTM-based CIL with
probe-guided replay". That is still a valid paper.

### Risk 2: the probe does not predict forgetting

If the held-out probe fails, we should stop trying to force the adaptive
allocation idea. At that point the right paper is a negative result or a
methodological note about why the signal is insufficient.

### Risk 3: probe-guided replay improves but not enough

If the gain is modest, we can still salvage the work by reporting:

- the probe validity analysis,
- the memory-scarcity effect,
- the best-performing hybrid replay rule.

This is still useful if the design is clean and reproducible.

---

## 11. Final Decision

Proceed with:

**PTM-based class-incremental learning + held-out probe + fixed-total-memory
probe-guided replay + calibration**

and treat the old PID framing as discarded infrastructure, not as the main
research story.

This gives the best chance of either:

- a stronger result than iCaRL, or
- a rigorous negative result that is actually publishable.

---

## References

- Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning"
  (2017): https://arxiv.org/abs/1611.07725
- Zhou et al., "Deep Class-Incremental Learning: A Survey" (2023):
  https://arxiv.org/abs/2302.03648
- Zhou et al., "Continual Learning with Pre-Trained Models: A Survey" (2024):
  https://arxiv.org/abs/2401.16386
- Zhou et al., "Revisiting Class-Incremental Learning with Pre-Trained Models:
  Generalizability and Adaptivity are All You Need" (2023/2024):
  https://arxiv.org/abs/2303.07338
- Zhang et al., "SLCA: Slow Learner with Classifier Alignment for Continual
  Learning on a Pre-trained Model" (2023):
  https://arxiv.org/abs/2303.05118
- Zhou et al., "Balanced Destruction-Reconstruction Dynamics for
  Memory-replay Class Incremental Learning" (2023):
  https://arxiv.org/abs/2308.01698
- Yin et al., "An Efficient Replay for Class-Incremental Learning with
  Pre-trained Models" (2024):
  https://arxiv.org/html/2408.08084v1
- Zhou et al., "Rethinking Class-Incremental Learning from a Dynamic
  Imbalanced Learning Perspective" (2024):
  https://arxiv.org/html/2405.15157v1
- Liu et al., "Class Incremental Learning with Pre-trained Vision-Language
  Models" (2023): https://arxiv.org/abs/2310.20348
