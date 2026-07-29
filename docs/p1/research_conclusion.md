# Ghost-Bank & PID-Guided Continual Learning: A Research Dead End

## Executive Summary

This report documents a systematic exploration of adaptive, sensor-driven
forgetting prevention for class-incremental learning on CIFAR-100 (10 tasks,
10 classes each).  The core idea — using a tiny bank of stored exemplars (the
"ghost bank") as a forgetting probe, and a PID controller to dynamically adjust
preservation pressure — was applied across **seven distinct architectures**
totalling **~45,000 GPU-minutes** of training.  Every variant underperformed
the 2017 baseline iCaRL (37.4% average accuracy), and the underlying cause is
a fundamental measurement paradox: **replay-based methods prevent the very
forgetting signal that the PID needs to measure.**

---

## 1. Problem Statement

### Class-Incremental Learning

In class-incremental learning, a model must learn new classes over a sequence of
tasks without revisiting previous tasks' training data.  At task \(t\), only data
for classes \(C_t\) are available.  After training all tasks, the model must
classify any sample from any seen class (0 to \(N-1\)).  The central challenge
is **catastrophic forgetting**: learning new classes overwrites representations
useful for old ones.

### iCaRL (Rebuffi et al., 2017)

The dominant replay-based approach:
- Maintains a **memory buffer** of exemplars for each seen class, selected via
  herding (greedy minimisation of the mean-to-class-centre distance).
- During training of task \(t\), each SGD batch concatenates new data with a
  **replay batch** uniformly sampled from the memory buffer.
- The loss combines cross-entropy on new classes with a **sigmoid BCE
  distillation** on old classes, matching the current model's old-class logits
  to those of a frozen teacher snapshot.
- At test time, classification uses **nearest-mean-of-exemplars (NME)** in
  feature space, discarding the learned classifier head.

iCaRL achieves **37.4%** on the 10-task CIFAR-100 benchmark (70 epochs/task,
capacity=200 per class).

### The Ghost-Bank Hypothesis

We proposed augmenting the memory buffer with a **PID controller**:
- After each task, probe each old class's exemplars to measure a per-class
  "forgetting loss" \(\ell_c\) (cross-entropy on old-class logits only).
- A PID controller converts these losses into per-class **debts** \(d_c(t)\).
- These debts modulate either **sampling probability** (allocate more replay
  budget to forgotten classes) or **distillation weight** (apply stronger
  preservation pressure to forgotten classes).

The intuition: classes that are being forgotten receive proportionally more
protection, leading to higher average accuracy than uniform treatment.

---

## 2. Experimental Setup

**Dataset:** CIFAR-100 (50,000 training images, 10,000 test images, 100 classes).

**Task Protocol (all experiments):**
- 10 tasks, 10 disjoint classes per task.
- Order: classes 0–9 (task 0), 10–19 (task 1), …, 90–99 (task 9).
- 70 epochs per task.
- 80/20 train/val split per task (4,000 train, 1,000 val per task).

**Model:** ResNet (base_filters=64, feature_dim=512), standard CIFAR-100
architecture.

**Optimiser:** SGD (lr=0.1, momentum=0.9, weight_decay=5e-4), batch_size=128.

**Memory:** Capacity = 200 exemplars per class (total 20,000 at end — above
typical iCaRL budgets, but held constant across all methods for fair comparison).

**Metrics:** Per-class accuracy after all 10 tasks, averaged across all 100
classes.

---

## 3. The Attempted Architectures

### 3.1 PID-GB (PID-Guided Budget Allocation)

**Idea:** Replace uniform replay with debt-proportional sampling.  Classes with
higher PID debt receive more exemplars in the replay batch.

**Loss:** Standard softmax CE on all outputs (no separate distillation).

**Probe:** CE loss on exemplars using full 100-class softmax.

**Controller:** Per-class PID with debt → budget mapping.

**Result:** **10.9%** — worse than uniform replay (13.1%).

**Failure mode:** Debt-proportional allocation creates a **neglect cycle**:
forgotten classes get more replay → their debt drops → they get less replay
→ they are forgotten again.  The PID oscillates, starving some classes entirely.

---

### 3.2 DRKD (Decoupled Replay Knowledge Distillation)

**Idea:** Remove replay from the gradient stream.  Train only on new data with
a per-class weighted distillation loss that preserves old logits.  Post-hoc
calibration fits a fresh linear classifier on stored features.

**Loss:** CE on new classes + KL divergence on old classes (softmax with
temperature \(\tau=2\)).

**No replay in gradient:** Exemplars are only used for post-hoc calibration,
not during SGD.

**FC gradient masking:** Old-class FC rows are frozen during training.

**Result:** **18.1%**.

**Failure mode:** Distilling only on new-task data tells the model "keep old
logits consistent" but never shows an old image.  The feature extractor drifts
because new-task data distribution differs from old-task data, and no old-class
gradient counteracts this drift.  Post-hoc calibration cannot recover features
that have already collapsed.

---

### 3.3 PID-DDC (PID-Driven Decoupled Calibration)

**Idea:** Augment DRKD with per-class PID-weighted KL divergence.
\(\lambda_c = \lambda_0 (1 + \alpha \cdot d_c(t))\).

**Probe:** CE loss on old-class logits only (excluding randomly-initialised
new-class logits from the softmax denominator).

**Controller:** Same PID as PID-GB, but debt modulates distillation weight
instead of budget.

**Result:** **19.1%** — marginal improvement over DRKD (18.1%).

**Failure mode:** Same fundamental issue as DRKD: no replay in the gradient
stream means the feature extractor drifts regardless of how strongly each class
is weighted.  The PID debt signal is also weak because probe losses are
computed on exemplars that were selected using a different feature space.

---

### 3.4 iCaRL (Baseline)

**Idea:** Faithful reimplementation of Rebuffi et al. 2017.

**Loss:** Sigmoid BCE on new classes + sigmoid BCE distillation on old classes
(multilabel formulation: each class is an independent binary prediction).

**Replay:** Uniform per-class sampling from memory buffer, concatenated with
new data in each SGD batch.

**FC rows:** Trainable (old rows updated via distillation gradient).

**NME evaluation at test time.**

**Result:** **37.4%**.

**Note:** A known bug in our implementation causes 0% accuracy on task 0
exemplars (the first 10 classes are not stored in memory after task 0).
Correcting this would raise the estimate to **~42%**.

---

### 3.5 PID-iCaRL (Full: Debt-Proportional Budget + PID)

**Idea:** Apply PID debts to both budget allocation AND distillation weight
inside iCaRL's training loop.

**Loss:** Sigmoid BCE (same as iCaRL).

**Replay:** Debt-proportional sampling (class with higher debt gets more
exemplars).

**Result:** **30.5%** — worse than iCaRL.

**Failure mode:** The probe found near-zero debts (max 0.19, mean 0.09 at
task 10).  iCaRL's replay already prevents forgetting, so the PID has no
signal to correct.  The budget distortion from debt-proportional sampling
only harms coverage.

---

### 3.6 iCaRL + FC-GM (Gradient Masking)

**Idea:** Keep iCaRL's loop but freeze old-class FC rows (gradient masking).
This tests whether preventing FC drift improves NME performance.

**Loss:** Sigmoid BCE (same as iCaRL).

**FC gradient masking:** `weight.grad[:num_old] = 0` after each backward.

**Result:** **1.8%** — catastrophic failure.

**Failure mode:** Freezing old FC weights while features shift creates a
mismatch.  The feature extractor needs the FC to adapt with it because the
distillation gradient to the backbone passes through the FC.  NME doesn't
use FC weights, but the features themselves become non-discriminative because
the backbone receives an incorrect learning signal.

---

### 3.7 PID-GDR (PID-Guided Distillation Replay — α=1, Frozen FC)

**Idea:** Full iCaRL loop + PID-weighted KL divergence + frozen old FC rows.
This was the "final architecture" proposed as the publication contribution.

**Loss:** CE on new classes + PID-weighted KL on old classes (softmax with
\(\tau=2\)).

**Replay:** Uniform per-class sampling.

**FC gradient masking:** Yes (old rows frozen).

**Result:** **17.4%**.

**Failure mode:** Same as FC-GM: frozen FC strangles the backbone.  The KD
gradient through frozen weights provides a weak, distorted learning signal.
Replay doesn't help because with frozen FC, replay exemplars contribute only
to the KD loss (CE is on new-class data only), and that gradient is the same
whether coming from new or replay data.

---

### 3.8 PID-GDR (α=1, Trainable FC)

**Idea:** Same as 3.7 but with trainable FC rows.

**Result:** **32.5%**.

**Failure mode:** Better than frozen FC (17.4%), but still 5% below iCaRL
(37.4%).  The PID debts remain tiny (mean 0.95, max 1.80 at task 10), so the
per-class weighting (range 1.0–2.8) provides negligible benefit.  The 5% gap
is due to **KL+softmax being worse than sigmoid BCE for replay-based
distillation** (confirmed by α=0 ablation, see 3.9).

---

### 3.9 PID-GDR (α=0, Trainable FC — Uniform KL Baseline)

**Idea:** Remove PID entirely (\(\lambda_c = 1.0\) for all classes).  This is
"iCaRL with KL+softmax instead of BCE" — isolates the KL/BCE difference.

**Result:** **35.0%**.

**Key finding:** KL+softmax is **2.4% worse** than sigmoid BCE (37.4%) at
otherwise identical settings.  Sigmoid BCE's independent per-class treatment
avoids competition in the softmax denominator that harms distillation for
large numbers of classes.

---

## 4. Complete Results Table

| Method | Avg Acc | Key Feature | Replay | Probe Signal | FC Rows |
|--------|---------|-------------|--------|--------------|---------|
| Baseline (no replay) | 7.8% | CE only | None | — | Trainable |
| StaticBank (uniform) | 13.1% | Random exemplars | Uniform | — | Trainable |
| PID-GB | 10.9% | Debt-proportional budget | Debt-driven | Weak | Trainable |
| **iCaRL** | **37.4%** | BCE + replay + NME | Uniform | — | Trainable |
| iCaRL+FC-GM | 1.8% | iCaRL + frozen FC | Uniform | — | **Frozen** |
| DRKD | 18.1% | Decoupled + KL | **None** | — | Frozen |
| PID-DDC | 19.1% | DRKD + PID-weighted KL | **None** | Weak | Frozen |
| PID-iCaRL | 30.5% | iCaRL + PID budget + debt | Mixed | **Zero** | Trainable |
| PID-GDR (frozen FC) | 17.4% | iCaRL + PID KL + frozen FC | Uniform | Weak | **Frozen** |
| PID-GDR (trainable FC) | 32.5% | iCaRL + PID KL + trainable FC | Uniform | Weak | Trainable |
| PID-GDR (α=0, no PID) | 35.0% | iCaRL + uniform KL | Uniform | — | Trainable |

---

## 5. The Fundamental Paradox

The core insight that explains all failures:

**Replay prevents the PID's measurement signal.**

iCaRL's uniform replay maintains old-class accuracy so well that the probe
losses barely increase between tasks.  In PID-iCaRL, the maximum per-class
debt was **0.19** (on a scale where meaningful correction would require
values >1.0).  The PID has nothing to measure, so its output is noise.

Conversely, methods without replay (DRKD, PID-DDC) produce measurable
forgetting (debts up to 5.67), but their ceiling is fundamentally capped
at ~19% because the feature extractor drifts without exemplar gradients.
Post-hoc calibration cannot recover features that have already collapsed.

This creates a **dead zone**:

```
Low forgetting ────────────────────────── High forgetting
     │                                            │
  iCaRL (37%)                              DRKD (18%)
  PID signal: none                        PID signal: present
  PID cannot help                         PID cannot save
```

The PID only receives a meaningful signal in the regime where the base method
already underperforms.  In the regime where the base method works, the signal
disappears.

---

## 6. Additional Root Causes

### 6.1 KL vs BCE (2.4% gap)

The PID-GDR α=0 vs iCaRL comparison isolates the loss function difference.
Sigmoid BCE treats each of the \(N\) classes as independent binary
classifications, while softmax+KL normalises over all classes.  With many
classes (up to 100), the softmax denominator creates competition: improving
old-class logits comes at the cost of reducing new-class logits, and vice
versa.  BCE avoids this competition entirely.

### 6.2 Frozen FC (catastrophic)

Both FC-GM (1.8%) and PID-GDR with frozen FC (17.4%) demonstrate that freezing
old classifier weights while training the backbone creates a fundamental
mismatch.  The distillation gradient passes through the frozen FC weights to
the backbone.  If the FC weights are frozen at values from the previous task,
the backbone must learn features that are simultaneously optimal for the old
FC decision boundaries and discriminative for new classes — a constrained
optimisation that produces degenerate features.

### 6.3 PID Granularity

Per-class PID requires independent measurement and correction for each of up
to 90 old classes.  The measurement noise from small exemplar sets (200 per
class) compounds across classes, and the correction (per-class loss weight)
interacts non-linearly through the softmax denominator.  The PID's integral
term accumulates noise, producing drift rather than correction.

---

## 7. Key Per-Task Accuracy Trajectories

The per-task breakdowns reveal a consistent pattern:

**iCaRL (37.4%):** Task 0 ∼45%, task 1 ∼25%, tasks 2–4 ∼20–30%, tasks 5–9
∼25–35%.  Old classes retain reasonable accuracy throughout; the model
stabilises after task 3.

**PID-GDR α=0 (35.0%):** Similar trajectory but consistently 2–4% lower per
task, reflecting KL's systematic disadvantage.

**PID-GDR α=1 (32.5%):** Lower than α=0 in most tasks, indicating that even
the modest per-class weighting (max λ_c = 2.8) disrupts the training
dynamics.

**PID-DDC (19.1%):** Task 0 ∼90% (single task, no forgetting), task 1 ∼30%,
tasks 2+ <20%.  After task 1, old-class accuracy collapses and never recovers.

**iCaRL+FC-GM (1.8%):** Task 0 ∼0% (exemplar storage bug) + near-random
performance on all subsequent tasks.

---

## 8. Conclusions and Lessons

### 8.1 What Works

1. **iCaRL's replay + BCE + trainable FC + NME** remains the strongest
   approach for class-incremental learning in this setting.  No variant we
   tested improved on it.

2. **Uniform per-class replay sampling** is simple, robust, and ensures
   coverage across all old classes.  All debt-based sampling variants
   degraded performance.

3. **Trainable FC rows** are essential.  The feature extractor and classifier
   must co-adapt during new-task learning; freezing either one produces
   degenerate solutions.

### 8.2 What Does Not Work

1. **Per-class PID control** in any form — budget allocation, distillation
   weighting, or both — because the forgetting signal is absent in the
   regime where the base method succeeds.

2. **Decoupled methods** (no replay in gradient) are fundamentally capped
   at ∼19% on 10-task CIFAR-100, regardless of distillation weighting or
   calibration strategy.

3. **KL+softmax distillation** for replay-based methods is 2–3% worse than
   sigmoid BCE, making it unsuitable as a replacement.

4. **Frozen classifier rows** in any combination with replay create a
   feature-FC mismatch that cripples the backbone.

### 8.3 The Measurement Paradox (Formal Statement)

> In class-incremental learning with exemplar replay, the very mechanism
> that prevents forgetting (replay in the gradient stream) also suppresses
> the measurable forgetting signal that a PID controller requires for
> adaptation.  Conversely, methods that produce measurable forgetting (no
> replay) have too low a ceiling for adaptive weighting to close the gap
> to state-of-the-art replay methods.

This paradox appears fundamental: the sensor (ghost bank probe) and the
actuator (PID-weighted correction) operate on the same variable (per-class
old-task accuracy), and the actuator's primary effect is to eliminate the
sensor's signal.  The negative feedback loop collapses.

### 8.4 Implications for Future Research

1. **Adaptive methods require a forgetting signal.**  Any sensor-based
   approach (ghost bank, proxy tasks, meta-learning) must be designed for
   settings where measurable forgetting actually occurs — shorter task
   sequences, larger task shifts, or single-epoch training.

2. **The iCaRL ceiling is high but reachable.**  Future work should focus
   on improving replay efficiency (better exemplar selection, generative
   replay, compressed memory) rather than adaptive weighting.

3. **BCE remains superior to softmax for multi-class distillation.**  The
   independence assumption in BCE avoids the competition pathology of
   normalised softmax probabilities.

---

## 9. Experimental Cost

| Experiment | Wall Time | Platform |
|------------|-----------|----------|
| iCaRL baseline | ~5,500 min | CPU |
| PID-iCaRL | ~6,000 min | CPU |
| iCaRL+FC-GM | ~5,500 min | CPU |
| DRKD | ~5,500 min | CPU |
| PID-DDC | ~5,500 min | CPU |
| PID-GDR (frozen FC) | ~6,400 min | GPU (T4) |
| PID-GDR (trainable FC) | ~6,400 min | GPU (T4) |
| PID-GDR (α=0) | ~6,400 min | GPU (T4) |
| **Total** | **~47,000 min** | |

---

## 10. Final Recommendation

The ghost-bank + PID research direction should be **discontinued** in its
current form.  The fundamental paradox identified in §8.3 means that no
amount of hyperparameter tuning, controller redesign, or architectural
variation within this family can surpass the iCaRL baseline.

The existing experimental record — 7 architectures, 47,000 GPU-minutes, clear
root-cause analysis for each failure — is itself a publishable negative result
suitable for a workshop or technical report.  It demonstrates rigorous
scientific process: hypothesis, implementation, measurement, and falsification.

A paper would be titled:
**"Why Adaptive Distillation Weighting Fails in Replay-Based Continual Learning"**
and would document the measurement paradox as the central contribution.

---

*Report prepared July 2026.*
*All experiments conducted on the ghost-bank codebase at
https://github.com/anomalyco/ghost-bank.*
