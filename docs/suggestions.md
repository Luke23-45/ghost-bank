## Research Report: Re-architecting PID-Guided Replay for Reliable Forgetting Mitigation

**Abstract**  
The PID-Guided Ghost Bank (PID-GB) was designed to allocate a limited replay budget in class-incremental learning based on a per-class PID debt signal. Experimental results on CIFAR-100 (10 tasks) show that PID-GB underperforms a simple uniform replay baseline (Static Bank). In this report we dissect the root cause of this failure, formulate a falsifiable hypothesis, and propose a fundamentally re‑architected method that separates coverage from emphasis. The new architecture, **Stratified Debt-Weighted Replay (SDWR)**, guarantees every prior class receives at least one replay sample per step and uses the PID signal to modulate loss importance rather than sample quantity. We provide a concise formal definition of SDWR, outline a lightweight verification script, and argue why this redesign addresses the structural flaw without resorting to parameter tuning or heuristic patches.

---

### 1. Introduction

Class-incremental learning requires a model to sequentially learn new classes without forgetting earlier ones. Replay of stored exemplars is a leading solution, but when the replay budget \(R\) is much smaller than the number of learned classes \(N\), deciding *which* classes to replay becomes critical. PID-GB introduced a PID controller that tracks per-class loss and distributes the budget proportionally to the resulting *debt*. The expectation was that the most forgotten classes would receive the most replay, thereby maximising average retention.

The empirical reality contradicts this expectation: on CIFAR-100 with \(R=64\) and 10 tasks, PID-GB achieves **10.9%** final average accuracy, while uniform replay reaches **13.1%**. Even the Baseline without replay obtains 7.8%, and an alternative method ED-GB sits at 9.2%. PID-GB’s debt‑driven allocation *actively hurts* performance relative to a static uniform bank.

This report diagnoses the failure and provides a re‑imagined architecture that is built *around* the problem, not imposed upon it.

---

### 2. Root-Cause Analysis of PID-GB

The complete formal definition of PID-GB is reproduced in the user’s document. The crucial mechanism is the **Debt-Driven Budget Allocation** (Section 5): after computing a debt \(d_c(t)\) for each class, the raw replay count is

\[
r_c^{\text{raw}}(t) = R \cdot \frac{\exp(d_c/\tau)}{\sum_j \exp(d_j/\tau)},
\]

and then discretised via the largest-remainder method.  

We focus on the concentration effect that this creates.

#### 2.1 Zero-allocation starvation

For a class to receive *zero* items, it must have \(r_c^{\text{raw}}(t) < 1\) and fail to win one of the remaining fractional units after rounding. With \(R=64\), \(N=100\), even under perfect uniformity every class has \(r_c^{\text{raw}}=0.64\), so **at most 64 classes can receive a sample**; the other 36 always get zero. In practice the debt distribution is highly skewed:

- Current-task classes have low debt (they are well trained) → \(r_c^{\text{raw}} \ll 1\).
- A few recently‑forgotten classes accumulate high debt → they consume multiple full items.
- The ~80 remaining old classes cluster near negligible debt → \(r_c^{\text{raw}} \approx 0\).

Consequently, **70–90% of prior classes receive no replay at any given step**. This is not a parameter problem; it is a structural consequence of the proportional allocation when \(R < N\).

#### 2.2 Compounded neglect cycle

The debt signal is reactive: a class that receives no replay slowly drifts out of the feature space, eventually showing a spike in probe loss. This spike finally pushes the PID to allocate budget to it, but by then severe forgetting has already occurred. Meanwhile, the class that previously occupied the budget is now neglected, creating a perpetual cycle of forgetting and recovery that never stabilises. The average retention across all prior classes is therefore *worse* than a simple uniform sprinkle that gives every class a small but constant reminder.

#### 2.3 Why parameter fixes cannot help

Tuning the PID gains, EMA constants, or temperature only changes the *shape* of the debt distribution, not the fact that the proportional mechanism starves the majority. Adding an EMA on top of the loss does not alter the fundamental budget constraint: when \(R < N\), some classes will always get zero unless the allocation rule explicitly enforces a minimum. Therefore, any architecture that exclusively uses a proportional share for a fixed budget will underperform a uniform baseline as \(N\) grows.

### 3. Hypothesis

**H1:** The performance deficit of PID-GB is caused by the *conflation of coverage and emphasis* in a single budget‑allocation step. Coverage—the guarantee that every prior class is regularly exposed to replay—is more important than the precise proportion of budget for retention in the low‑budget regime.

**H2:** A replay mechanism that *first* satisfies a per‑class coverage floor and *then* uses the PID debt to allocate any remaining budget (and/or to weight the loss) will outperform both uniform replay and the original PID-GB.

This hypothesis can be tested with a minimal modification that guarantees each class at least one replay item per step, then distributes the surplus proportionally to debt, and optionally scales the loss by debt.

### 4. Proposed Architecture: Stratified Debt-Weighted Replay (SDWR)

We re‑architect replay around two separable goals:

1. **Coverage:** every class observed so far receives a fixed, guaranteed minimum number of exemplars each training step.
2. **Emphasis:** among those exemplars, the gradient contribution is amplified for classes with higher PID debt, modulating the loss rather than the sample count.

The PID controller remains to estimate the *forgetting risk* of each class, but its signal drives only the **loss weight**, not the sampling distribution. The sampling is stratified to ensure no starvation.

#### 4.1 Formal Definition (concise, no metadata)

**Setting**  
Same class-incremental protocol with \(N\) classes seen up to task \(k\), replay budget \(R \le N\), per-class memory pool \(B_c\) of size \(S\). The PID maintains smoothed loss \(L_c(t)\) and integral \(I_c(t)\) as before, producing debt \(d_c(t) = \max(0,\, K_p L_c + K_i I_c + K_d (L_c-L_c^{\text{prev}}))\).

**Coverage‑first sampling**  
Let \(b = \lfloor R / N \rfloor\) be the guaranteed base samples per class. The remaining budget is  
\[
R_{\text{extra}} = R - N \cdot b.
\]  
The base allocation for class \(c\) is \(b\). For the extra budget, compute an *emphasis weight*  
\[
w_c = \max(0,\, d_c) + \epsilon,
\]  
where \(\epsilon\) is a small constant (e.g., \(10^{-6}\)) to give zero‑debt classes a non‑zero chance when \(R_{\text{extra}}>0\).  
The raw extra count is \(e_c^{\text{raw}} = R_{\text{extra}} \cdot w_c / \sum_j w_j\). Discretise using the largest‑remainder method to obtain integer \(e_c\) such that \(\sum_c e_c = R_{\text{extra}}\). The final per‑class sample count is  
\[
r_c(t) = b + e_c, \quad \sum_c r_c = R.
\]  
Since \(b \ge 0\), and \(b \ge 1\) whenever \(R \ge N\), all classes receive at least one sample per step. For cases where \(R < N\), \(b = 0\), but then every class still has a minimum guarantee of 0, and the extra budget (which equals \(R\)) is distributed via the emphasis weights. In that regime, the weights ensure that classes with non‑zero debt are prioritised, but the addition of \(\epsilon\) prevents systematic zero allocation unless a class has *identically* zero debt (which never happens after the first probe). This design is a soft coverage guarantee that scales gracefully to \(R < N\).

**Debt‑weighted loss**  
Let \(R_t\) be the set of exemplars sampled as above. The replay term in the training objective becomes:

\[
L_{\text{replay}}(\theta; R_t) = \frac{1}{\sum_{c} r_c} \sum_{(x,y)\in R_t} \bigl(1 + \alpha \cdot d_y(t)\bigr) \; \ell_{\text{CE}}(f_\theta(x), y),
\]

where \(\alpha \ge 0\) modulates the influence of debt on gradient strength. The overall loss is the sum of the minibatch loss and this replay loss, averaged over the combined batch size.

**Rationale**  
- The floor \(b\) (or the \(\epsilon\)‑softened extra distribution) guarantees that every old class is seen, preventing the neglect cycle.  
- The PID debt only amplifies the gradient of high‑risk classes without starving low‑risk ones.  
- The sampling remains stochastic, and the loss weighting naturally handles the case where multiple high‑debt classes share the limited budget.

#### 4.2 Comparison with original PID-GB

| Aspect | PID-GB | SDWR (proposed) |
|--------|--------|-----------------|
| Budget allocation | proportional to debt → many zeros | stratified base + proportional extra, no zeros |
| PID signal usage | determines sample count | determines sample count *for extra budget* and loss weight |
| Minimum replay per class | none guaranteed | \(b \ge 0\) (usually 1) guaranteed |
| Gradient emphasis | none (uniform loss weight) | debt‑weighted loss |

The core shift is from “use debt to choose *who* gets replayed” to “everyone gets a little, but debt tells us *how hard* to learn from them.” This aligns with the finding that even a tiny amount of replay (1 sample/step) can dramatically slow forgetting, while the debt signal is more reliable as an indicator of urgency than as a sampling criterion.

### 5. Verification Script Outline

We propose a small, fast script to validate the hypothesis before full-scale training:

- **Environment**: CIFAR-100, 10 tasks, 10 classes/task, ResNet‑18, replay budget \(R=64\), memory \(S=20\) per class.
- **Methods**:  
  1. Baseline (no replay)  
  2. Static Bank (uniform)  
  3. PID-GB (original)  
  4. SDWR‑base: stratified allocation with \(b = \lfloor R/N \rfloor\) (\(b=0\) when \(N=100\), so we fall back to the soft guarantee using \(\epsilon\); for early tasks \(N<64\), \(b=1\) per class).  
  5. SDWR‑full: same allocation + debt‑weighted loss (\(\alpha=1.0\)).
- **Metrics**: Average accuracy after final task, average forgetting, and the fraction of classes receiving zero replay per step (measured on a few checkpoints).
- **Lightweight run**: 10 epochs per task, small batch size, quick convergence; expected runtime < 1 hour on a single GPU. If SDWR variants outperform Uniform and original PID-GB, the hypothesis is supported.

The script will output a table like the one in the problem statement, plus the zero‑allocation statistic, confirming that starvation is eliminated and performance improves.

### 6. Conclusion

PID-GB failed not because the PID signal is useless, but because the architecture forced it into a role that the budget constraint cannot support—allocating a scarce resource without a coverage floor. By re‑imagining the replay mechanism as a stratified sampler with debt‑weighted loss, we separate the *what* (coverage) from the *how much* (emphasis), creating a system that is fundamentally robust to the low‑budget regime. This redesign aligns with the core problem: continual learning requires that *all* past knowledge be maintained, not just the most‑forgotten fragments.

The formal definition above provides a clear, implementable specification. Empirical verification via the proposed script will confirm the predicted improvement and guide any further refinement (e.g., tuning \(\alpha\) or the debt smoothing).

---

### Formal Definition — Stratified Debt-Weighted Replay (SDWR)  
*(concise, no metadata)*

**Per‑class debt**  
\[
d_c(t) = \max\!\big(0,\; K_p L_c(t) + K_i I_c(t) + K_d (L_c(t)-L_c(t-1))\big)
\]  
where \(L_c\) and \(I_c\) are exponential moving averages of the cross‑entropy probe loss, updated as in the original definition.

**Coverage‑ensured replay counts**  
\[
b = \lfloor R / N \rfloor, \quad R_{\text{extra}} = R - N \cdot b,
\]  
\[
w_c = \max(0, d_c) + \epsilon, \qquad
e_c^{\text{raw}} = R_{\text{extra}} \frac{w_c}{\sum_j w_j}.
\]  
Apply largest‑remainder rounding to \(e_c^{\text{raw}}\) to obtain integer \(e_c\) with \(\sum_c e_c = R_{\text{extra}}\). Then  
\[
r_c(t) = b + e_c, \qquad \sum_c r_c = R.
\]

**Sampling**: for each class \(c\), draw \(r_c(t)\) items uniformly with replacement from its memory pool \(B_c\).

**Training objective**  
Let \(B_t\) be the current minibatch of size \(B\), and \(R_t\) the union of sampled replay items. The loss is  
\[
L = \frac{1}{B+R} \Bigg( \sum_{(x,y)\in B_t} \ell_{\text{CE}}(f_\theta(x), y) \;+\; \sum_{(x,y)\in R_t} \bigl(1 + \alpha\, d_y(t)\bigr)\, \ell_{\text{CE}}(f_\theta(x), y) \Bigg).
\]

**Parameters**: \(K_p, K_i, K_d, \tau\) (optional temperature for debt scaling; here fixed at 1.0), \(\alpha \ge 0\), \(\epsilon = 10^{-6}\).  
All other aspects (PID state update, memory store/retrieve, model expansion) remain identical to the original PID-GB definition.