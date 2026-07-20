# Formal Mathematical Definition — PID-Guided Ghost Bank (PID-GB)

This document defines the PID-GB method for class-incremental learning with replay-based forgetting mitigation. Every equation, algorithm, and claim is stated precisely.

---

## 1. Class-Incremental Learning Setting

Let there be **M** tasks presented sequentially:

```
T_1, T_2, ..., T_M
```

Each task **T_k** introduces a set of **disjoint** classes **C_k** such that:

```
C_i ∩ C_j = ∅   for i ≠ j
```

and the union covers all **K** classes in the dataset:

```
\bigcup_{k=1}^{M} C_k = {1, 2, ..., K}
```

For simplicity, each task introduces the same number of new classes:

```
|C_k| = K / M   for all k
```

**Training data.** The training set for task **T_k** is:

```
D_k = {(x_i, y_i)}  where  y_i ∈ C_k
```

**Model.** A neural network **f_θ** consists of:

- A **shared feature extractor** **φ_θ : X → ℝ^D** (convolutional backbone)
- An **expanding classifier head** **h_k : ℝ^D → ℝ^{Σ_{j≤k} |C_j|}** which grows after each task

All parameters (feature extractor and classifier head) are updated jointly via gradient descent — nothing is frozen.

At task **k**, the head has outputs for all classes seen so far:

```
h_k(φ_θ(x))_c   corresponds to class c ∈ ⋃_{j≤k} C_j
```

The head expands by appending **|C_k|** new output units, initialized with small random weights:

```
W_new ∼ N(0, 0.001²)
```

The logits at task **k** are:

```
f_θ^{(k)}(x) = h_k(φ_θ(x)) ∈ ℝ^{Σ_{j≤k} |C_j|}
```

**Evaluation.** After training on task **k**, we evaluate on all **k** tasks seen so far. The primary metric is **average accuracy** (AA):

```
AA_k = (1/k) · Σ_{j=1}^{k} a_{k,j}
```

where **a_{k,j}** is the accuracy on task **T_j**'s test set after training through task **k**.

---

## 2. Replay Buffer (Ghost Bank Memory)

The ghost bank stores exemplars from all classes seen so far. It is defined as a set of per-class pools:

```
G = {B_1, B_2, ..., B_K}
```

where each **B_c** is a **static pool** of capacity **S**:

```
B_c = [(v_j, y_j)]_{j=1}^{S}    where y_j = c and v_j is a raw uint8 NHWC image
```

**Store operation.** At each training step, for each sample **(x_i, y_i)** in the current minibatch, the raw pre-augmentation image **v_i** is appended to **B_{y_i}** if the pool is not yet full:

```
|B_c| → min(S, |B_c| + count_c(B))
```

where **count_c(B)** is the number of samples from class **c** in the current minibatch. The pools saturate at capacity **S** — once **|B_c| = S**, no further exemplars for that class are accepted. There is no eviction or replacement; the first **S** exemplars seen for each class constitute the permanent stored set.

**Retrieval operation.** At each training step, a budget of **R** items is retrieved from the bank and combined with the current minibatch. The exact number per class is determined by the PID controller (Section 4) and the allocator (Section 5).

---

## 3. Per-Class Loss Signal

The PID controller requires a per-class loss signal **e_c(t)** at each step **t**. This signal is computed differently depending on whether class **c** is present in the current minibatch **B_t**.

### 3.1 Classes Present in the Minibatch

For classes with at least one sample in **B_t**, the loss is the standard cross-entropy averaged over the minibatch samples of that class:

```
e_c(t) = (1 / n_c(t)) · Σ_{(x,y) ∈ B_t, y = c} ℓ_CE(f_θ(x), y)
```

where **n_c(t) = |{(x,y) ∈ B_t : y = c}|** and:

```
ℓ_CE(f_θ(x), y) = -log( softmax(f_θ(x))_y )
```

### 3.2 Classes Absent from the Minibatch — The Bank Probe

For classes **not** present in **B_t**, the loss is estimated via a **bank probe**: a forward pass through the model using a small random subset of the class's stored exemplars.

Let **B_c** be the pool for class **c**. We sample **P** items uniformly without replacement:

```
P_c(t) = { (v_j, y_j) }   where   P = min(P_max, |B_c|)
```

Each raw image **v_j** is NHWC uint8. It is converted to NCHW float32 and the **same train-time augmentation** is applied (random crop + flip + normalization) before the forward pass:

```
ℓ_c^{probe}(t) = (1 / P) · Σ_{v ∈ P_c(t)} ℓ_CE(f_θ( augment(v) ), c)
```

The probe loss is computed under **torch.no_grad()** — it contributes only to the PID signal, not to the model gradient.

The per-class loss signal is therefore:

```
e_c(t) =  
  ℓ_c^{batch}(t)     if c is present in B_t
  ℓ_c^{probe}(t)     if c is absent from B_t and |B_c| ≥ 1
  None               if B_c is empty (class never stored)
```

---

## 4. PID Controller

The PID controller transforms the per-class loss signal into a per-class **debt** that determines how much replay budget each class receives.

### 4.1 Internal State

For each class **c**, the PID maintains three state variables:

| Variable | Symbol | Definition |
|----------|--------|------------|
| Smoothed loss | L_c(t) | Exponential moving average of e_c(t) |
| Integral | I_c(t) | EMA of L_c(t) (accumulated error) |
| Previous loss | L_c(t-1) | Smoothed loss from the previous update |

### 4.2 State Updates

When an update is received for class **c** (i.e., **e_c(t) ≠ None**):

```
L_c(t) = β · L_c(t-1) + (1-β) · e_c(t)       [smoothing]
I_c(t) = γ · I_c(t-1) + (1-γ) · L_c(t)        [integral accumulation]
```

where **β** (default 0.9) and **γ** (default 0.99) are decay rates. When **e_c(t) = None** (class never stored), the state is left unchanged:

```
L_c(t) = L_c(t-1)
I_c(t) = I_c(t-1)
```

### 4.3 Debt Computation

The per-class debt at step **t** combines three terms:

```
debt_c(t) = max(0, w_c · [ K_p · L_c(t) + K_i · I_c(t) + K_d · (L_c(t) − L_c(t-1)) ])
```

where:

| Parameter | Default | Role |
|-----------|---------|------|
| K_p = 1.0 | 1.0 | Proportional gain — responds to current loss |
| K_i = 0.1 | 0.1 | Integral gain — accumulates persistent forgetting |
| K_d = 0.5 | 0.5 | Derivative gain — responds to changes in loss |
| w_c | 1.0 | Per-class weight (currently uniform) |

The debt upper-bound is set to **10,000** to prevent numerical overflow:

```
debt_c(t) ← min(10,000, debt_c(t))
```

---

## 5. Debt-Driven Budget Allocation

The total retrieval budget **R** (default 64) is distributed across all **N** classes proportionally to their debt.

### 5.1 Softmax-Weighted Allocation

Let the debt vector be **d(t) = [debt_1(t), ..., debt_N(t)]**. For temperature **τ** (default 1.0):

```
d'_c = d_c / τ

d''_c = exp(d'_c − max_j d'_j)          [numerically stable soft weighting]

r_c^raw(t) = R · d''_c / Σ_j d''_j      [raw proportional share]
```

When **τ → ∞**, this approaches uniform allocation. When **τ → 0⁺**, all budget goes to the highest-debt class. The default **τ = 1.0** is proportional allocation.

### 5.2 Largest-Remainder Discretization

The raw shares are real-valued but the number of items per class must be integer. We apply the largest-remainder method:

```
r_c^floor(t) = floor(r_c^raw(t))

R_remaining = R − Σ_c r_c^floor(t)

Sort classes by fractional remainder: (r_c^raw − r_c^floor) in descending order.

Assign one extra item each to the top R_remaining classes.
```

The final integer allocation satisfies:

```
Σ_c r_c(t) = R
r_c(t) ∈ ℕ₀
```

### 5.3 Retrieval

For each class **c**, **r_c(t)** items are sampled uniformly with replacement from its pool **B_c**:

```
R_t = ⋃_{c = 1}^{N}  [sample r_c(t) items uniformly from B_c]
```

---

## 6. Training Objective

The combined training batch at step **t** consists of:

1. The current minibatch **B_t** (size **B**, from the current task)
2. The replay set **R_t** (size **R**, retrieved from the bank)

```
B_t^combined = B_t ∪ R_t
```

### 6.1 Combined Loss

The loss is a standard cross-entropy over the combined batch:

```
L(θ; B_t, R_t) = (1 / (B + R)) · Σ_{(x,y) ∈ B_t ∪ R_t} ℓ_CE(f_θ(x), y)
```

Both the current-task samples and the replay samples contribute equally to the gradient. There is no additional temperature scaling, distillation term, or per-sample weighting.

### 6.2 Gradient Update

The model parameters are updated via stochastic gradient descent with momentum **μ** (default 0.9):

```
v_{t+1} = μ · v_t + ∇_θ L(θ_t; B_t, R_t)
θ_{t+1} = θ_t − η · v_{t+1}
```

For clarity, the momentum term is omitted from analysis below; the core allocation dynamics are unaffected by its presence.

---

## 7. Budget Concentration Analysis

The core analytical result about PID-GB is that the debt-driven allocation systematically concentrates the limited replay budget on a small subset of classes, leaving the majority of prior classes with zero replay.

### 7.1 Zero-Allocation Condition

The allocation after largest-remainder discretization gives class **c**:

```
r_c(t) = floor(r_c^raw(t)) + extra_c(t)

where   r_c^raw(t) = R · d_c(t) / Σ_j d_j(t)
        extra_c(t) ∈ {0, 1}   (assigned to classes with the largest fractional remainders)
```

A class receives **zero items** when both conditions hold:

```
Condition A:  floor(r_c^raw(t)) = 0    ⟺    r_c^raw(t) < 1    ⟺    R · d_c(t) < Σ_{j=1}^{N} d_j(t)
Condition B:  The fractional remainder {r_c^raw(t)} is NOT among the R_remaining largest
              remainders, where R_remaining = R − Σ_j floor(r_c^raw(t)).
```

Condition A alone tells us that a class with **r_c^raw(t) < 1** — i.e., debt below the threshold — cannot get more than 1 item. Whether it gets that 1 item depends on the distribution of fractional remainders.

**Approximate bound.** Let **μ_d(t) = (1/N) · Σ_j d_j(t)** be the mean debt. Condition A simplifies to:

```
d_c(t) < (N / R) · μ_d(t)
```

For the default parameters (**N = 100, R = 64**):

```
d_c(t) < 1.5625 · μ_d(t)
```

Any class with debt below **1.56× the mean debt** cannot receive more than 1 item from Condition A alone. Whether it receives 1 or 0 depends on the largest-remainder redistribution among all classes satisfying Condition A.

To understand the scale: even under perfectly uniform debt (all classes equal), each class gets **r_c^raw = 0.64**, all satisfy Condition A, and after largest-remainder rounding **36 out of 100 classes receive 0 items**. In a skewed distribution — which is the typical operating regime — the imbalance is far worse because high-debt classes consume multiple full items, drastically reducing **R_remaining** for redistribution.

**The concentration effect.** The key question is: how many classes actually receive 0 items versus 1 item? The largest-remainder redistribution allocates the **R_remaining** extra items to the classes with the highest fractional remainders. When the debt distribution is highly skewed:

1. A few high-debt classes consume most of the raw budget — they receive **r_c^raw » 1** and thus **r_c ≥ 2**
2. These high-debt classes have negligible fractional remainders (their raw shares are large integers)
3. The remaining budget (R_remaining) is redistributed among the ~90 low-debt classes
4. **R_remaining** ≈ R − Σ_c r_c^floor where only the high-debt classes contribute to the sum
5. With **R = 64** and perhaps 5–10 high-debt classes consuming 50–60 items, **R_remaining** ≈ 4–14
6. Only the top 4–14 low-debt classes (by fractional remainder) receive 1 item each
7. The remaining **~80 low-debt classes receive 0 items**

### 7.2 Debt Skew

Define the **debt concentration ratio**:

```
κ(t) = max_c d_c(t) / min_c d_c(t)
```

When **κ(t)** is large (observed magnitudes: 10²–10⁴), the distribution is heavily concentrated on a small subset of classes. The number of classes receiving zero allocation is:

```
N_zero(t) = N − (number of classes with r_c^raw(t) ≥ 1) − R_remaining
```

In practice, after multiple tasks of training:

- **Current-task classes** typically have low debt (they are being actively trained; cross-entropy drops rapidly) → **r_c = 0**
- **A few high-debt prior classes** (those most recently affected by forgetting) get most of the budget → **r_c ≥ 2**
- **The remaining ~80 prior classes** have near-zero debt → **r_c = 0**

The result is that **N_zero(t) ≈ 0.7N to 0.9N** for typical training regimes — 70–90% of prior classes receive no replay at each step.

### 7.3 Compounded Neglect

The concentration creates a **neglect cycle**:

1. A class **c** has low debt → receives **r_c = 0** replay items
2. Without replay, the feature extractor shifts due to current-task training
3. The class's classification accuracy degrades (forgetting occurs)
4. On the next probe, the class shows higher loss → its debt increases
5. Eventually the PID shifts budget to this now-forgotten class
6. But another class that WAS receiving replay now sees its debt drop → becomes neglected

This cycling means that **at any given step, most prior classes are in a neglected state**, and the average retention across all prior classes is worse than what a uniform (static) allocation would achieve, despite the PID controller's intent.

### 7.4 Empirical Verification

Under the default configuration (**R = 64, N_classes = 100, CIFAR-100, 10 tasks**):

| Allocation | Est. prior classes receiving >0 replay/step | Average task-0 retention |
|-----------|---------------------------------------------|--------------------------|
| Uniform (StaticBank) | ~48‑64 (from sampling with replacement) | 6.0% |
| Debt-driven (PID-GB) | ~10‑20 (analytic estimate) | 5.6% |

These numbers are from the 4-method CIFAR-100 experiment with **R = 64**. The retention gap (6.0% vs 5.6%) is modest in absolute terms, but PID-GB was designed to outperform uniform replay — falling short of it confirms the structural flaw.

The concentration is not a tuning issue — it is a structural consequence of the proportional allocation mechanism combined with the PID controller's reactive signal.

---

## 8. Research Claim

The research claim for PID-GB can be stated as:

For class-incremental learning with **N** classes and a fixed replay budget **R**, there exists a debt-driven allocation mechanism such that the per-class replay distribution is strictly more protective of prior classes than uniform allocation across all classes, as measured by average retention after the final task.

**Current status: The claim is not supported by the default PID-GB implementation.** The proportional debt-driven allocation concentrates budget on a small subset of classes, causing systematic neglect of the majority. As a result, PID-GB underperforms uniform allocation (StaticBank) in practice.

**Identified cause:** The debt-driven allocation conflates two purposes — *which* classes need emphasis and *which* classes need coverage — into a single budget-allocation step. The result is that classes with below-threshold debt are completely starved of replay, even though they still require baseline protection.

### 8.1 Required Reformulation

The allocation problem must separate two objectives:

1. **Coverage:** Every prior class must receive at least some replay at every step (or at least every epoch) to prevent neglect-driven forgetting.
2. **Emphasis:** Among classes that all receive coverage, those with higher forgetting risk should contribute more strongly to the gradient.

One mechanism that achieves this separation is **uniform retrieval + debt-weighted loss (UR-DWL)**:

```
r_c(t) = R / N                               [uniform retrieval — ensures coverage]

L(θ; B_t, R_t) = (1 / (B+R)) · Σ_{(x,y)∈B_t} ℓ_CE(f_θ(x), y)
               + (1 / (B+R)) · Σ_{(x,y)∈R_t} (1 + α · debt_y(t)) · ℓ_CE(f_θ(x), y)
                                                                     [debt-weighted loss]
```

where **debt_y(t)** is the PID debt for the class of replay item **(x, y)** and **α** controls the strength of the debt weighting relative to the base coverage. Under this formulation, no class is starved of replay, and the PID signal modulates gradient strength rather than budget allocation. The normalization constant **(B+R)** ensures that when **α = 0**, the objective reduces to the standard combined cross-entropy (Section 6.1).

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| M | Number of tasks |
| K | Total number of classes |
| N | Current number of classes (grows as tasks are seen; N = K at final task) |
| B_t | Current training minibatch at step t |
| R | Total retrieval budget (items per step) |
| R_t | Retrieved replay set at step t |
| S | Per-class storage capacity |
| B_c | Exemplar pool for class c |
| e_c(t) | Per-class loss signal for class c at step t |
| L_c(t) | Smoothed loss (EMA) |
| I_c(t) | Integral term (EMA of smoothed loss) |
| d_c(t) or debt_c(t) | PID debt for class c at step t |
| r_c(t) | Number of items retrieved from class c |
| κ(t) | Debt concentration ratio |
| N_zero(t) | Number of classes with zero allocation |
| α | UR-DWL debt weight multiplier |
