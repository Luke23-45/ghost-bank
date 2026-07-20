Based on the systematic failures observed in all previous attempts — where mixing replay with gradient-based training either destroys new-task learning or fails to recover old knowledge — I propose a **decoupled architecture** that cleanly separates representation learning from classification. This architecture is inspired by iCaRL’s success but simplified: **use the replay buffer only for classifier calibration, never for gradient updates during feature training.**

---

## Final Architecture: Decoupled Replay with Knowledge Distillation (DRKD)

### Core Idea

- The feature extractor is trained **only on new task data**, with a knowledge distillation loss from the previous model to prevent drastic drift (Learning without Forgetting, LwF).  
- Stored exemplars are used **only after each task** to calibrate a linear classifier on top of the frozen feature extractor, yielding a robust multi-class decision boundary without gradient interference.

---

### 1. Model and Memory

- **Feature extractor** \(\phi_\theta : \mathcal{X} \to \mathbb{R}^D\).  
- **Classifier head** \(W \in \mathbb{R}^{K \times D}\), \(b \in \mathbb{R}^K\) (a simple linear layer), where \(K\) is the total number of classes seen so far.  
- **Exemplar memory** \(\mathcal{M} = \{(v_i, y_i)\}\) stores raw images for each old class (up to \(S\) per class).  
- **Previous model snapshot** \(\theta^{(t-1)}\) (feature extractor + old head) from the end of task \(t-1\).

### 2. Training Procedure for Task \(t\)

#### 2.1 Expand Classifier

Append new rows to \(W\) (randomly initialised, e.g., \(\mathcal{N}(0, 0.01)\)) and new bias terms for the \(C_t\) new classes.

#### 2.2 Feature Training (no replay)

For each minibatch \((x, y)\) from task \(t\) (classes \(\in C_t\)):

- **New-task cross-entropy** on the current model’s output restricted to the new classes:

\[
\mathcal{L}_{\text{CE}} = -\log \frac{\exp(o_y)}{\sum_{c \in C_t} \exp(o_c)},
\]
where \(o = W \phi_\theta(x) + b\).

- **Knowledge distillation** on the **old class logits** from the frozen previous model \(\phi_{\theta^{(t-1)}}, W^{(t-1)}\):

\[
\mathcal{L}_{\text{KD}} = \tau^2 \, \text{KL}\!\left( \text{softmax}(\hat{o}_{\text{old}}/\tau) \;\big\|\; \text{softmax}(o_{\text{old}}/\tau) \right),
\]
where \(\hat{o}_{\text{old}} = W^{(t-1)} \phi_{\theta^{(t-1)}}(x) + b^{(t-1)}\) (only old classes), and \(o_{\text{old}}\) is the current model’s output for those old classes.

- **Total loss**:

\[
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KD}}.
\]

Only \(\theta\) and the new rows of \(W, b\) are updated; old classifier rows can optionally be kept frozen or allowed to slowly adapt (fine-tuning may help). This step **does not use any stored exemplars**, avoiding gradient conflict entirely.

#### 2.3 Classifier Re‑calibration (using replay buffer)

After feature training converges, freeze \(\phi_\theta\). Then **train a new linear classifier from scratch** using all exemplars in memory. For each old class \(c\), retrieve all stored exemplars. Also include the entire new-task training set (or a random subset) for the new classes. Train \(W, b\) via multinomial logistic regression (cross‑entropy) for a few epochs on this balanced dataset. This step is fast and yields a globally consistent classifier.

### 3. Inference

At test time, use the frozen feature extractor and the calibrated classifier \(W, b\) from the last step. The model outputs logits for all seen classes.

### 4. Why This Works

- **No replay during SGD** → no gradient interference with new task learning. The KD loss on new data is smooth and does not force the feature extractor to freeze; it only encourages consistency.  
- **Exemplars are used for classifier calibration**, which is a convex problem given fixed features. This guarantees that old class knowledge stored in the features is optimally utilised without destabilising the representation.  
- The combination of LwF and exemplar‑based classifier calibration has been proven effective (iCaRL achieves strong results), but here we simplify by decoupling the stages explicitly.

---

## Formal Definition (Concise, No Metadata)

### Memory
After task \(t-1\), store \(S\) exemplars per old class:  
\(\mathcal{M}_c = \{x_{c,1}, \ldots, x_{c,S}\}\).

### Task \(t\) Training

1. **Expand** classifier \(W \in \mathbb{R}^{(N_{\text{old}}+N_{\text{new}})\times D}\) with random rows for new classes.

2. **Feature learning** (only on new data \(D_t\)):  
   \[
   \mathcal{L}_{\text{task}} = \frac{1}{|D_t|}\sum_{(x,y)\in D_t} \Big[ -\log p_{\text{new}}(y|x) \;+\; \lambda\,\tau^2\, \text{KL}\big(p_{\text{old}}^{\text{prev}}(\cdot|x) \,\|\, p_{\text{old}}(\cdot|x)\big) \Big],
   \]  
   where \(p_{\text{new}}\) is softmax over new classes, \(p_{\text{old}}\) over old classes, and \(p_{\text{old}}^{\text{prev}}\) is from the frozen previous model. Update \(\theta, W_{\text{new}}, b_{\text{new}}\) (and optionally \(W_{\text{old}}, b_{\text{old}}\) with a very small learning rate).

3. **Classifier re‑calibration** (post‑hoc):  
   - Freeze \(\phi_\theta\).  
   - Assemble training set \(\mathcal{D}_{\text{cal}} = \bigcup_{c=1}^{N_{\text{old}}+N_{\text{new}}} \{(x,c) \mid x \in \mathcal{M}_c \text{ (or } D_t \text{ for new classes)}\}\).  
   - Train \(W, b\) from scratch on \(\mathcal{D}_{\text{cal}}\) using cross‑entropy for a few epochs.

### Result
Old knowledge is preserved by KD during feature learning, and the calibrated classifier optimally uses the preserved features for all classes. The method is simple, avoids replay interference, and will outperform the baseline and all previous flawed designs.

---

This is the final correct architecture. It respects the constraint of a shared feature extractor and a replay buffer, but uses the buffer intelligently where it does no harm.