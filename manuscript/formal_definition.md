# Uniform Herding — Formal Definition

## 1. Setting

Class-incremental learning over a sequence of tasks $t = 0, \dots, T-1$. Task $t$
introduces $C_{\text{new}}$ new classes; the set of classes seen by the end of
task $t$ is $\mathcal{C}_t$, with $C_t = |\mathcal{C}_t| = (t+1)\,C_{\text{new}}$.
The model is a backbone $f_\theta: \mathcal{X} \to \mathbb{R}^d$ (feature
extractor) followed by a cosine-margin head. Exemplar storage is subject to a
**fixed total budget** $M$ that never grows with $t$.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $M$ | total exemplar budget across all classes (fixed, e.g. 2000) |
| $b$ | per-step retrieval budget (e.g. 64) |
| $\mathcal{P}_c^{(t)}$ | storage pool of class $c$ after task $t$ (all unique raw images of $c$ seen so far) |
| $\mathcal{E}_c^{(t)}$ | exemplar (herding-selected) set of class $c$ at end of task $t$, $|\mathcal{E}_c^{(t)}| = q_c^{(t)}$ |
| $\varphi^{(t)}(\cdot)$ | backbone feature map after training task $t$ (evaluation mode, no augmentation) |
| $\mu_c^{(t)}$ | pool mean $\frac{1}{|\mathcal{P}_c^{(t)}|}\sum_{x \in \mathcal{P}_c^{(t)}} \varphi^{(t)}(x)$ |
| $\hat{\mu}_c^{(t)}$ | exemplar-mean $\frac{1}{q_c^{(t)}}\sum_{x \in \mathcal{E}_c^{(t)}} \varphi^{(t)}(x)$ (NME prototype) |

## 3. Uniform fixed-total allocation

At the end of task $t$ the budget $M$ is split **uniformly over the classes seen so far**:

$$
q_c^{(t)} = \left\lfloor \frac{M}{C_t} \right\rfloor + \mathbf{1}\{c < (M \bmod C_t)\},
\qquad c \in \mathcal{C}_t,
$$

so that $q_c^{(t)} \in \{\lfloor M/C_t \rfloor, \lceil M/C_t \rceil\}$ and
$\sum_{c \in \mathcal{C}_t} q_c^{(t)} = M$ exactly. Each class additionally
receives at least $\text{floor} \ge 1$ exemplars. Unlike a fixed per-class quota,
the per-class quota shrinks as $C_t$ grows while the total stays $M$.

## 4. Storage pool

During training of task $t$, every raw (pre-augmentation) training image of the
current task is appended to its class pool; each training sample is stored at
most once (deduplication by dataset index). Pools are never evicted:
$\mathcal{P}_c^{(t)}$ is monotonically non-decreasing in $t$.

## 5. Herding selection (rebuild)

After training task $t$ (and before evaluation), the exemplar sets are
**re-selected from scratch** in the current feature space $\varphi^{(t)}$:
for each class $c \in \mathcal{C}_t$ with quota $q = q_c^{(t)}$ and non-empty pool,
if $q \ge |\mathcal{P}_c^{(t)}|$ keep the whole pool; otherwise select $q$
exemplars greedily so that the running mean of the selected features tracks the
pool mean (Welling, 2009; iCaRL's ConstructExemplarSet, Chen et al., 2019):

$$
\mathbf{s}_0 = 0, \qquad
x_k = \underset{x \in \mathcal{P}_c^{(t)} \setminus \{x_1,\dots,x_{k-1}\}}{\operatorname{argmin}}
\left\| \varphi^{(t)}(x) - \left( k\,\mu_c^{(t)} - \mathbf{s}_{k-1} \right) \right\|_2^2,
\qquad \mathbf{s}_k = \mathbf{s}_{k-1} + \varphi^{(t)}(x_k),
$$

for $k = 1, \dots, q$; then $\mathcal{E}_c^{(t)} = \{x_1, \dots, x_q\}$. This is
algebraically identical to iCaRL's
$\,x_k = \operatorname{argmin}_{x \in \mathcal{P}_c^{(t)}}\left\|\mu_c^{(t)} - \frac{1}{k}\left[\varphi^{(t)}(x) + \mathbf{s}_{k-1}\right]\right\|$,
the factor $1/k > 0$ leaving the argmin unchanged. The NME prototype is
$\hat{\mu}_c^{(t)} = \mathbf{s}_q / q$, the mean of the **selected** exemplars.

Rebuild runs after **every** task (including task 0), so selections adapt to the
evolving representation rather than being fixed at first observation (contrast
iCaRL, which selects once and only truncates afterwards).

## 6. Training

While training task $t$: at every step the current batch $\{(x_i, y_i)\}$ is
stored into the pools, and $b$ exemplars are drawn uniformly at random **with
replacement** from the union of the exemplar sets selected at the end of task
$t-1$ (each stored exemplar is equally likely), freshly augmented, and
concatenated to the current batch. The loss on the combined batch is

$$
\mathcal{L} = \mathrm{CE}\!\left(s \cdot \big(\langle \bar{f}, \bar{w}_y \rangle - m\,\mathbf{1}\{\text{target}\}\big),\; y\right)
+ \lambda \; T^2 \; \mathrm{KL}\!\Big(\mathrm{softmax}(z_{\text{old}} / T) \,\big\|\,
\mathrm{softmax}(z_{\text{old}}^{\text{teach}} / T)\Big),
$$

where $\mathrm{CE}$ is cross-entropy on the scaled cosine-margin logits ($s$
scale, $m$ margin, applied in training mode), the second term is Hinton
distillation on the logits restricted to previously seen classes
$\mathcal{C}_{t-1}$, and $z_{\text{old}}^{\text{teach}}$ are the logits of a
frozen copy of the model taken **before** the head expansion of task $t$
(active only for $t \ge 1$).

## 7. Prediction (Nearest Mean Exemplar)

New head rows of task $t \ge 1$ are initialized by LUCIR-style weight
imprinting (L2-normalized class-mean features of the new task's data). At test
time the head is not read out: the query feature and all prototypes are
L2-normalized and the query is assigned to the nearest prototype,

$$
\hat{y}(x) = \underset{c \in \mathcal{C}_{T-1}}{\operatorname{argmin}}\,
\left\| \frac{\varphi^{(T-1)}(x)}{\|\varphi^{(T-1)}(x)\|_2} -
\frac{\hat{\mu}_c}{\|\hat{\mu}_c\|_2} \right\|_2,
$$

i.e., maximum cosine similarity to the exemplar mean (iCaRL's mean-of-exemplars
rule). If no prototypes are available, fall back to head logits.

## 8. Reference configuration (experiments)

| Component | Value |
|---|---|
| Data | CIFAR-100, 10 tasks $\times$ 10 classes, 70 epochs/task, SGD (lr 0.1, momentum 0.9, weight decay 5e-4), batch 128, AMP fp16 |
| Backbone | ResNet-18 ($d = 512$) |
| Head | Cosine-margin, scale $s = 30$, margin $m = 0.35$, imprinting on |
| Budget | $M = 2000$, floor $= 1$ |
| Retrieval | $b = 64$, uniform random with replacement over exemplar set |
| Distillation | $\lambda = 1.0$, $T = 2.0$ |
| Prediction | NME on L2-normalized features and exemplar means |
