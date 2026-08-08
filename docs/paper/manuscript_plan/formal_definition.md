# Uniform Herding: Formal Definition

## 1. Setting

We study class-incremental learning over tasks \(t = 0, \dots, T-1\). Task \(t\)
introduces \(C_{\mathrm{new}}\) new classes, and \(\mathcal{C}_t\) denotes the set
of classes observed up to and including task \(t\). Hence
\[
C_t = |\mathcal{C}_t| = (t+1) C_{\mathrm{new}}
\]
in the CIFAR-100 split used in the experiments, where \(C_{\mathrm{new}} = 10\).

The model consists of a feature extractor
\[
f_\theta : \mathcal{X} \to \mathbb{R}^d
\]
and a cosine-margin classifier head used during training. Exemplar storage is governed
by a fixed total memory budget \(M\) that does not grow with \(t\). In the
reference configuration, evaluation is performed with nearest-mean exemplar
(NME) classification.

## 2. Notation

| Symbol | Meaning |
|---|---|
| \(M\) | total exemplar budget across all classes |
| \(b\) | number of exemplars retrieved per optimization step |
| \(\mathcal{P}_c^{(t)}\) | raw storage pool for class \(c\) after task \(t\) |
| \(\mathcal{E}_c^{(t)}\) | selected exemplar set for class \(c\) after task \(t\) |
| \(q_c^{(t)}\) | quota assigned to class \(c\) after task \(t\) |
| \(\varphi^{(t)}(\cdot)\) | backbone feature map after training task \(t\) |
| \(\mu_c^{(t)}\) | mean feature of the storage pool \(\mathcal{P}_c^{(t)}\) |
| \(\hat{\mu}_c^{(t)}\) | mean feature of the selected exemplar set \(\mathcal{E}_c^{(t)}\) |
| \(z(x; y)\) | cosine-margin logits for input \(x\) with target class \(y\) |

Throughout, \(\bar{u} = u / \|u\|_2\) denotes \(\ell_2\)-normalization.

## 3. Fixed-Total Allocation

At the end of task \(t\), the memory budget \(M\) is distributed uniformly over
the \(C_t\) classes seen so far. The implementation uses contiguous internal
class identifiers in order of their appearance in the class-incremental stream;
therefore, write \(\mathcal{C}_t = \{c_0, \dots, c_{C_t-1}\}\) in that canonical
order. The allocation is
\[
q_{c_i}^{(t)} = \left\lfloor \frac{M}{C_t} \right\rfloor
+ \mathbf{1}\{i < M \bmod C_t\}, \qquad i = 0, \dots, C_t-1.
\]
The remainder is assigned to the lowest internal class identifiers. This gives
\(q_{c_i}^{(t)} \in \{\lfloor M / C_t \rfloor, \lceil M / C_t \rceil\}\)
and
\[
\sum_{c \in \mathcal{C}_t} q_c^{(t)} = M.
\]
In the experimental regime of this study, \(M \ge C_t\), so every seen class
receives at least one exemplar.

## 4. Storage Pools

During training of task \(t\), each raw training example is inserted once into
its class pool. Only unaugmented samples are stored, and each dataset index is
deduplicated so that no example appears more than once in a given pool.
Consequently, \(\mathcal{P}_c^{(t)}\) is monotone nondecreasing in \(t\):
\[
\mathcal{P}_c^{(t-1)} \subseteq \mathcal{P}_c^{(t)}.
\]

## 5. Herding Selection

After task \(t\) has finished training and before evaluation, the exemplar sets
are rebuilt from scratch in the current feature space \(\varphi^{(t)}\). For
each class \(c \in \mathcal{C}_t\):

- if \(q_c^{(t)} \ge |\mathcal{P}_c^{(t)}|\), then \(\mathcal{E}_c^{(t)} =
  \mathcal{P}_c^{(t)}\);
- otherwise, select \(q_c^{(t)}\) exemplars greedily so that the running mean of
  the selected features tracks the class mean.

Let \(q = q_c^{(t)}\), \(S_0 = 0\), and let \(x_1, \dots, x_q\) be the selected
examples. The greedy step is
\[
x_k \in \underset{x \in \mathcal{P}_c^{(t)} \setminus \{x_1, \dots, x_{k-1}\}}
{\operatorname*{argmin}}
\left\|
\varphi^{(t)}(x) - \left(k \mu_c^{(t)} - S_{k-1}\right)
\right\|_2^2,
\qquad
S_k = S_{k-1} + \varphi^{(t)}(x_k),
\]
for \(k = 1, \dots, q\). The selected exemplar set is
\[
\mathcal{E}_c^{(t)} = \{x_1, \dots, x_q\},
\qquad
\hat{\mu}_c^{(t)} = \frac{1}{q} \sum_{j=1}^q \varphi^{(t)}(x_j) = \frac{S_q}{q}.
\]
This is the standard iCaRL herding rule written in equivalent form: the selected
set is chosen so that its mean approximates the full-pool mean \(\mu_c^{(t)}\).

Selection is recomputed after every task, including task \(0\). The exemplar set
is therefore adapted to the current representation rather than fixed once and
only truncated thereafter.

## 6. Training Objective

Let
\[
\mathcal{E}^{(t-1)} = \bigcup_{c \in \mathcal{C}_{t-1}} \mathcal{E}_c^{(t-1)}
\]
denote the union of exemplars available before task \(t\). During training on
task \(t \ge 1\), each minibatch from the current task is augmented with \(b\)
exemplars drawn uniformly at random with replacement from \(\mathcal{E}^{(t-1)}\).
The sampling is uniform over the stored exemplar items, not uniform over
classes. The retrieved samples are re-augmented before being concatenated to the
current minibatch. Task \(0\) has no previous exemplar set and therefore uses
only its current-task minibatches.

The classifier uses cosine-margin logits. Let \(w_j\) be the classifier weight
for class \(j\), \(s > 0\) the scale, and \(m > 0\) the additive margin. For an
input \(x\) with label \(y\), define the normalized feature and weight vectors by
\[
\bar{f}(x) = \frac{f_\theta(x)}{\|f_\theta(x)\|_2},
\qquad
\bar{w}_j = \frac{w_j}{\|w_j\|_2}.
\]
The logits are
\[
z_j(x; y) =
\begin{cases}
s \left( \langle \bar{f}(x), \bar{w}_j \rangle - m \right), & j = y, \\
s \langle \bar{f}(x), \bar{w}_j \rangle, & j \ne y.
\end{cases}
\]

For task \(t\), the training loss is
\[
\mathcal{L}_t(x,y) = \mathcal{L}_{\mathrm{CE}}(z(x; y), y)
+ \mathbf{1}\{t \ge 1\}\,\lambda T^2\,
\mathrm{KL}\!\left(
\operatorname{softmax}\!\left(\frac{z^{\mathrm{teach}}_{0:C_{t-1}-1}(x)}{T}\right)
\bigg\|
\operatorname{softmax}\!\left(\frac{z_{0:C_{t-1}-1}(x)}{T}\right)
\right),
\]
where \(z^{\mathrm{teach}}(x)\) is the frozen teacher logit vector captured
immediately before expanding the head for task \(t\). The distillation term is
applied only on classes seen before task \(t\).

## 7. Prediction

At test time, the reference protocol does not use the classifier head. Instead,
each query is assigned to the nearest NME prototype in normalized feature space:
\[
\hat{y}(x) =
\underset{c \in \mathcal{C}_{T-1}}{\operatorname*{argmin}}
\left\|
\bar{\varphi}^{(T-1)}(x) - \bar{\hat{\mu}}_c^{(T-1)}
\right\|_2,
\qquad
\bar{\varphi}^{(T-1)}(x) = \frac{\varphi^{(T-1)}(x)}{\|\varphi^{(T-1)}(x)\|_2},
\qquad
\bar{\hat{\mu}}_c^{(T-1)} = \frac{\hat{\mu}_c^{(T-1)}}{\|\hat{\mu}_c^{(T-1)}\|_2}.
\]
Equivalently, this is maximum cosine similarity to the exemplar mean. A
head-logit fallback is retained only as an implementation safeguard; it is not
the reference evaluation rule.

## 8. Reference Configuration

The locked reference configuration used for the manuscript is:

| Component | Value |
|---|---|
| Dataset | CIFAR-100, 10 tasks x 10 classes |
| Backbone | ResNet-18, base filters 64, \(d = 512\) |
| Training | SGD with learning rate 0.1, momentum 0.9, weight decay 5e-4, batch size 128, 70 epochs per task, AMP fp16 |
| Head | Cosine-margin, \(s = 30\), \(m = 0.35\), imprinting enabled; explicit reference override \(\texttt{model.head=cosine\_margin}\) |
| Memory | \(M = 2000\), floor quota \(= 1\) |
| Retrieval | \(b = 64\), uniform sampling with replacement from the exemplar set |
| Distillation | \(\lambda = 1.0\), \(T = 2.0\); explicit reference override \(\texttt{method.kd\_weight=1.0}\) |
| Prediction | NME on \(\ell_2\)-normalized features and exemplar means |

This definition is the mathematical contract for the reference method used in
the paper. Every result in the manuscript should be interpretable against this
specification without additional assumptions.
