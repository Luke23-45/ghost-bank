# Uniform Herding: Formal Definition

## 1. Setting

We study class-incremental learning over tasks \(t=0,\ldots,T-1\). Task
\(t\) introduces \(C_{\mathrm{new}}\) new classes, and
\(\mathcal{C}_t\) denotes the classes observed through task \(t\). Thus

\[
C_t=|\mathcal{C}_t|=(t+1)C_{\mathrm{new}}
\]

for the CIFAR-100 split used here, where \(C_{\mathrm{new}}=10\). The model
has a feature extractor

\[
f_\theta:\mathcal{X}\to\mathbb{R}^d
\]

and a cosine-margin classifier head during training. The method distinguishes
the active selected-exemplar budget from the additional candidate storage used
to refresh that active set. Evaluation uses nearest-mean-of-exemplars (NME)
classification unless an experiment explicitly changes the readout.

## 2. Notation

| Symbol | Meaning |
|---|---|
| \(M\) | Active selected-exemplar budget across observed classes |
| \(b\) | Exemplars retrieved per optimization step |
| \(\mathcal{B}_c^{(t)}\) | Persistent candidate pool for class \(c\) after task \(t\) |
| \(\mathcal{P}_{c,\mathrm{cur}}^{(t)}\) | Transient current-task stream for class \(c\) |
| \(\mathcal{Q}_c^{(t)}\) | Candidates available for the boundary rebuild |
| \(\mathcal{E}_c^{(t)}\) | Active selected-exemplar set for class \(c\) |
| \(q_c^{(t)}\) | Active quota assigned to class \(c\) |
| \(f_\theta\) | Feature map after the current task's training |
| \(\mu_c^{(t)}\) | Mean feature of \(\mathcal{Q}_c^{(t)}\) |
| \(\hat{\mu}_c^{(t)}\) | Mean feature of \(\mathcal{E}_c^{(t)}\) |
| \(\rho\) | Candidate-pool multiplier; the implementation default is \(3\) |

Throughout, \(\bar{u}=u/\|u\|_2\) denotes \(\ell_2\)-normalization.

## 3. Active Allocation

At the end of task \(t\), the active budget \(M\) is distributed uniformly
over the \(C_t\) observed classes. Internal class identifiers are ordered by
their appearance in the stream; write
\(\mathcal{C}_t=\{c_0,\ldots,c_{C_t-1}\}\) in that order. The allocation is

\[
q_{c_i}^{(t)}=
\left\lfloor\frac{M}{C_t}\right\rfloor
+\mathbf{1}\{i<M\bmod C_t\},
\qquad i=0,\ldots,C_t-1.
\]

The remainder is assigned to the lowest internal identifiers, so

\[
q_{c_i}^{(t)}\in
\left\{\left\lfloor M/C_t\right\rfloor,
\left\lceil M/C_t\right\rceil\right\},
\qquad
\sum_{c\in\mathcal{C}_t}q_c^{(t)}=M.
\]

In the experimental regime, \(M\ge C_t\), so every observed class receives a
positive quota. The implementation selects no more than the available number
of candidates if a candidate pool is smaller than its quota.

## 4. Active Exemplars and Candidate Pools

At task \(t\), \(\mathcal{B}_c^{(t-1)}\) denotes the persistent candidate pool
available for an old class. Raw examples from the current task form the
transient stream \(\mathcal{P}_{c,\mathrm{cur}}^{(t)}\). The candidates
available for the boundary rebuild are

\[
\mathcal{Q}_c^{(t)}=
\mathcal{B}_c^{(t-1)}\cup
\mathcal{P}_{c,\mathrm{cur}}^{(t)},
\]

with an empty persistent pool for a newly introduced class. Stored images are
unaugmented and current-task dataset indices are deduplicated. After the
boundary selection, the active set \(\mathcal{E}_c^{(t)}\) is replayed during
future training. The persistent candidate pool retains the selected items and
bounded leftover candidates; the current-task stream itself is not retained as
an unbounded historical archive.

With \(\rho=3\), the candidate pool has capacity at most
\(\rho q_c^{(t)}\) per class after a completed rebuild and at most \(\rho M\)
in total when all reported quotas are positive. Before a newly observed class
has completed its first rebuild, its transient stream may exceed this boundary
bound. Thus, \(M\) is the active replay budget, not the total storage
footprint.

## 5. Herding Selection

After task \(t\) has finished training and before evaluation, Uniform Herding
rebuilds every observed class in the current feature space. Let

\[
\mu_c^{(t)}=
\frac{1}{|\mathcal{Q}_c^{(t)}|}
\sum_{x\in\mathcal{Q}_c^{(t)}}f_\theta(x).
\]

For class \(c\), if \(q_c^{(t)}\ge|\mathcal{Q}_c^{(t)}|\), all available
candidates are selected. Otherwise, let \(S_0=0\) and select
\(x_1,\ldots,x_q\), where \(q=q_c^{(t)}\), by

\[
x_k\in\underset{x\in\mathcal{Q}_c^{(t)}\setminus
\{x_1,\ldots,x_{k-1}\}}{\operatorname*{argmin}}
\left\|f_\theta(x)-\left(k\mu_c^{(t)}-S_{k-1}\right)\right\|_2^2,
\qquad
S_k=S_{k-1}+f_\theta(x_k).
\]

The active exemplar set and NME prototype are

\[
\mathcal{E}_c^{(t)}=\{x_1,\ldots,x_q\},
\qquad
\hat{\mu}_c^{(t)}=
\frac{1}{|\mathcal{E}_c^{(t)}|}
\sum_{x\in\mathcal{E}_c^{(t)}}f_\theta(x).
\]

The same greedy herding primitive is used by iCaRL. The distinction is that
Uniform Herding refreshes all observed classes from current candidate pools,
whereas iCaRL herds a class once on arrival and later keeps a prefix of its
prioritized order.

## 6. Training Objective

Let

\[
\mathcal{E}^{(t-1)}=
\bigcup_{c\in\mathcal{C}_{t-1}}\mathcal{E}_c^{(t-1)}
\]

be the active selected exemplars available before task \(t\). For \(t\ge1\),
each current-task minibatch is augmented with \(b\) items drawn uniformly with
replacement from \(\mathcal{E}^{(t-1)}\). Sampling is uniform over items, not
classes, and retrieved images are re-augmented. Task \(0\) uses only its
current-task minibatches.

For scale \(s\), margin \(m\), normalized feature \(\bar f(x)\), normalized
weight \(\bar w_j\), and target \(y\), the training logits are

\[
z_j(x;y)=
\begin{cases}
+s\langle\bar f(x),\bar w_j\rangle-sm,&j=y,\\
+s\langle\bar f(x),\bar w_j\rangle,&j\ne y.
\end{cases}
\]

For \(t\ge1\), Uniform Herding uses cross-entropy plus old-class
distillation:

\[
\mathcal{L}_t=
\mathcal{L}_{\mathrm{CE}}(z(x;y),y)+
\lambda T^2\operatorname{KL}\!\left(
\operatorname{softmax}\!\left(z^{\mathrm{teach}}_{0:C_{t-1}-1}(x)/T\right)
\middle\|
\operatorname{softmax}\!\left(\tilde z_{0:C_{t-1}-1}(x;y)/T\right)
\right),
\]

where \(z^{\mathrm{teach}}\) is captured immediately before expanding the
classifier head and \(\tilde z\) is the student's logit vector after undoing
the target-dependent cosine margin for the KD comparison. The reported values
are \(\lambda=1\) and \(T=2\); the KD term is absent for \(t=0\).

## 7. Prediction

At test time, Uniform Herding uses NME prediction over selected exemplars:

\[
\hat y(x)=
\underset{c\in\mathcal{C}_{T-1}}{\operatorname*{argmin}}
\left\|
\bar f_\theta(x)-\bar{\hat\mu}_c^{(T-1)}
\right\|_2.
\]

This is equivalent to maximum cosine similarity to the selected-exemplar
prototype. Head-logit prediction is used only for the designated ablation and
the static-bank baseline's native protocol.

## 8. Configuration Contract

The manuscript's main Uniform Herding configuration is:

| Component | Value |
|---|---|
| Dataset | CIFAR-100, 10 tasks \(\times\) 10 classes |
| Backbone | ResNet-18, base filters 64, \(d=512\) |
| Training | SGD, learning rate 0.1, momentum 0.9, weight decay \(5\times10^{-4}\), batch size 128, 70 epochs per task, mixed precision |
| Head | Cosine-margin, \(s=30\), \(m=0.35\), imprinting enabled |
| Active budget | \(M=2000\), floor quota 1 |
| Retrieval | \(b=64\), uniform sampling with replacement from selected exemplars |
| Candidate multiplier | \(\rho=3\), implementation default; not independently swept |
| Distillation | \(\lambda=1.0\), \(T=2.0\) |
| Prediction | NME on normalized features and selected-exemplar means |

This definition is the mathematical contract for Uniform Herding. The iCaRL
baseline is a separate complete protocol: it uses arrival-time herding,
prefix truncation for old classes, NME prediction, and its BCE-style target
objective.
