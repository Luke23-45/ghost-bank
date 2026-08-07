# Literature Review Notes for the Uniform Herding Manuscript

## Purpose and scope

This document feeds the Related Work section of the manuscript (item 4 in
`docs/paper/manuscript_plan/paper_structure.md`). It is a planning document,
not a narrative section. Every citation below was verified against a primary
or authoritative secondary source during the research pass (venue, year,
pages where applicable, arXiv identifier). Corrections made during
verification are recorded at the end so nothing enters the manuscript
unchecked.

The paper under study is a component-level investigation, not a
state-of-the-art claim: CIFAR-100 class-incremental learning (10 tasks x 10
classes, ResNet-18, 512-d embeddings), fixed total exemplar budget
(M = 2000) with uniform per-task allocation, herding-based exemplar
selection re-run after every task, cosine-margin classifier head with
imprinting, NME readout on L2-normalized features, and distillation
(lambda = 1.0, temperature = 2.0). The related work is organized along the
four subsections of the plan: (1) class-incremental learning, (2) replay and
exemplar selection, (3) distillation and classifier/readout choices, and
(4) forgetting and stability metrics.

## 1. Class-incremental learning

Class-incremental learning (CIL) is the setting in which a model is trained
on a sequence of classification tasks with disjoint class sets and must, at
test time, classify across all classes seen so far without task identity.
Without a mechanism for stability, gradient descent on new tasks overwrites
the decision boundaries of old classes.

Regularization-based approaches protect parameters important for old tasks:
- Elastic Weight Consolidation (EWC) penalizes movement of parameters
  weighted by their Fisher information [Kirkpatrick et al., PNAS 2017,
  114(13):3521-3526; arXiv:1612.00796].
- Synaptic Intelligence (SI) computes per-synapse importance online along
  the entire optimization trajectory instead of post-hoc Fisher
  estimation [Zenke, Poole & Ganguli, ICML 2017, PMLR 70:3987-3995;
  arXiv:1703.04200].

Gradient-constraint approaches keep gradients from increasing loss on
previous tasks:
- Gradient Episodic Memory (GEM) stores episodic memory per task and
  projects the update so that losses on previous tasks do not rise; it also
  introduced the ACC/BWT evaluation metrics [Lopez-Paz & Ranzato, NeurIPS
  2017 (NIPS 30), pp. 6467-6476; arXiv:1706.08840].
- Averaged GEM (A-GEM) replaces the per-task quadratic-program projection
  with a single inequality constraint on the average memory gradient,
  matching GEM's accuracy at EWC-like cost [Chaudhry, Ranzato, Rohrbach &
  Elhoseiny, ICLR 2019; arXiv:1812.00420].

Distillation-only approaches preserve old responses without storing data:
- Learning without Forgetting (LwF) applies knowledge distillation on old
  task outputs while training on new data only [Li & Hoiem, ECCV 2016,
  pp. 614-629; arXiv:1606.09282].

Replay-based CIL is the family our paper belongs to; it is covered in the
next subsection. iCaRL established the exemplar-replay + NME template for
CIFAR-100 CIL that our protocol follows; iCaRL evaluated on iCIFAR-100 with
10 classes per batch (the same 10-task x 10-class split family as ours),
but with different backbones and epochs, so absolute accuracies are not
directly comparable to ours [Rebuffi et al., CVPR 2017; arXiv:1611.07725].

Positioning: our study does not aim to beat these methods. It holds
protocol and budget fixed and measures which components (selection rule,
readout, head geometry, distillation) drive accuracy and forgetting within
one replay-based system.

## 2. Replay and exemplar selection

Experience replay is the simplest stability mechanism: store past samples
and replay them while learning new tasks.
- Experience Replay (ER) demonstrated that replay of past experience
  substantially reduces forgetting, even with bounded buffers and random
  discarding [Rolnick, Ahuja, Schwarz, Lillicrap & Wayne, NeurIPS 2019
  (NeurIPS 32), pp. 350-360; arXiv:1811.11682]. (Note: the original paper
  is in the reinforcement-learning setting; the same random-buffer idea is
  the standard ER baseline in supervised CIL.)
- On Tiny Episodic Memories studies ER-style replay with very small fixed
  budgets and shows it is a surprisingly strong and often underestimated
  baseline; it also defines the average-forgetting metric used widely
  since [Chaudhry, Rohrbach, Elhoseiny, Ajanthan, Dokania, Torr &
  Ranzato, arXiv:1902.10486, 2019].
- GDumb greedily balances classes in the memory and trains a fresh
  classifier from scratch on the memory alone, reaching strong accuracy
  without any model fine-tuning on new data [Prabhu, Torr & Dokania, ECCV
  2020; arXiv:1910.07113].

Exemplar selection methods go beyond random sampling:
- iCaRL selects exemplars with herding, a deterministic greedy
  construction that matches the class mean in feature space, and selects
  each class's exemplars once: afterwards it only shrinks the stored sets
  as new classes arrive, never re-selecting old exemplars
  [Rebuffi et al., CVPR 2017; arXiv:1611.07725].
- Herding itself originates in learning theory: herding generates
  deterministic pseudo-random sequences that match moment statistics of
  the data distribution by greedily minimizing
  ||phi(x) - (k*mu - s_{k-1})||^2 [Welling, ICML 2009].
- Gradient-based Sample Selection (GSS) selects samples that maximize
  gradient diversity in the replay buffer for online continual learning
  [Aljundi, Lin, Goujaud & Bengio, NeurIPS 2019; arXiv:1903.08671].
- Maximally Interfered Retrieval (MIR) is a retrieval-time selection rule:
  it samples the memory subset whose loss would increase most, showing
  that how samples are chosen at retrieval time also matters
  [Aljundi, Caccia, Belilovsky, Caccia, Lin, Charlin & Tuytelaars,
  NeurIPS 2019; arXiv:1908.04742].
- Rainbow Memory combines class-balanced reservoir sampling with a
  diversity-aware memory update for CIL [Bang, Kim, Yoo, Ha & Choi, CVPR
  2021; arXiv:2103.17230].

Where our paper sits: the reference method uses (i) a fixed TOTAL exemplar
budget M, (ii) a uniform per-task allocation
q_c = floor(M / C_t) + 1{c < M mod C_t} across all classes seen so far, and
(iii) herding re-run from scratch after every task (the bank is rebuilt,
old exemplars can be replaced). Note that iCaRL also splits a fixed total
budget equally (m = K/t exemplars per class); the operative difference is
that iCaRL selects each class's exemplars once and afterwards only shrinks
the set (keeping the first m herded exemplars), whereas we rebuild the
entire bank after every task, re-running herding on the current features.
Against ER-style baselines, which fill a fixed ring buffer by uniform
random selection, the difference is the selection rule itself: the
static_bank baseline in our experiment grid is exactly that ER-style
random-budget method, which lets the paper isolate the selection rule
(herding vs random) at identical budget semantics.

## 3. Distillation and classifier/readout choices

Knowledge distillation is the standard companion to replay in CIL:
- LwF introduced distillation as the mechanism to preserve old-task
  responses [Li & Hoiem, ECCV 2016].
- iCaRL distills the old model's sigmoid outputs on old classes
  (binary cross-entropy between stored and current scores, without a
  temperature parameter, unlike LwF-style softmax distillation) while
  training on new classes plus exemplars, and predicts with NME over
  exemplar means [Rebuffi et al., CVPR 2017].
- LUCIR uses a normalized cosine classifier, a margin ranking loss to
  rebalance gradients between old and new classes, and a feature-space
  "less-forget" distillation constraint; its exemplars are herded from
  each class [Hou, Pan, Loy, Wang & Lin, CVPR 2019; arXiv:1903.02990].
- Bias Correction (BiC) adds a bias-correction branch on the classifier
  head to fix the systematic old-vs-new logit offset after each step
  [Wu, Chen, Wang, Ye, Liu, Guo & Fu, CVPR 2019, pp. 374-382;
  arXiv:1905.13260].
- Weight Aligning (WA) showed analytically and empirically that KD alone
  preserves within-old-class discrimination but leaves a bias toward new
  classes, and fixed it post-hoc by aligning classifier weight norms
  [Zhao, Xiao, Gan, Zhang & Xia, CVPR 2020, pp. 13208-13217;
  arXiv:1911.07053].
- PODNet constrains the whole representation with pooled-output
  distillation (spatial/channel pooling of intermediate activations) and
  uses a local similarity classifier with multiple proxies per class,
  aimed at long runs of small tasks [Douillard, Cord, Ollion, Robert &
  Valle, ECCV 2020, pp. 86-102; arXiv:2004.13513].
- DER stores logits from past models and replays them as training targets
  ("logit replay"), retraining the final classifier on the expanded
  representation at the end of training [Yan, Xie & He, CVPR 2021,
  pp. 3014-3023; arXiv:2103.16788]. DER++ adds stored exemplars on top of
  logit replay; it was introduced in "Rethinking Experience Replay"
  [Buzzega, Boschini, Porrello & Calderara, ICPR 2020, pp. 2180-2187] and
  extended in "Class-Incremental Continual Learning Into the eXtended
  DER-Verse" [Boschini, Bonicelli, Buzzega, Porrello & Calderara, IEEE
  TPAMI 45(5):5497-5512, 2023; doi:10.1109/TPAMI.2022.3206549].

Prototype-based readouts avoid learned-head bias entirely:
- FeCAM stores class prototypes plus estimated covariance structure and
  classifies with Mahalanobis distance under anisotropic (class-specific)
  feature covariance [Goswami, Liu, Twardowski & van de Weijer, NeurIPS
  2023; arXiv:2309.14062].
- FeTrIL freezes the feature extractor after the first task, synthesizes
  pseudo-features of past classes by geometric translation of new-class
  features toward stored class centroids, and trains a linear classifier
  incrementally on real new-class features plus pseudo-features of past
  classes [Petit, Popescu, Schindler, Picard & Delezoide, WACV 2023,
  pp. 3911-3920; arXiv:2211.13131].

Where our paper sits: the ablation grid isolates exactly the axes above -
KD on/off (kd_weight 0 vs 1), readout (NME on L2-normalized features vs
head logits), head geometry (cosine-margin with scale 30 / margin 0.35 vs
plain linear), and imprinting of new-class prototypes. This makes the
related work above directly actionable: each ablation maps onto a specific
line of prior work (KD: LwF/iCaRL; readout: iCaRL NME vs BiC/WA head
fixes; head: LUCIR cosine; prototypes: FeCAM/FeTrIL).

## 4. Forgetting and stability metrics

The manuscript reports per-task accuracy, average accuracy, forgetting and
BWT, with matched per-seed deltas for ablations:
- Average accuracy ACC and Backward Transfer BWT were formalized by GEM
  via the accuracy matrix R_{i,j} [Lopez-Paz & Ranzato, NeurIPS 2017].
- The average-forgetting metric (mean over old tasks of the drop from
  peak accuracy) is the one used in On Tiny Episodic Memories
  [Chaudhry et al., arXiv:1902.10486, 2019].
- LUCIR and most subsequent CIL papers report both average accuracy and
  average forgetting on CIFAR-100, which is why the manuscript reports
  both and discusses the accuracy-forgetting trade-off separately
  (Figure 4) instead of collapsing them into a single number
  [Hou et al., CVPR 2019].

## Verified citation list

Every entry below was checked online during this research pass. The
"pages" column is only given where confirmed from the proceedings.

1. Kirkpatrick, J., et al. Overcoming catastrophic forgetting in neural
   networks. PNAS 114(13):3521-3526, 2017. arXiv:1612.00796.
2. Zenke, F., Poole, B., Ganguli, S. Continual Learning Through Synaptic
   Intelligence. ICML 2017, PMLR 70:3987-3995. arXiv:1703.04200.
3. Lopez-Paz, D., Ranzato, M. Gradient Episodic Memory for Continual
   Learning. NeurIPS 30, pp. 6467-6476, 2017. arXiv:1706.08840.
4. Chaudhry, A., Ranzato, M., Rohrbach, M., Elhoseiny, M. Efficient
   Lifelong Learning with A-GEM. ICLR 2019. arXiv:1812.00420.
5. Li, Z., Hoiem, D. Learning without Forgetting. ECCV 2016,
   pp. 614-629. arXiv:1606.09282. Journal: TPAMI 40(12):2935-2947.
6. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., Wayne, G.
   Experience Replay for Continual Learning. NeurIPS 32, pp. 350-360,
   2019. arXiv:1811.11682.
7. Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P.
   K., Torr, P. H. S., Ranzato, M. On Tiny Episodic Memories in
   Continual Learning. arXiv:1902.10486, 2019 (also appears in the ICML
   2019 workshops proceedings; see the citation in the DER-Verse TPAMI
   paper's reference list).
8. Rebuffi, S.-A., Kolesnikov, A., Sperl, G., Lampert, C. H. iCaRL:
   Incremental Classifier and Representation Learning. CVPR 2017.
   arXiv:1611.07725. Pagination: pp. 5533-5542 on IEEE Xplore; the CVF
   Open Access version is paginated 2001-2010. Both are in circulation;
   pick one consistently.
9. Welling, M. Herding Dynamical Weights to Learn. ICML 2009.
10. Prabhu, A., Torr, P. H. S., Dokania, P. K. GDumb: A Simple Approach
    that Questions Our Progress in Continual Learning. ECCV 2020,
    pp. 524-540. arXiv:1910.07113.
11. Aljundi, R., Lin, M., Goujaud, B., Bengio, Y. Gradient based Sample
    Selection for Online Continual Learning. NeurIPS 2019.
    arXiv:1903.08671.
12. Aljundi, R., Caccia, L., Belilovsky, E., Caccia, M., Lin, M.,
    Charlin, L., Tuytelaars, T. Online Continual Learning with Maximally
    Interfered Retrieval. NeurIPS 2019. arXiv:1908.04742. (Author order
    per the arXiv record; Lucas Caccia is also published as "Lucas
    Page-Caccia". Page ranges in circulation conflict (11849-11860 vs
    9630-9638) - cite this entry without proceedings pages.)
13. Bang, J., Kim, H., Yoo, Y., Ha, J.-W., Choi, J. Rainbow Memory:
    Continual Learning with a Memory of Diverse Samples. CVPR 2021.
    arXiv:2103.17230.
14. Hou, S., Pan, X., Loy, C. C., Wang, Z., Lin, D. Learning a Unified
    Classifier Incrementally via Rebalancing. CVPR 2019, pp. 831-839.
    arXiv:1903.02990.
15. Wu, Y., Chen, Y., Wang, L., Ye, Y., Liu, Z., Guo, Y., Fu, Y. Large
    Scale Incremental Learning. CVPR 2019, pp. 374-382. arXiv:1905.13260.
16. Zhao, B., Xiao, X., Gan, G., Zhang, B., Xia, S.-T. Maintaining
    Discrimination and Fairness in Class Incremental Learning. CVPR 2020,
    pp. 13208-13217. arXiv:1911.07053.
17. Douillard, A., Cord, M., Ollion, C., Robert, T., Valle, E. PODNet:
    Pooled Outputs Distillation for Small-Tasks Incremental Learning.
    ECCV 2020, pp. 86-102. arXiv:2004.13513.
18. Yan, S., Xie, J., He, X. DER: Dynamically Expandable Representation
    for Class Incremental Learning. CVPR 2021, pp. 3014-3023.
    arXiv:2103.16788.
19. Buzzega, P., Boschini, M., Porrello, A., Calderara, S. Rethinking
    Experience Replay: A Bag of Tricks for Continual Learning. ICPR 2020,
    pp. 2180-2187. (Introduces DER and DER++.)
20. Boschini, M., Bonicelli, L., Buzzega, P., Porrello, A., Calderara, S.
    Class-Incremental Continual Learning Into the eXtended DER-Verse.
    IEEE TPAMI 45(5):5497-5512, 2023. doi:10.1109/TPAMI.2022.3206549.
    (Journal extension of the DER/DER++ line.)
21. Goswami, D., Liu, Y., Twardowski, B., van de Weijer, J. FeCAM:
    Exploiting the Heterogeneity of Class Distributions in Exemplar-Free
    Continual Learning. NeurIPS 2023. arXiv:2309.14062.
22. Petit, G., Popescu, A., Schindler, H., Picard, D., Delezoide, B. FeTrIL:
    Feature Translation for Exemplar-Free Class-Incremental Learning. WACV
    2023, pp. 3911-3920. arXiv:2211.13131.

## Corrections caught during verification

Recorded so that the manuscript does not repeat common citation errors:

- FeCAM is NeurIPS 2023 (arXiv:2309.14062), not WACV 2024.
- A-GEM is ICLR 2019 (arXiv comment: "Published as a conference paper at
  ICLR 2019"), not ICML 2019.
- Synaptic Intelligence is ICML 2017 (PMLR 70:3987-3995), not PNAS.
- The BiC implementation in this codebase corresponds to Wu et al.
  (Microsoft Research), CVPR 2019, pp. 374-382, arXiv:1905.13260. (An
  earlier draft claimed a second, Facebook-published paper of the same
  title with arXiv:1904.07756; that claim was NOT verified - arXiv:1904.07756
  is in fact an unrelated physics paper - and must not be cited.)
- MIR: author list corrected to Aljundi, Caccia (Lucas), Belilovsky,
  Caccia (Massimo), Lin, Charlin, Tuytelaars (per the arXiv record).
- iCaRL distillation uses per-class sigmoid binary cross-entropy WITHOUT a
  temperature parameter (Algorithm 3 in the paper), not LwF-style
  temperature-scaled softmax KL; the T = 2 in our protocol is LwF-style,
  not iCaRL's.
- "Transfer without Forgetting" (TwF) is a DIFFERENT paper (Boschini,
  Bonicelli, Porrello, Bellitto, Pennisi, Palazzo, Spampinato, Calderara,
  ECCV 2022, arXiv:2206.00388); it is not the DER++ line. DER++ is
  introduced in "Rethinking Experience Replay" (ICPR 2020, pp. 2180-2187)
  and extended in the eXtended DER-Verse TPAMI paper
  (doi:10.1109/TPAMI.2022.3206549). Note that arXiv:2201.00766 is the
  DER-Verse preprint itself (confirmed in the TwF reference list), not a
  separate DER++ paper.
- FeTrIL trains a linear classifier (not logistic regression) and
  synthesizes pseudo-features of PAST classes by translating NEW-class
  features (direction of translation is often misreported).
- iCaRL pagination: pp. 5533-5542 on IEEE Xplore vs pp. 2001-2010 on CVF
  Open Access; pick one convention and use it consistently.
- FeTrIL pagination: pp. 3911-3920 (CVF Open Access and the authors' own
  citation) vs pp. 3900-3909 on the IEEE CSDL listing; use 3911-3920.
- Rainbow Memory: the arXiv identifier is 2103.17230.
- GEM: proceedings pages 6467-6476 (NIPS 2017) per the ML Anthology /
  DBLP convention; note that the DER-Verse TPAMI reference list cites
  6470-6479 - verify against the final proceedings volume before
  submitting the bibliography.
- DER: proceedings pages 3014-3023 (CVPR 2021).

## Open gaps (unverified, do not cite yet)

Candidates for later inclusion that were NOT verified in this pass and
must not enter the manuscript until checked: RanPAC (NeurIPS 2023, feature
projection + prototypes), Co2L (TPAMI 2023, contrastive CIL), ER-ACE
(arXiv:2106.14497), SSRE / RPSNet (expanding-representation lines),
Adam-NSCL (NeurIPS 2021), RWalk (ECCV 2018, intransigence framing), and
the empirical-forgetting study of Goodfellow et al. (arXiv:1312.6211).

## Suggested Related Work structure for the manuscript

1. Class-incremental learning: define the setting; the fine-tuning failure;
   EWC/SI (regularization), GEM/A-GEM (gradient constraints), LwF
   (distillation) as the three non-replay families; state that this paper
   is replay-based.
2. Replay and exemplar selection: ER and tiny-memory results; iCaRL
   herding + select-once; GSS/MIR/Rainbow Memory as selection diversity
   work; GDumb; then position uniform fixed-total herding with per-task
   re-selection as the design point studied here.
3. Distillation and classifier/readout choices: LwF -> iCaRL KD -> LUCIR
   -> BiC/WA bias fixes -> PODNet/DER representation-level approaches ->
   FeCAM/FeTrIL prototype readouts; map each ablation in our grid to one
   of these.
4. Forgetting and stability metrics: ACC/BWT (GEM), average forgetting
   (tiny-memory line), why both are reported (LUCIR and successors).

Total word budget suggestion for the manuscript: Related Work section
should stay a survey, not a catalogue; about half a page per subsection
for a NeurIPS-style paper.
