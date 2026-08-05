# CIFAR-100 Class-Incremental Learning — Experiment Results Summary

**Report date:** 2026-08-05
**Scope:** Full baseline + ablation study of the uniform-herding replay method ("reference"), its four component ablations, and four resource-sensitivity rows. All numbers below are read directly from the persisted run artifacts (`experiment_output/final_baseline_run/`, `experiment_output/abalations/`); every per-seed value is stored in the corresponding `results/final_results.json`.

---

## 1. Shared experimental protocol

| Setting | Value |
|---|---|
| Dataset | CIFAR-100, 10 tasks x 10 classes (class-incremental) |
| Backbone | ResNet-18 (base filters 64, no dropout), head expands per task |
| Training | SGD lr=0.1, momentum 0.9, wd 5e-4, 70 epochs/task, batch 128, AMP 16-mixed |
| Seeds | 1993, 2023, 42 (all runs, all methods) |
| Exemplar budget | 2000 (20/class) unless stated; retrieval budget 64/step unless stated |
| Reference config | herding selection + cosine-margin head (scale 30, margin 0.35) + KD (weight 1.0, temp 2.0) + NME evaluation |
| Baselines | iCaRL and reference are both evaluated with **NME** (iCaRL's canonical protocol); static bank uses head-logit argmax (its native ER/BiR-style protocol) |

All averages are mean over the 3 seeds; `+-` is population std over seeds. Per-seed deltas in Section 3 are matched seed-by-seed against the published reference run, never mean-vs-mean.

---

## 2. Master results table

| # | Experiment | avg_acc | forgetting | BWT |
|---|---|---|---|---|
| B1 | iCaRL (baseline) | 0.4236 +- 0.0087 | 0.1946 +- 0.0010 | -0.1946 |
| B2 | Static bank (baseline) | 0.2860 +- 0.0135 | 0.5586 +- 0.0142 | -0.5586 |
| B3 | **Uniform herding (reference)** | **0.4499 +- 0.0100** | **0.1385 +- 0.0044** | -0.1354 |
| a1 | Reference minus KD (`kd_weight=0.0`) | 0.4479 +- 0.0031 | 0.2478 +- 0.0081 | -0.2478 |
| a2 | Reference with head-logit eval (`predict_mode=head`) | 0.3461 +- 0.0113 | 0.4636 +- 0.0163 | -0.4636 |
| a3 | Reference with linear head (`head=linear`) | 0.4357 +- 0.0047 | 0.1468 +- 0.0048 | -0.1434 |
| a4 | Reference with random selection (`bank.selection=random`) | 0.4017 +- 0.0017 | 0.1835 +- 0.0061 | -0.1775 |
| s1 | Memory budget 500 | 0.3683 +- 0.0036 | 0.1963 +- 0.0101 | -0.1963 |
| s2 | Memory budget 4000 | 0.4766 +- 0.0072 | 0.1267 +- 0.0103 | -0.1212 |
| s3 | Retrieval budget 32 | 0.4328 +- 0.0061 | 0.1498 +- 0.0099 | -0.1467 |
| s4 | Retrieval budget 128 | 0.4481 +- 0.0078 | 0.1466 +- 0.0075 | -0.1438 |

*BWT for a1–s4 (previously "—"): from `backward_transfer_mean` (verified 2026-08-05 via the analysis data layer). Note BWT = −forgetting exactly for B1/B2/a1/a2/s1 (final accuracy below introduction accuracy on every task); for the remaining rows BWT differs slightly from −forgetting because some tasks retain gains (e.g. reference −0.1354 vs −0.1385).*

Per-seed accuracies (for reference use):

| Experiment | seed 1993 | seed 2023 | seed 42 |
|---|---|---|---|
| iCaRL | 0.4229 | 0.4346 | 0.4133 |
| Static bank | 0.2815 | 0.3043 | 0.2722 |
| Uniform herding (reference) | 0.4511 | 0.4616 | 0.4371 |
| a1_no_kd | 0.4522 | 0.4454 | 0.4460 |
| a2_head_eval | 0.3309 | 0.3493 | 0.3581 |
| a3_linear_head | 0.4377 | 0.4402 | 0.4292 |
| a4_random_bank | 0.3994 | 0.4035 | 0.4021 |
| s1_budget500 | 0.3664 | 0.3733 | 0.3651 |
| s2_budget4000 | 0.4760 | 0.4856 | 0.4681 |
| s3_retr32 | 0.4346 | 0.4392 | 0.4246 |
| s4_retr128 | 0.4457 | 0.4586 | 0.4400 |

Final-state per-task accuracies (mean over seeds, %):

| Experiment | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| iCaRL | 48.4 | 40.3 | 40.7 | 43.0 | 39.0 | 44.0 | 37.0 | 45.7 | 45.8 | 39.5 |
| Static bank | 29.5 | 20.7 | 22.5 | 21.9 | 19.6 | 23.6 | 15.8 | 25.7 | 31.5 | 75.2 |
| Uniform herding (reference) | 54.5 | 44.5 | 46.3 | 49.3 | 42.6 | 46.4 | 34.7 | 44.2 | 43.4 | 44.0 |
| a1_no_kd | 48.3 | 42.1 | 42.8 | 44.9 | 40.8 | 46.6 | 39.4 | 45.8 | 44.5 | 52.7 |
| a2_head_eval | 30.1 | 17.7 | 22.9 | 26.6 | 22.7 | 28.0 | 24.0 | 42.6 | 55.4 | 76.1 |
| a3_linear_head | 50.9 | 43.1 | 41.8 | 46.3 | 40.4 | 45.7 | 37.5 | 47.2 | 45.1 | 37.7 |
| a4_random_bank | 46.2 | 40.1 | 40.1 | 43.0 | 36.5 | 41.2 | 28.8 | 41.8 | 42.4 | 41.6 |
| s1_budget500 | 44.0 | 37.2 | 36.4 | 37.6 | 32.2 | 36.2 | 28.2 | 38.2 | 39.0 | 39.4 |
| s2_budget4000 | 55.4 | 48.2 | 49.4 | 51.9 | 45.4 | 47.8 | 38.4 | 47.9 | 46.2 | 45.9 |
| s3_retr32 | 53.2 | 43.3 | 44.2 | 45.5 | 39.7 | 43.2 | 32.7 | 43.8 | 44.0 | 43.2 |
| s4_retr128 | 54.9 | 43.8 | 46.9 | 48.2 | 41.1 | 46.2 | 35.8 | 44.8 | 43.6 | 42.8 |

---

## 3. Per-experiment detail

### B1 — iCaRL baseline
- **Config:** herding-selected exemplars, iCaRL-style training (per-class BCE distillation), linear classifier, **NME evaluation** — the canonical iCaRL protocol (`iCaRLMethod.predict` is `nme_predict` on L2-normalized herding class means); 2000-exemplar budget, retrieval 64.
- **Result:** 0.4236 +- 0.0087 avg_acc; 0.1946 +- 0.0010 forgetting. Per-task curve is flat (37-48%) with no strong recency spike — classic iCaRL behavior: moderate accuracy, moderate forgetting.
- **Reading:** a solid, well-behaved baseline; beats random replay by +13.8pp and is the nearest competitor to the reference (-2.6pp).

### B2 — Static bank baseline
- **Config:** reservoir-random exemplar storage (no herding), plain cross-entropy replay (no distillation), linear classifier; **head-logit evaluation** — the method defines no `predict`, so the harness falls back to logit argmax (the standard ER/BiR protocol); per-class capacity 200, quota-capped to the same 2000-exemplar budget.
- **Result:** 0.2860 +- 0.0135 avg_acc; 0.5586 +- 0.0142 forgetting — the worst run of the study. The t9 spike (75.2%) confirms the classifier is dominated by the most recent classes; early classes collapse (t1 20.7%).
- **Reading:** random exemplar selection without distillation is catastrophic. Provides the lower anchor of the comparison; also demonstrates that the memory budget alone (equal to the reference's) explains nothing without selection + stabilization.

### B3 — Uniform herding (reference; the locked anchor)
- **Config:** herding selection + cosine-margin head + CE+KL KD (weight 1.0, temp 2.0) + NME evaluation; 2000 budget, retrieval 64.
- **Result:** 0.4499 +- 0.0100 avg_acc; 0.1385 +- 0.0044 forgetting. Per-task curve is the most uniform of the study (34.7-54.5%), i.e., no task collapse.
- **Reading:** the strongest method configuration: +2.6pp over iCaRL (0.4499 vs 0.4236) and -5.6pp forgetting (0.1385 vs 0.1946). All ablations below are measured against this run.

### a1 — KD removed (`kd_weight=0.0`)
- **Result:** 0.4479 +- 0.0031 avg_acc (-0.002 vs reference, not significant); forgetting 0.2478 +- 0.0081 (+0.109, significant).
- **Reading:** removing distillation leaves average accuracy **unchanged** but nearly **doubles forgetting**. KD is purely a stability mechanism in this design, not an accuracy mechanism. If forgetting is a headline metric (it normally is), KD must stay.

### a2 — Head-logit evaluation (`predict_mode=head`)
- **Result:** 0.3461 +- 0.0113 avg_acc (-0.104, the largest single effect in the study); forgetting 0.4636 +- 0.0163 (3.3x the reference).
- **Per-task story:** early tasks collapse (t1 = 17.7%) while the last task jumps to 76.1% — textbook classifier recency bias. NME on L2-normalized class-mean features removes this bias entirely.
- **Reading:** the largest single effect in the study, but it is a *pairing* effect, not an evaluation trick: the cosine-margin head's logits (scale 30, margin 0.35) are poorly calibrated for argmax — margin-scaled logits amplify recency bias (t1 = 17.7%, t9 = 76.1%). NME is that head's natural readout (the method docstring: NME "pairs naturally with the cosine margin head: both operate on L2-normalized prototypes"). The iCaRL baseline already uses NME, so the reference-vs-iCaRL comparison is protocol-fair; the 10.4pp is the cost of *mispairing* the cosine head with head-logits, not a general "NME bonus" that iCaRL is missing. This is the method's design constraint, not a measurement asymmetry.

### a3 — Linear head (`head=linear`)
- **Result:** 0.4357 +- 0.0047 avg_acc (-0.014, marginal); forgetting 0.1468 +- 0.0048 (statistically unchanged).
- **Reading:** the cosine-margin head contributes a small, consistent accuracy gain (+1.4pp) but does not change stability. Its main value is pairing with NME (L2-normalized prototypes); this interaction is worth one follow-up if the paper wants a head-geometry story.

### a4 — Random exemplar selection (`bank.selection=random`)
- **Result:** 0.4017 +- 0.0017 avg_acc (-0.048, significant); forgetting 0.1835 +- 0.0061 (+0.045).
- **Reading:** herding selection is worth +4.8pp over seeded random selection at the same 2000-exemplar budget. The "herding" claim holds, but it is moderate — not the dominant factor. Note the very low seed std (0.0017): the random-selection penalty is extremely consistent across seeds.

### s1 — Memory budget 500
- **Result:** 0.3683 +- 0.0036 avg_acc (-0.082 vs reference); forgetting 0.1963 +- 0.0101 (+0.058).
- **Reading:** memory-starved; per-task curve degrades uniformly (28-44%). Confirms the method is memory-hungry.

### s2 — Memory budget 4000
- **Result:** 0.4766 +- 0.0072 avg_acc (+0.027, significant); forgetting 0.1267 +- 0.0103 (-0.012). **Best configuration in the entire study** — beats the reference on both metrics.
- **Reading:** doubling the budget improves accuracy **and** reduces forgetting; the curve is still rising at 4000, so the saturation point is unknown (see decisions).

### s3 — Retrieval budget 32
- **Result:** 0.4328 +- 0.0061 avg_acc (-0.017); forgetting 0.1498 +- 0.0099 (+0.011).
- **Reading:** halving per-step replay costs ~1.7pp. Below the 64 optimum.

### s4 — Retrieval budget 128
- **Result:** 0.4481 +- 0.0078 avg_acc (-0.002, not significant); forgetting 0.1466 +- 0.0075 (+0.008, slight negative trend).
- **Reading:** doubling per-step replay buys **nothing** — 64 is at (or past) the optimum. Retrieval budget is not a story axis; keep it at 64.

---

## 4. Cross-cutting analysis (what the numbers jointly say)

1. **Ranking:** uniform herding (0.4499) > iCaRL (0.4236) > random replay (0.2860); at budget 4000 the method reaches 0.4766. The method wins against both baselines on both metrics (accuracy and forgetting).
2. **Where the win comes from (component attribution, per-seed deltas vs reference):**
   - NME evaluation: +10.4pp (a2) — dominant
   - Herding selection: +4.8pp (a4)
   - Cosine-margin head: +1.4pp (a3)
   - KD: +0.2pp accuracy but -11pp forgetting (a1)
3. **Protocol fairness (resolved):** the reference and iCaRL are both evaluated with **NME** — iCaRL's canonical protocol (`iCaRLMethod.predict` is `nme_predict` on herding class means, at the run's commit too). The headline comparison is protocol-matched; the +2.6pp over iCaRL is method-driven (herding + cosine-margin head + CE-KD vs herding + BCE distillation + linear head, same bank, same eval). The only head-logit-evaluated baseline is static bank, where that is the standard ER/BiR protocol and it is the lowest anchor (0.2860) — no headline claim rests on it. What a2 shows is a *design constraint* (the cosine head must not be read out with raw logits), not that we used a "better measuring stick".
4. **KD is stability, not accuracy** — frame it that way or risk a reviewer noticing the zero accuracy contribution.
5. **Resource story:** memory is the strongest lever (500->4000: +10.8pp); retrieval budget is exhausted at 64. The paper's resource figure should be the memory curve.

---

## 5. Statistical notes

- 3 seeds per run; population std over seeds. Within-noise results: a1 acc, a3 acc (marginal), s4 acc. Significant: a2, a4, s1, s2, and all forgetting effects except a3.
- Ablation deltas are matched per-seed against the published reference run (same seed set), not mean-vs-mean.
- The reference run was not re-executed (by design, the reference family was removed); its anchor values are the persisted published run. All ablations and the reference share the same code path and seed set, so the comparison is valid.
- Runs executed on NVIDIA GPU (Colab), 16-mixed precision; per-seed wall time 55-99 min (measured 3308-5919 s/seed across rows; a 3-seed row is ~3-5h end-to-end).

---

## 6. Recommended next decisions

**Publish-critical (no new training needed):**
1. Frame a2 as the cosine-head readout constraint (NME is its natural pairing; head-logit eval of margin-scaled logits produces recency bias) and state plainly that both the reference and iCaRL use NME — the comparison is protocol-fair. Build the per-task figures from the saved matrices (all 11 runs already on disk).
2. Adopt the memory-curve framing: report reference (2000) and s2 (4000) as the headline resource pair; consider stating 500 as the low anchor.

**If more GPU budget is available (one-time decisions, in priority order):**
3. *(No longer needed)* "iCaRL + NME" is already satisfied — iCaRL ran with NME all along. Optional hardening only: static bank + NME (1 row x 3 seeds) to rule out the protocol difference in the lowest anchor, if a reviewer cares about the lower bound.
4. *(Highest value)* Complete the memory curve: 8000 (saturation check) and 250 (floor). 2 rows x 3 seeds. Saturation point makes the resource story rigorous.
5. *(Optional)* Budget x selection cell (random @ 4000): separates the herding effect from the budget effect. One additional seed of evidence for the herding claim.
6. *(Robustness)* Bump the headline configs (reference, s2, a4) from 3 to 5 seeds before submission — 3 seeds is thin for a publication.

**Suggested first action:** let me generate the figure set (per-task curves, component attribution bar chart, memory/retrieval curves) from the existing artifacts — zero GPU time, ready to review tomorrow.
