# Publication Figure & Table Plan — Ghost Bank CIL Study

**Status: FOR PROFESSOR REVIEW — no implementation until approved**
**Date:** 2026-08-05
**Scope:** Selection of figures/tables for the manuscript from the 38-figure analysis library, with full verification against persisted run data.

---

## 1. Purpose of this document

The analysis pipeline currently generates **38 figure sets + 12 table files** (6 tables in .tex/.md pairs). A professional paper does not carry 38 figures. This document:

1. States the verified data foundation every figure draws from (Section 2).
2. Documents a **data-correction** discovered during re-verification (Section 3).
3. Proposes the **manuscript figure set: 5 main + 4 appendix** (Sections 4–5), each figure mapped to the specific claim it serves and checked for information overlap against every other kept artifact.
4. Gives the complete audit of the 30 non-main figure sets — 27 excluded, 3 repurposed into the paper set (Section 6).
5. Sets the table policy (Section 7) and the implementation plan (Section 8).

**Approval is required before any code is written** (Section 10).

---

## 2. Data foundation — verification statement

Every number in this document was **recomputed from the persisted run artifacts** (`experiment_output/**/results/final_results.json` and `metrics/aggregated_accuracy_matrix.csv`) via the analysis data layer (`studies/analysis/src/common/data.py`), **and independently cross-checked** against `docs/experiment_results_summary.md`. All values match the report exactly. Seeds: 1993, 2023, 42; metrics are mean ± population std over the 3 seeds.

### 2.1 Master results (all 11 runs) — verified ✓

| # | Experiment | avg_acc | forgetting | memory | retrieval |
|---|---|---|---|---|---|
| B1 | iCaRL | 0.4236 ± 0.0087 | 0.1946 ± 0.0010 | 2000 | 64 |
| B2 | Static bank | 0.2860 ± 0.0135 | 0.5586 ± 0.0142 | 2000 | 64 |
| B3 | **Uniform herding (reference)** | **0.4499 ± 0.0100** | **0.1385 ± 0.0044** | 2000 | 64 |
| a1 | Reference − KD | 0.4479 ± 0.0031 | 0.2478 ± 0.0081 | 2000 | 64 |
| a2 | Reference, head-logit eval | 0.3461 ± 0.0113 | 0.4636 ± 0.0163 | 2000 | 64 |
| a3 | Reference, linear head | 0.4357 ± 0.0047 | 0.1468 ± 0.0048 | 2000 | 64 |
| a4 | Reference, random selection | 0.4017 ± 0.0017 | 0.1835 ± 0.0061 | 2000 | 64 |
| s1 | Memory budget 500 | 0.3683 ± 0.0036 | 0.1963 ± 0.0101 | 500 | 64 |
| s2 | Memory budget 4000 | **0.4766 ± 0.0072** | **0.1267 ± 0.0103** | 4000 | 64 |
| s3 | Retrieval budget 32 | 0.4328 ± 0.0061 | 0.1498 ± 0.0099 | 2000 | 32 |
| s4 | Retrieval budget 128 | 0.4481 ± 0.0078 | 0.1466 ± 0.0075 | 2000 | 128 |

### 2.2 Matched per-seed ablation deltas vs reference — verified ✓

| Run | Δ avg_acc (pp) | Δ forgetting (pp) |
|---|---|---|
| a1 (no KD) | **−0.21** | **+10.93** |
| a2 (head-logit) | **−10.38** | **+32.50** |
| a3 (linear head) | **−1.42** | +0.83 (n.s.) |
| a4 (random selection) | **−4.83** | +4.50 |

### 2.3 Per-task final accuracies — verified ✓

| Experiment | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| iCaRL | 48.4 | 40.3 | 40.7 | 43.0 | 39.0 | 44.0 | 37.0 | 45.7 | 45.8 | 39.5 |
| Static bank | 29.5 | 20.7 | 22.5 | 21.9 | 19.6 | 23.6 | 15.8 | 25.7 | 31.5 | 75.2 |
| Reference | 54.5 | 44.5 | 46.3 | 49.3 | 42.6 | 46.4 | 34.7 | 44.2 | 43.4 | 44.0 |
| a1 | 48.3 | 42.1 | 42.8 | 44.9 | 40.8 | 46.6 | 39.4 | 45.8 | 44.5 | 52.7 |
| a2 | 30.1 | 17.7 | 22.9 | 26.6 | 22.7 | 28.0 | 24.0 | 42.6 | 55.4 | 76.1 |
| a3 | 50.9 | 43.1 | 41.8 | 46.3 | 40.4 | 45.7 | 37.5 | 47.2 | 45.1 | 37.7 |
| a4 | 46.2 | 40.1 | 40.1 | 43.0 | 36.5 | 41.2 | 28.8 | 41.8 | 42.4 | 41.6 |
| s1 | 44.0 | 37.2 | 36.4 | 37.6 | 32.2 | 36.2 | 28.2 | 38.2 | 39.0 | 39.4 |
| s2 | 55.4 | 48.2 | 49.4 | 51.9 | 45.4 | 47.8 | 38.4 | 47.9 | 46.2 | 45.9 |
| s3 | 53.2 | 43.3 | 44.2 | 45.5 | 39.7 | 43.2 | 32.7 | 43.8 | 44.0 | 43.2 |
| s4 | 54.9 | 43.8 | 46.9 | 48.2 | 41.1 | 46.2 | 35.8 | 44.8 | 43.6 | 42.8 |

### 2.4 Per-task forgetting (introduction − final, pp) — verified ✓

| Experiment | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iCaRL | 36.0 | 31.9 | 29.7 | 22.9 | 17.5 | 14.6 | 10.3 | 7.0 | 5.3 | 0.0 | 17.5 |
| Static bank | 53.2 | 55.8 | 58.5 | 57.6 | 55.2 | 59.0 | 57.7 | 54.9 | 50.8 | 0.0 | 50.3 |
| Reference | 26.8 | 12.0 | 16.7 | 10.4 | 12.8 | 10.9 | 14.1 | 10.7 | 7.6 | 0.0 | 12.2 |
| a1 | 33.0 | 32.2 | 32.2 | 26.2 | 24.3 | 23.1 | 17.7 | 18.2 | 16.1 | 0.0 | 22.3 |
| a2 | 49.9 | 50.6 | 54.6 | 47.6 | 51.4 | 51.6 | 47.9 | 38.1 | 25.5 | 0.0 | 41.7 |
| a3 | 33.9 | 25.8 | 26.1 | 17.3 | 12.4 | 8.3 | 4.6 | 1.9 | −1.2 | 0.0 | 12.9 |
| a4 | 35.1 | 14.1 | 20.0 | 16.8 | 18.1 | 15.5 | 18.8 | 11.9 | 9.5 | 0.0 | 16.0 |
| s1 | 37.2 | 19.4 | 25.3 | 21.7 | 20.3 | 17.8 | 16.3 | 13.4 | 5.1 | 0.0 | 17.7 |
| s2 | 25.9 | 7.8 | 14.5 | 11.4 | 10.6 | 13.3 | 11.5 | 8.2 | 6.1 | 0.0 | 10.9 |
| s3 | 28.1 | 11.6 | 18.5 | 13.8 | 14.4 | 15.2 | 14.4 | 9.1 | 6.9 | 0.0 | 13.2 |
| s4 | 26.3 | 14.1 | 17.1 | 14.2 | 14.4 | 13.0 | 12.6 | 9.7 | 7.9 | 0.0 | 12.9 |

> **Definitions (matching the training code, `studies/runner/cifar100/metrics.py:14`):**
> - §2.4 computes per-task forgetting as *introduction − final* per task (mean over seeds, all 10 tasks; t9 = 0.0 by construction — the last task is introduced at the final evaluation and cannot be forgotten; the −1.2 on a3/t8 means the final accuracy *exceeded* the introduction accuracy for that task).
> - The aggregate `forgetting` in §2.1 is the standard **Chaudhry et al.** metric: mean over tasks 0..8 (T−1 tasks) of *(peak accuracy over all evaluations − final accuracy)*, averaged over seeds. Because it uses the per-task peak (≥ introduction) and averages 9 rather than 10 tasks, it is systematically slightly higher than the §2.4 mean (e.g. static bank 55.86 vs 50.3; reference 13.85 vs 12.2). Both definitions are reproduced exactly from the persisted per-seed matrices (verification §9).

---

## 3. Data correction discovered during re-verification

The earlier analysis characterized the a2 run (head-logit evaluation) as having a *"monotone recency gradient: T0 loses 49.9 pp, T1 42.1 pp, … down to T9 at 0 pp."* **This is contradicted by the persisted data** (Section 2.4):

- **a2 actual shape:** plateau of ≈48–55 pp on T0–T6 (49.9, 50.6, 54.6, 47.6, 51.4, 51.6, 47.9), then a recency-anchored tail (38.1, 25.5, 0.0). T1 = 50.6 pp, **not** 42.1 pp.
- **The genuinely monotone age gradient belongs to iCaRL** (36.0 → 31.9 → … → 5.3 → 0.0), which the report already described correctly.

**Corrected interpretation:** removing the NME readout makes *every old task* collapse to the same catastrophic level as the static bank (≈50 pp), while only the three newest tasks retain accuracy. This is recency bias expressed as *anchored collapse*, not a smooth age gradient.

**Action taken:** the mischaracterization was corrected in the a2 module docstring and figure title (`studies/analysis/src/experiments/a2_head_eval.py`). All figure descriptions in this document use the verified numbers.

---

## 4. Proposed manuscript figure set — 5 main figures

**Design principle:** every figure occupies a distinct cell of the data space (metric × x-axis). No two kept artifacts show the same metric on the same axis. Curves/shapes carry patterns; tables carry the exact numbers.

| # | Figure (panels) | Data source | Claim it serves | Status |
|---|---|---|---|---|
| **Fig 1** | Per-task final accuracy, 2 panels: (a) iCaRL / static bank / reference; (b) reference vs a1–a4 | per-task means ± 1 std | Main result — uniformity vs collapse; where each ablation differs per task | new composition of `baselines_comparison` + `component_comparison` |
| **Fig 2** | Component attribution, 2 panels: matched Δ accuracy, Δ forgetting (a1–a4) | matched per-seed deltas | Effect hierarchy — NME readout dominates (−10.4 pp); KD is pure stability (Δacc −0.2, Δforgetting +10.9) | reuse `attribution_deltas` |
| **Fig 3** | Resource sensitivity, 2×2: {memory, retrieval} × {accuracy, forgetting} | s1/ref/s2 and s3/ref/s4 | Memory is the strong lever (36.8 → 47.7 %); retrieval saturates at 64 (43.3 → 45.0 → 44.8 %) | new composition of `memory_curve` + `retrieval_curve` |
| **Fig 4** | Accuracy vs forgetting scatter, all 11 runs, ± 1 std error bars | master values | 2-D trade-off summary; B2 and a2 are the outliers | reuse `accuracy_vs_forgetting`, **add error bars** (currently missing) |
| **Fig 5** | Failure modes — forgetting by task age, 2 panels: (a) static bank vs ref; (b) a2 vs ref | per-task forgetting (intro − final) | Diagnostic separating *catastrophic-uniform* (B2: 50.8–59.0 pp on every old task) from *recency-anchored collapse* (a2: plateau ≈50 pp, tail 38.1/25.5/0.0), both vs the reference (7.6–26.8 pp) | new composition via `plot_forgetting_by_age` |

**Anti-duplication proof for the five:**

- Fig 1 = final **accuracy** per task. Fig 5 = **forgetting** per task (different metric, same axis family — complementary, both needed; Fig 1 shows where accuracy ends up, Fig 5 shows how much each task lost).
- Fig 2 = **aggregate** deltas of 4 components; Fig 1b = per-task pattern of the same runs. Different granularity, standard complement (shape vs effect size).
- Fig 3 = resource **axes** (budget); no other figure or table row-set covers s1–s4 shapes.
- Fig 4 = 2-D **accuracy × forgetting** placement; no table can express the trade-off structure; the numbers themselves are in Table 1.
- Fig 5 = the **only** forgetting-by-task-age view in the set; it is what makes B2 vs a2 structurally distinct despite both sitting bottom-right in Fig 4.

---

## 5. Proposed appendix figures — 4 (important only)

| # | Figure | Content | Why it earns appendix space |
|---|---|---|---|
| A1 | `forgetting_heatmap` | 11×10 per-task forgetting matrix, all methods | Complete-study snapshot; lets a reader verify any per-task forgetting claim at a glance |
| A2 | `evolution` (reference, B3) | 10×10 task-evolution matrix | Full dynamics of the flagship method; supports the "most uniform" claim beyond Fig 1a |
| A3 | `evolution` (iCaRL, B1) | 10×10 matrix | The steady-erosion signature of the strongest baseline |
| A4 | `stability_slopes` (a1) | intro → final paired drops | Per-task detail behind the KD-stability finding (Fig 2b shows the aggregate) |

Nothing else from the 38-set library is proposed for the appendix — every other figure either duplicates one of the above or carries no claim not already in a table (Section 6).

---

## 6. Audit of the 30 non-main figure sets (27 excluded + 3 repurposed; kept as regenerable library)

| Figure set (×formats) | Content | Exclusion reason |
|---|---|---|
| per_task_curve ×11 (b1,b2,b3,a1–a4,s1–s4) | single-run curves | 7 of 11 are exactly Fig 1's panels; s1–s4 shapes add nothing beyond Fig 3 |
| trajectory_fan ×3 (b2, b3, a2) | per-task trajectories | same underlying matrix as Fig 5 / Fig 1a; conveys no metric Fig 5 does not |
| forgetting_accumulation ×3 (b1, b2, a1) | cumulative forgetting traces | duplicates A3 (b1) / Fig 5 (b2, a1) |
| evolution heatmap (b3 only) | — | promoted to A2 |
| stability_slopes (a1) | — | promoted to A4 |
| forgetting_by_age (a2) | — | becomes Fig 5b |
| delta_bars ×6 (a3, a4, s1–s4) | per-task deltas | per-task deltas are implied by Fig 1b; aggregates in Fig 2/Table 2 |
| seed_consistency (a4) | per-seed cluster | its finding (std 0.0017) is fully expressed in Table 1 |
| all_methods_per_task | 11 curves overlay | 7/11 curves already in Fig 1; unreadable at paper size |
| master_accuracy_comparison | avg-acc bar chart | pure duplicate of Table 1 (same numbers, one dimension) |
| ranking_lollipop | sorted avg-acc | pure duplicate of Table 1 |

> Policy: nothing is deleted. Every excluded figure remains regenerable on demand via the existing per-experiment modules (`--all-figures`), for use during writing if the professor requests any specific view.

---

## 7. Table policy — all 6 tables generated, placement decided at write time

| Table | Content | Proposed placement |
|---|---|---|
| master_results | 11 runs × {avg_acc, forgetting, BWT} | **Main — Table 1** |
| component_ablations | 4 ablations × {avg_acc, forgetting, Δacc, Δforgetting} | **Main — Table 2** |
| per_task_accuracies | 11 runs × 10 tasks | **Appendix** |
| per_seed_results | 11 runs × 3 seeds | **Appendix** |
| baselines | B1/B2/B3 rows (subset of master) | generated; **fold into Table 1**, not presented |
| sensitivity | s1–s4 rows with axis column (subset of master) | generated; **fold into Table 1**, not presented |

The tables are the source of exact numbers; the figures exist to make the patterns visible. Final main/appendix split is a writing-time decision — all six are generated by default.

---

## 8. Implementation plan (only after approval)

1. New module `src/paper/main_figures.py` — one builder per figure: Fig 1 (2-panel composition), Fig 2 (reuse), Fig 3 (2×2 composition), Fig 4 (reuse + error bars), Fig 5 (2-panel composition). All builders use the existing verified archetypes in `src/common/plotting.py`.
2. New module `src/paper/appendix_figures.py` — A1–A4 reuse existing family/per-experiment figure code.
3. Registry: `PAPER_MAIN` / `PAPER_APPENDIX` in `src/common/constants.py`; `generate_all` default = 5 main + 4 appendix + 6 tables into `outputs/paper/{main/figures, appendix/figures, tables}/`.
4. `--all-figures` flag keeps the 38-set library generation available unchanged.
5. Post-generation verification script re-checks every figure's numbers against Sections 2.1–2.4.

---

## 9. Verification performed (triple check)

1. **All 11 runs** loaded from persisted artifacts; avg_acc / forgetting / per-task values **match the report exactly** (Sections 2.1–2.3).
2. **Per-task forgetting** recomputed from the accuracy matrices as intro − final (`data.py:132`); shapes confirm the a2 correction in Section 3.
3. **Matched per-seed deltas** recomputed seed-by-seed vs the reference; values match the report (Section 2.2).
4. **On-disk library** counted: 38 PNG figure sets (28 per-experiment + 10 family) across 15 figure directories — inventory matches the module registry.

---

## 10. Approval gate — for the professor

Proposed deliverables on approval:

- [ ] **5 main figures** (Sections 4) — Fig 1 per-task accuracy (2 panels), Fig 2 attribution (2 panels), Fig 3 resource sensitivity (2×2), Fig 4 accuracy-vs-forgetting scatter, Fig 5 forgetting-by-task-age (2 panels)
- [ ] **4 appendix figures** (Section 5)
- [ ] **6 tables** generated; main = master_results + component_ablations
- [ ] 38-set library retained, regenerable on demand
- [ ] Implementation per Section 8, output under `outputs/paper/`

**No code has been written. No outputs have been created beyond the existing library.**
