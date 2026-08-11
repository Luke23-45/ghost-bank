# Table Plan (Corrected) — Ghost Bank CIL Study

**Status: IMPLEMENTED (2026-08-05) — T1–T3, A1–A6 generated and verified**
**Date:** 2026-08-05
**Supersedes:** Section 7 of `analysis_plan.md` (the 6-table policy). This document is the complete, verified table specification: **3 main tables + 6 appendix tables** covering **every** persisted field and every derived quantity used in the manuscript. Nothing is silently dropped — anything not tabulated is listed explicitly in Section 6 with its rationale.

**Every value previewed in this document was recomputed from the persisted artifacts** (`experiment_output/**/results/final_results.json`, `metrics/*.csv`, `metrics/seed_*_bank_sizes.json`, `configs/resolved_config.yaml`, `run_meta.json`) and cross-checked against `docs/experiment_results_summary.md`. All match.

---

## 1. Complete data inventory (source of truth)

Per run (11 runs; seeds 1993, 2023, 42):

| Artifact | Fields available | New coverage |
|---|---|---|
| `results/final_results.json` — aggregated | avg_acc ± std, forgetting ± std, **BWT ± std**, task_0..9 final acc ± std, epochs/task (71.0), wall_time_s ± std | T1, T2, T3, A2, A4, A5 |
| `results/final_results.json` — per_seed ×3 | avg_acc, forgetting, BWT, wall_time_s, task_0..9 final acc | A4 (metrics), A5 (time) |
| `metrics/aggregated_accuracy_matrix.csv` | 10×10 evolution matrix (mean) | artifact-only (§6), A2/A3 figures |
| `metrics/aggregated_accuracy_matrix_std.csv` | 10×10 std | artifact-only (§6) |
| `metrics/seed_{s}_accuracy_matrix.csv` ×3 | per-seed evolution matrices | **A3 std** (per-task forgetting σ), artifact-only |
| `metrics/seed_{s}_bank_sizes.json` ×3 | per-class exemplar quota per task | **A6** (verified identical across seeds) |
| `metrics/seed_{s}_task_classes.json` ×3 | class IDs per task | artifact-only (§6) |
| `configs/resolved_config.yaml` | memory_total, retrieval_budget, head, KD, optimizer, epochs, split_seed | T1 (budgets), **A1** (protocol) |
| `run_meta.json` | device, git commit, torch/python, total wall time | **A1/A5** |

Derived quantities (implemented in `studies/analysis/src/common/data.py`, verified here):
per-task forgetting = intro − final (`per_task_forgetting`, line 132) → **A3**; per-task final acc std (`final_task_stds`, line 84) → **A2**; matched per-seed deltas (`matched_deltas`, line 263) → **T2, T3**; per-task forgetting std from seed matrices (verified computable, e.g. reference σ = 3.7, 1.4, 0.9, 1.9, 2.0, 3.5, 2.3, 2.6, 3.4, 0.0 pp) → **A3**.

---

## 2. Main tables (3)

### Table 1 — Master results (all 12 runs)

| # | Experiment | Active budget | Retrieval | avg_acc (%) | forgetting (%) | BWT (%) |
|---|---|---|---|---|---|---|
| B1 | iCaRL | 2000 | 64 | 42.33 ± 1.20 | 24.87 ± 1.11 | −24.87 ± * |
| B2 | Static bank | 2000 | 64 | 28.60 ± 1.35 | 55.86 ± 1.42 | −55.86 ± * |
| B3 | **Uniform Herding** | 2000 | 64 | **44.00 ± 0.51** | **17.22 ± 0.43** | −16.91 ± * |
| a1 | Without KD | 2000 | 64 | 42.52 ± 0.61 | 28.88 ± 0.19 | −28.88 ± * |
| a2 | Head-logit evaluation | 2000 | 64 | 33.54 ± 0.39 | 47.58 ± 0.84 | −47.58 ± * |
| a3 | Linear head | 2000 | 64 | 43.22 ± 0.36 | 17.34 ± 0.68 | −17.33 ± * |
| a4 | Random selection | 2000 | 64 | 41.61 ± 0.53 | 18.38 ± 0.47 | −18.28 ± * |
| s1 | Active budget 500 | 500 | 64 | 33.95 ± 1.51 | 30.48 ± 0.54 | −30.48 ± * |
| s2 | Active budget 4000 | 4000 | 64 | **47.32 ± 0.24** | **13.31 ± 0.59** | −12.74 ± * |
| s3 | Retrieval 32 | 2000 | 32 | 42.47 ± 0.63 | 17.21 ± 0.47 | −16.87 ± * |
| s4 | Retrieval 128 | 2000 | 128 | 44.16 ± 0.59 | 17.81 ± 0.38 | −17.49 ± * |

*All 11 BWT σ values are present in `backward_transfer_std` (verified). Mem/Retr from `resolved_config.yaml` (`data.memory_total`, `method.retrieval_budget`).

**Changes vs the current `master_results.tex`:** adds memory/retrieval budget columns (currently absent; the plan's own §2.1 had them), adds BWT ± std (currently bare), expresses values in % (matching the report).

**Serves:** report §2 ranking; plan §2.1; Fig 4 (all 11 points); Fig 3 endpoints; the protocol-fairness argument (B1/B3/a* all NME).

### Table 2 — Within-method comparisons (4 variants vs main configuration, matched per-seed)

| # | Component removed | avg_acc (%) | forgetting (%) | Δ avg_acc (pp) | Δ forgetting (pp) | Sig (acc / fgt) |
|---|---|---|---|---|---|---|
| a1 | KD (kd_weight=0) | 42.52 ± 0.61 | 28.88 ± 0.19 | −1.48 | +11.66 | n.s. / sig |
| a2 | NME readout (head-logit eval) | 33.54 ± 0.39 | 47.58 ± 0.84 | −10.46 | +30.36 | sig / sig |
| a3 | Cosine-margin head (linear head) | 43.22 ± 0.36 | 17.34 ± 0.68 | −0.79 | +0.12 | marginal / n.s. |
| a4 | Herding selection (random bank) | 41.61 ± 0.53 | 18.38 ± 0.47 | −2.39 | +1.16 | sig / n.s. |

**Changes vs the current `component_ablations.tex`:** deltas at 2-dp matched precision — exact matched means: a1 −1.4800/+11.6593, a2 −10.4600/+30.3556, a3 −0.7867/+0.1222, a4 −2.3900/+1.1593, so at 2 dp a1 −1.48/+11.66, a2 −10.46/+30.36, a3 −0.79/+0.12, a4 −2.39/+1.16; adds significance flags (from report §5: significant Δacc for a2, a4; within-noise a1 acc; marginal a3 acc; forgetting flags as shown per row).

**Serves:** plan §2.2; Fig 2; the finding that distillation primarily affects retention and that NME readout is important in this configuration.

### Table 3 — Resource sensitivity (active budget and retrieval, with main-configuration anchors)

| # | Experiment | Axis | Value | avg_acc (%) | forgetting (%) | Δ avg_acc (pp) | Δ forgetting (pp) |
|---|---|---|---|---|---|---|---|
| s1 | Active budget 500 | Active budget | 500 | 33.95 ± 1.51 | 30.48 ± 0.54 | −10.05 | +13.26 |
| B3 | **Uniform Herding** | Active budget | 2000 | **44.00 ± 0.51** | **17.22 ± 0.43** | 0 | 0 |
| s2 | Active budget 4000 | Active budget | 4000 | **47.32 ± 0.24** | **13.31 ± 0.59** | +3.32 | −3.91 |
| s3 | Retrieval 32 | Retrieval | 32 | 42.47 ± 0.63 | 17.21 ± 0.47 | −1.53 | −0.01 |
| B3 | **Uniform Herding** | Retrieval | 64 | **44.00 ± 0.51** | **17.22 ± 0.43** | 0 | 0 |
| s4 | Retrieval 128 | Retrieval | 128 | 44.16 ± 0.59 | 17.81 ± 0.38 | +0.16 | +0.59 |

All deltas are matched per-seed vs the main configuration (verified: s1 −10.05/+13.26, s2 +3.32/−3.91, s3 −1.53/−0.01, s4 +0.16/+0.59).

**Changes vs the current `sensitivity.tex`:** adds the main-configuration anchors (the active-budget curve is 500/2000/4000 and the retrieval curve 32/64/128), adds matched deltas.

**Serves:** report §4.5 resource story; Fig 3; the "memory is the strong lever, retrieval saturates at 64" claim.

---

## 3. Appendix tables (6)

### A1 — Protocol and reproducibility (shared settings + environment)
Rows: dataset (CIFAR-100, 10 tasks × 10 classes, split_seed 13, probe/val splits 30/20), backbone (ResNet-18, base filters 64, dropout 0), head (cosine-margin, scale 30, margin 0.35, first-task imprinting; linear for a3), optimizer (SGD 0.1 / 0.9 / 5e-4, grad clip 1.0, no LR schedule, warmup_steps 0), epochs per task (70 configured, **71 recorded for all 33 seeds — off-by-one in the epoch counter, verified; see Open item 1, resolved**), batch 128, AMP 16-mixed, seeds, exemplar budgets, retrieval budgets, KD (weight 1.0, temp 2.0; off for a1), per-method evaluation protocol (NME for B1/B3/a1–a4; head-logit for B2), hardware (Tesla T4), software (torch 2.11.0+cu128, pytorch-lightning 2.6.5, python 3.12.13), git commits (B1/B2 `9dde4622`, B3 `3436665a`, a1–s4 `2444dcd1`; all runs `git_dirty: true`), wall-time total per run.
Source: `resolved_config.yaml` ×11, `run_meta.json` ×11. **New table** (protocol currently lives in report §1 prose). The resolved config is the canonical exhaustive record (num_workers, mean/std, bank floor, probe_enabled, logging and progress-bar settings are implementation defaults, not tabulated).
**Naming note:** appendix table numbers (A1–A6) are independent of the appendix *figure* numbers (Fig A1–A4) defined in `analysis_plan.md` §5 — any "A1 heatmap" reference here means **Fig A1** (forgetting heatmap), not this protocol table.

### A2 — Per-task final accuracies (11 × 10), mean ± std
Cells "54.5 ± 4.1" format (landscape). Means verified (plan §2.3); stds from `task_N_final_acc_std` (verified present for all 11 runs, e.g. reference t4 σ = 8.5 pp).
**Supersedes** `per_task_accuracies.tex` (adds the std block the current file omits).
Serves: Fig 1 numbers; the s1 "uniform degradation" claim now has a tabular home.

### A3 — Per-task forgetting (11 × 10), intro − final (pp), mean ± std
Means verified (plan §2.4); σ from per-seed matrices (verification method confirmed). **New table** — the single largest gap in the current set: it is the numeric twin of Fig 5 and Fig A1 (forgetting heatmap), and the evidence for the study's headline distinction (B2 catastrophic-uniform 50.8–59.0 pp vs a2 anchored collapse 47.7–55.1 pp plateau + 42.1/27.2/0.0 tail).
Serves: Fig 5, Fig A1, §3's corrected interpretation.

### A4 — Per-seed metrics (33 rows: 11 runs × 3 seeds)
Columns: # | Experiment | Seed | avg_acc (%) | forgetting (%) | BWT (%). Long format.
Verified: all 44 cells of the current `per_seed_results.tex` match; **adds per-seed forgetting and BWT** (present in JSON, previously untabulated; e.g. iCaRL forgetting 25.83/25.46/23.32, BWT −25.83/−25.46/−23.32).
Serves: seed-level reproducibility; the a4 consistency claim (σ 0.53) is now fully inspectable.
Note: per-seed *per-task* accuracies (33×10) remain artifact-only (§6) — too raw to tabulate; their stds are summarized in A2/A3.

### A5 — Compute cost (11 runs)
Columns: # | Experiment | wall_time_s (mean ± std) | seed 1993 | seed 2023 | seed 42 | device.
Verified means: B1 3649, B2 2756, B3 4635, a1 3446, a2 4332, a3 3957, a4 4337, s1 4237, s2 4309, s3 3517, s4 5286 s; per-seed wall times verified present for all 33 seeds.
**New table** — backs the report's "55–99 min/seed" runtime claim and the reproducibility appendix.

### A6 — Exemplar budget verification (bank sizes)
| Budget | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 50 | 25 | 16–17 | 12–13 | 10 | 8–9 | 7–8 | 6–7 | 5–6 | 5 |
| 2000 | 200 | 100 | 66–67 | 50 | 40 | 33–34 | 28–29 | 25 | 22–23 | 20 |
| 4000 | 400 | 200 | 133–134 | 100 | 80 | 66–67 | 57–58 | 50 | 44–45 | 40 |

Per-class quota = budget ÷ classes seen (floor rounding causes the ±1 pairs); values verified identical across seeds and runs at the same budget. This table proves the budget-fairness claim (all methods carry exactly the configured budget; a4 random ≠ herding is *selection*, not *capacity*) and documents the quota-capped exemplar schedule, including the "20/class final" statement. **New table.**

---

## 4. Coverage matrix — no section and no figure left without numeric backing

| Plan/report element | Backed by |
|---|---|
| Plan §2.1 master values | Table 1 |
| Plan §2.2 matched deltas | Table 2 |
| Plan §2.3 per-task accuracies | Appendix table A2 |
| Plan §2.4 per-task forgetting | A3 |
| Fig 1 (per-task accuracy) | A2 |
| Fig 2 (attribution) | Table 2 |
| Fig 3 (resource) | Table 3 |
| Fig 4 (acc × forgetting scatter) | Table 1 |
| Fig 5 (forgetting by age) | A3 |
| Fig A1 heatmap | A3 |
| A2/A3 evolution matrices | A2 + A3 + §6 artifact note |
| A4 stability slopes | A4 (per-seed) + A3 |
| Report §4.5 resource story | Table 3 |
| Report §1 protocol | A1 |
| Report §5 compute note | A5 |
| Report §2 per-seed accuracies | A4 |
| Budget fairness (20/class; quota cap) | A6 |

---

## 5. Supersession of the current 6-table set

| Current file | Fate |
|---|---|
| `master_results.tex/.md` | → Table 1 (adds budgets, BWT ± std, %) |
| `component_ablations.tex/.md` | → Table 2 (2-dp matched deltas, sig flags) |
| `per_task_accuracies.tex/.md` | → A2 (adds std) |
| `per_seed_results.tex/.md` | → A4 (adds forgetting, BWT) |
| `baselines.tex/.md` | generated; rows B1–B3 live in Table 1 — **not presented** |
| `sensitivity.tex/.md` | → Table 3 (adds reference anchors, deltas) |
| (new) | A1 protocol, A3 per-task forgetting, A5 compute, A6 bank sizes |

Total: 3 main + 6 appendix = **9 tables × 2 formats = 18 files** under `outputs/paper/tables/`.

---

## 6. Artifact-only items (deliberately not tabulated — stated openly)

1. **Full 10×10 evolution matrices** (mean, std, per-seed): 11 runs × 100 cells each — too large for a table; presented via A1/A2/A3 figures and the A2/A3 appendix tables' derived columns; machine-readable CSV remains the canonical artifact.
2. **Per-seed per-task accuracies** (33 × 10): summarized through A2/A3 stds; raw CSVs retained.
3. **Class-to-task mapping** (`task_classes.json`): reproducible from config `split_seed: 13`; public CIFAR-100 classes; noted in A1.
4. **Forgetting-accumulation / trajectory traces**: figure-level views (excluded from the paper set); numbers derivable from the matrix CSVs.
5. **Per-epoch training traces** (`seed_{s}_task_{t}/metrics.csv` ×30 per run): epoch-level test/acc, per-class accuracies, balanced_acc, macro_f1, train/val loss — ~393 rows per task. Raw optimization curves, too granular to tabulate and used by no figure or claim; CSVs retained as the reproducibility trail (see README in each run dir).

Nothing is deleted or hidden; each item has a stated reason.

---

## 7. Open items — ALL RESOLVED at implementation (2026-08-05)

1. **Epochs: config says 70 (`max_epochs`, `epochs_per_task`), all runs record 71.0.** RESOLVED: verified all 33 seeds × 11 runs record exactly 71.0 epochs (`data.py epochs_per_task()` asserts uniformity). It is an off-by-one in the epoch counter; Table A1 states "70 configured; 71 recorded".
2. **a4 matched Δforgetting = +1.1593 pp → +1.16 at 2 dp** RESOLVED: verified +1.159260; Table 2 carries +1.16; the a4 forgetting delta is not significant under the paired test (p = 0.19), so the flag is n.s.
3. **Report §5 significance** RESOLVED: all T2 rows computed with one consistent two-sided paired t-test (per-seed deltas vs 0): a1 acc p=0.1016 n.s. / fgt p=0.0014 sig; a2 p=0.0025/0.0001 sig/sig; a3 p=0.0820 marginal / p=0.8844 n.s.; a4 p=0.0197/0.1875 sig/n.s. Matches plan Table 2 flags; now asserted by `verify_paper.py`.
4. **BWT for a1–s4** RESOLVED: filled in Table 1 from `backward_transfer_mean` (verified); report text update tracked in `docs/experiment_results_summary.md`.
5. **B2 protocol note** RESOLVED: A1 states static bank uses head-logit eval (its native protocol).

---

## 8. Implementation & verification protocol (after approval only) — IMPLEMENTED

As built (all verified 2026-08-05):

1. Table generators live in `src/paper/tables.py` (replacing the removed `src/scripts/generate_tables.py`); T1–T3, A1–A6 are emitted in `.tex/.md` pairs under `outputs/paper/tables/`.
2. All cells read from `RunResult` / artifacts — no hard-coded numbers (A1/A5/A6 are fully data-driven).
3. `src/scripts/verify_paper.py` re-derives every cell from `final_results.json`/CSVs and diffs against embedded expected constants; any mismatch fails the run. **265/265 checks pass.**
4. Significance column filled by one paired test (two-sided `scipy.stats.ttest_1samp`, per-seed deltas vs zero; sig p<0.05, marginal p<0.10, else n.s.); expected flags asserted per row.
5. `analysis_plan.md` §7 references this document.
