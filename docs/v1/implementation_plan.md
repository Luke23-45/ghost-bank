# Implementation Plan

## Scope

This document translates [`docs/v1/final_plan.md`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/docs/v1/final_plan.md)
into an implementation plan for the current repository layout.

The goal is to keep the existing architecture intact where possible and add
only the minimum new code needed to test the final hypothesis:

> A held-out probe, measured on classes that never enter replay memory, can
> drive a replay policy that improves class-incremental learning under a fixed
> memory budget.

The implementation path is intentionally conservative:

- reuse the current `src/`, `studies/`, `configs/`, and `tests/` layout,
- avoid introducing a second parallel framework,
- keep the old PID scripts in `studies/analysis/` as archival references only,
- move all new work into a dedicated PTM/probe-guided branch of the codebase.

---

## 1. Existing Architecture That Stays

The following top-level areas already exist and should remain the backbone of
the project:

- `src/data/cifar100/`
- `src/models/resnet/`
- `src/bank/core/`
- `src/bank/strategies/`
- `src/methods/`
- `src/training/`
- `studies/runner/cifar100/`
- `studies/output/`
- `configs/`
- `tests/`

The current `studies/analysis/` scripts remain useful as historical evidence
and for regression checks, but they should not be the primary implementation
surface for the new method.

The existing runner pattern in
[`studies/runner/cifar100/run.py`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/runner/cifar100/run.py)
already shows the right orchestration style:

- Hydra composes the config,
- the runner instantiates data, model, bank, and method,
- training and evaluation are run per seed,
- results are written through `OutputManager`.

The new implementation should follow the same pattern.

---

## 2. New Directories To Add

Only a small number of new directories are needed.

### 2.1 `src/models/ptm/`

Purpose: PTM-based class-incremental models and lightweight adaptation.

Recommended files:

- `__init__.py`
- `model.py`
- `defaults.py`

What it should contain:

- a model wrapper around a pre-trained visual backbone,
- a feature extractor API that exposes embeddings for prototype-based or
  calibration-based classification,
- optional adapter or partial fine-tuning hooks,
- an interface compatible with the current incremental head-expansion flow.

Why this directory is needed:

- the current codebase is centered on a from-scratch ResNet;
- the final plan moves the project into the PTM regime because that is where
  recent CIL results are strongest [Zhou et al., 2023](https://arxiv.org/abs/2303.07338),
  [Zhang et al., 2023](https://arxiv.org/abs/2303.05118),
  [Zhou et al., 2024](https://arxiv.org/abs/2401.16386).

### 2.2 `src/methods/probe_guided/`

Purpose: the new research method.

Recommended files:

- `__init__.py`
- `method.py`
- `defaults.py`

What it should contain:

- the training objective for the probe-guided method,
- the logic that combines current-task learning, replay, and optional
  distillation,
- the replay-weight update rule that consumes held-out probe scores,
- any task-boundary state transitions needed to update memory allocation.

This directory is the main scientific contribution of the new implementation.

### 2.3 `studies/runner/cifar100/probe_guided/`

Purpose: the executable experiment entry point for the final method family.

Recommended files:

- `__init__.py`
- `run.py`
- `sweep.py` or `compare.py` if a separate multi-run wrapper is needed

What it should contain:

- the end-to-end experiment runner for the PTM/probe-guided method,
- seed management,
- baseline comparison orchestration,
- result aggregation,
- final experiment output writing.

This should be a dedicated runner rather than an expansion of the archival
analysis scripts.

### 2.4 `docs/v1/`

Purpose: project plan, implementation plan, and final research narrative.

Existing files:

- `final_plan.md`
- `implementation_plan.md`

What they should contain:

- `final_plan.md`: the research direction, hypothesis, and success criteria,
- `implementation_plan.md`: the file-level and phase-level execution plan,
- future updates should only change these documents if the implementation
  scope changes materially.

### 2.5 `tests/probe_guided/` or flat tests under `tests/`

Either:

- add a dedicated `tests/probe_guided/` package, or
- keep the tests flat in `tests/` if that is simpler for the repo style.

If a new test directory is created, it should contain:

- unit tests for held-out split construction,
- unit tests for probe scoring,
- unit tests for memory allocation invariants,
- end-to-end runner smoke tests.

---

## 3. New Files To Add Or Extend

This section is the concrete file map. Some of these are new files inside
existing directories rather than new directories.

### 3.1 Data

#### `src/data/cifar100/defaults.py`

Extend the CIFAR-100 config schema with:

- `probe_split_size`
- `val_split_size`
- `split_seed`
- `memory_total`
- `probe_enabled`

Purpose:

- make the data partition deterministic and reproducible,
- separate training, validation, probe, and test roles explicitly.

#### `src/data/cifar100/datamodule.py`

Extend the data module so it can:

- materialize class-wise train/val/probe partitions,
- expose a held-out probe loader per class,
- expose a combined evaluation loader for seen classes,
- keep the current CIFAR-100 task loader behavior intact.

The data module should remain the owner of dataset partitioning logic.

#### `src/data/cifar100/ingest.py`

No redesign is required, but the ingestion pipeline must preserve enough
structure to build deterministic class-wise splits and maintain raw image
storage for replay selection.

#### Optional helper file: `src/data/cifar100/splits.py`

If the split logic becomes too large for the data module, extract it here.
This file should contain only pure partitioning logic:

- deterministic per-class splits,
- serialization-friendly split manifest creation,
- helper utilities for probe/train/validation partitioning.

### 3.2 Models

#### `src/models/resnet/model.py`

Keep the current from-scratch ResNet implementation as the reference
baseline. It is still useful for controlled comparisons.

#### `src/models/ptm/model.py`

Implement the PTM wrapper. It should describe:

- backbone loading,
- frozen vs partially trainable modes,
- embedding extraction,
- head construction or prototype classifier compatibility,
- optional adapter/prompt branch if included later.

#### `src/models/ptm/defaults.py`

Define the structured config for:

- backbone name,
- pretrained weights source,
- layer freezing policy,
- embedding dimension,
- classifier mode,
- adapter settings.

### 3.3 Bank / Allocation

The current `src/bank/core/` package should be extended rather than replaced.

#### `src/bank/core/exposure.py`

Keep the current debt-tracking structure, but repurpose it to record:

- probe loss history,
- per-class forgetting scores,
- smoothed scores if needed,
- task-to-task score transitions.

#### `src/bank/core/allocator.py`

Extend allocation logic with:

- fixed-total budget distribution,
- floor-clipped class weights,
- uniform-plus-probe mixture allocation,
- renormalization after clipping.

This is where the probe signal becomes a replay quota.

#### `src/bank/core/retrieval.py`

Add replay sampling helpers that can:

- sample uniformly,
- sample according to per-class quotas,
- preserve deterministic behavior under a seed.

#### Optional new file: `src/bank/core/probe.py`

If the logic above becomes crowded, isolate probe scoring here:

- compute per-class probe loss,
- normalize forgetting scores,
- prepare replay weights for the allocator.

### 3.4 Methods

#### `src/methods/base.py`

Keep the current method interface as the dispatch point for training loss
composition.

#### `src/methods/probe_guided/method.py`

Implement the new method family:

- current-task supervised loss,
- replay loss for selected old classes,
- optional distillation against the previous snapshot,
- task-boundary update that refreshes replay weights from probe scores.

This file should not own data partitioning or output management.
It should only define the learning rule.

#### `src/methods/probe_guided/defaults.py`

Define method-level hyperparameters:

- replay mixing coefficient,
- probe smoothing,
- allocation temperature / sharpness,
- minimum per-class quota,
- distillation weight if used.

### 3.5 Training

#### `src/training/pl_module.py`

Keep the LightningModule as the execution shell.
It should continue to:

- forward batches,
- call the selected method,
- log metrics,
- expose hooks for task-boundary events.

#### `src/training/callbacks.py`

Extend logging hooks if needed to capture:

- probe score curves,
- replay allocation curves,
- forgetting summaries,
- calibration metrics.

### 3.6 Runners

#### `studies/runner/cifar100/probe_guided/run.py`

This is the main executable.

It should:

- compose the PTM/probe-guided config,
- set seeds,
- build the CIFAR-100 partitions,
- initialize model, method, and memory objects,
- run task-by-task training,
- compute held-out probe scores at task boundaries,
- update replay allocation,
- run final calibration,
- evaluate on the official test split,
- write per-seed and aggregated results.

#### Optional helper: `studies/runner/cifar100/probe_guided/sweep.py`

Only needed if a separate sweep driver is easier than a Hydra multirun.
It should run:

- uniform replay baseline,
- held-out probe replay,
- PTM baseline,
- calibration variants.

### 3.7 Configs

Use the existing `configs/` tree.
Add only the new YAMLs that correspond to the new method family.

Recommended files:

- `configs/model/ptm_resnet.yaml`
- `configs/method/probe_guided.yaml`
- `configs/runner/probe_guided.yaml`
- update `configs/data/cifar100.yaml` if it needs probe/validation fields
- update `configs/training/cifar100.yaml` if additional logging or checkpoint
  settings are required

No new config directory is required unless we later split the method family
into multiple sub-variants.

### 3.8 Tests

Add tests for the new research path.

Recommended files:

- `tests/test_probe_split.py`
- `tests/test_probe_allocator.py`
- `tests/test_ptm_model.py`
- `tests/test_probe_guided_method.py`
- `tests/test_runner_probe_guided.py`

These tests should verify:

- probe images never enter replay,
- allocations sum to the fixed total memory,
- per-class minimum quotas are preserved,
- PTM feature extraction works,
- runner smoke test composes and executes.

---

## 4. Experiment Setup

### 4.1 Benchmark

Primary benchmark:

- CIFAR-100
- 10 tasks
- 10 classes per task
- fixed total memory `M = 2000`
- class order: default order first, shuffled order as a later ablation

This benchmark choice is consistent with the class-incremental literature
and with the memory-budget emphasis in the surveys [Zhou et al., 2023](https://arxiv.org/abs/2302.03648)
and [Zhou et al., 2024](https://arxiv.org/abs/2401.16386).

### 4.2 Per-class data split

For each CIFAR-100 training class:

- `probe`: reserved before any training and never replayed,
- `validation`: used only for hyperparameter selection and calibration checks,
- `train`: used for SGD.

Recommended split:

- 30 probe images,
- 20 validation images,
- 450 training images.

The split should be deterministic given the split seed.

### 4.3 Training flow

For each task:

1. load the current task data,
2. compute current probe scores for all seen classes,
3. convert scores to replay weights,
4. expand the classifier or adapter if required,
5. train on current-task data plus replayed memory,
6. update memory selection,
7. calibrate the classifier if the method uses post-hoc calibration,
8. evaluate on all seen classes.

### 4.4 Output flow

Each run should write to the existing `studies/output/` framework with:

- resolved config snapshot,
- per-task metrics,
- per-seed metrics,
- aggregated metrics,
- optional calibration artifacts,
- optional score curves and allocation curves.

The output manager should remain the single source of truth for where results
are stored.

### 4.5 Evaluation metrics

Minimum metrics:

- average incremental accuracy,
- final average accuracy,
- average forgetting,
- backward transfer,
- per-class accuracy,
- probe-to-forgetting correlation,
- runtime,
- memory budget usage.

These metrics should map directly to the success criteria in
[`docs/v1/final_plan.md`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/docs/v1/final_plan.md).

---

## 5. How This Completes `final_plan.md`

The implementation is complete only when every final-plan section has an
artifact in code or output.

### 5.1 Goal section

Satisfied when:

- the PTM/probe-guided method exists,
- the runner can execute it end to end,
- the method is evaluated against iCaRL, uniform replay, and PTM baselines.

### 5.2 Hypothesis section

Satisfied when:

- held-out probe scores are logged,
- probe-to-forgetting correlation is measured,
- the correlation is reported over at least 3 seeds.

### 5.3 Method section

Satisfied when:

- `src/models/ptm/` exists,
- `src/methods/probe_guided/` exists,
- the bank allocator consumes probe scores,
- memory remains fixed-total and floor-clipped.

### 5.4 Experimental protocol section

Satisfied when:

- CIFAR-100 splits are deterministic,
- probe/validation/train roles are separated,
- shuffled-order ablations are runnable,
- result tables are written to output artifacts.

### 5.5 Success criteria section

Satisfied when:

- the final result table contains the main metrics,
- the method beats or matches the intended baselines,
- or, if it fails, the negative result is clean enough to defend.

### 5.6 Final narrative section

The final plan is “done” when the codebase can answer this question with
evidence rather than intuition:

> Does a held-out forgetting probe improve replay allocation enough to beat
> current baselines under realistic memory scarcity?

If yes, the paper is a positive method paper.
If no, the paper is a rigorous negative result or a methodologically useful
analysis.

---

## 6. Recommended Implementation Order

### Phase 1: protocol lock

1. define deterministic class splits,
2. create held-out probe partitions,
3. fix the total memory budget,
4. verify that probe images never flow into replay or calibration.

### Phase 2: PTM baseline

1. implement the PTM wrapper,
2. reproduce a frozen-prototype baseline,
3. add minimal adaptation if needed,
4. establish the non-adaptive performance floor.

### Phase 3: probe-guided replay

1. implement probe scoring,
2. implement score-to-allocation mapping,
3. integrate allocation into replay selection,
4. compare against uniform replay.

### Phase 4: calibration and evaluation

1. add post-hoc calibration,
2. log forgetting and correlation metrics,
3. run multi-seed sweeps,
4. produce final result tables and plots.

### Phase 5: paper readiness

1. freeze the winning configuration,
2. re-run with 3 seeds,
3. collect comparison tables,
4. update the final plan with the actual findings.

---

## 7. Non-goals

- Do not keep expanding the old PID family.
- Do not write new experiments inside `studies/analysis/` unless they are
  short-lived diagnostics.
- Do not introduce a second experiment framework.
- Do not change the repo into a generic benchmark zoo.

The project should remain focused on one question and one implementation path.

---

## 8. Reference Papers

The implementation plan is grounded in the following literature:

- iCaRL: [Rebuffi et al., 2017](https://arxiv.org/abs/1611.07725)
- Deep Class-Incremental Learning survey:
  [Zhou et al., 2023](https://arxiv.org/abs/2302.03648)
- Continual Learning with Pre-Trained Models survey:
  [Zhou et al., 2024](https://arxiv.org/abs/2401.16386)
- Revisiting Class-Incremental Learning with Pre-Trained Models:
  [Zhou et al., 2023/2024](https://arxiv.org/abs/2303.07338)
- SLCA:
  [Zhang et al., 2023](https://arxiv.org/abs/2303.05118)
- Balanced Destruction-Reconstruction Dynamics:
  [Zhou et al., 2023](https://arxiv.org/abs/2308.01698)
- Efficient Replay for PTM-based CIL:
  [Yin et al., 2024](https://arxiv.org/html/2408.08084v1)
- Dynamic Imbalanced Learning in CIL:
  [Zhou et al., 2024](https://arxiv.org/html/2405.15157v1)
- Class-Incremental Learning with Pre-trained Vision-Language Models:
  [Liu et al., 2023](https://arxiv.org/abs/2310.20348)

