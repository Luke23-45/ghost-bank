# Final Implementation Plan — Ghost Bank

> Architecture & design decisions for the Exposure-Debt Ghost Bank project.
> Stack: PyTorch Lightning, Hydra, OmegaConf.

---

## 1. High-Level Design Principles

| Principle | Rationale |
|---|---|
| **True modularity** | Every component lives in its own subdirectory. Adding a new dataset, model, method, or bank strategy means creating a new directory — no existing file is touched. |
| **Separation of concerns** | `src/` = reusable implementation. `studies/` = experiments and output. `configs/` = all configuration. `verification/` = math proofs. `tests/` = unit tests. |
| **Cross-platform runners** | No shell scripts. All runners are pure Python using `os` and `sys`. Runs identically on Windows, Linux, macOS. |
| **Config-driven experiments** | Hydra composes configs hierarchically. A runner config (e.g. `runner=baseline`) selects data, model, method, bank, and training configs with sensible defaults. CLI overrides always possible. |
| **Centralized output** | Every run writes to `output/<experiment>/<timestamp>/` with a strict subdirectory contract: `metrics/`, `configs/`, `results/`, `artifacts/`. An output state machine guarantees consistency even on failure. |

---

## 2. Complete Directory Structure

```
ghost-bank/
│
├── src/                                  # Shared implementation code
│   ├── __init__.py
│   │
│   ├── data/                             # Modular dataset implementations
│   │   ├── __init__.py                   # Dataset registry
│   │   ├── base/                         # Abstract base classes
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py                # BaseDataset (torch.utils.data.Dataset)
│   │   │   └── datamodule.py             # BaseDataModule (pl.LightningDataModule)
│   │   ├── synthetic/                    # Gaussian synthetic data
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py                # GaussianDataset
│   │   │   ├── datamodule.py             # SyntheticDataModule
│   │   │   └── defaults.py              # Hydra structured config dataclass
│   │   └── cifar_lt/                     # CIFAR long-tailed (future)
│   │       ├── __init__.py
│   │       ├── dataset.py
│   │       └── datamodule.py
│   │
│   ├── models/                           # Modular model architectures
│   │   ├── __init__.py                   # Model registry
│   │   ├── base/                         # Abstract base model
│   │   │   ├── __init__.py
│   │   │   └── model.py                 # BaseModel (nn.Module)
│   │   ├── classifier/                   # MLP for synthetic data
│   │   │   ├── __init__.py
│   │   │   ├── model.py                 # MLPClassifier
│   │   │   └── defaults.py              # Structured config
│   │   └── resnet/                       # ResNet-32 for CIFAR (future)
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── defaults.py
│   │
│   ├── bank/                             # Ghost Bank — core logic + strategies
│   │   ├── __init__.py
│   │   ├── core/                         # Building blocks (no PL dependency)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # Abstract GhostBank interface
│   │   │   ├── exposure.py              # Exposure ledger, debt D_c(t) computation
│   │   │   ├── allocator.py             # Proportional budget allocation (floor+remainder)
│   │   │   └── retrieval.py             # Retrieval primitives (static, debt-weighted)
│   │   └── strategies/                   # Concrete bank implementations
│   │       ├── __init__.py
│   │       ├── static.py                # StaticReplayBank — uniform random retrieval
│   │       └── ed_gb.py                 # ExposureDebtGhostBank — debt-driven retrieval
│   │
│   ├── methods/                          # Training method = loss + bank integration
│   │   ├── __init__.py
│   │   ├── base.py                      # Abstract Method: compute_loss(batch, bank)
│   │   ├── baseline/                     # Plain cross-entropy
│   │   │   ├── __init__.py
│   │   │   └── method.py
│   │   ├── static_bank/                  # CE + static replay bank
│   │   │   ├── __init__.py
│   │   │   └── method.py
│   │   ├── ed_gb/                        # CE + exposure-debt ghost bank
│   │   │   ├── __init__.py
│   │   │   └── method.py
│   │   ├── focal_loss/                   # Focal loss baseline
│   │   │   ├── __init__.py
│   │   │   └── method.py
│   │   └── class_balanced/               # Class-balanced loss baseline
│   │       ├── __init__.py
│   │       └── method.py
│   │
│   ├── loss/                             # Loss function implementations
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseLoss interface
│   │   ├── focal/                        # Focal Loss
│   │   │   ├── __init__.py
│   │   │   ├── loss.py
│   │   │   └── defaults.py
│   │   ├── class_balanced/               # Class-Balanced Loss
│   │   │   ├── __init__.py
│   │   │   ├── loss.py
│   │   │   └── defaults.py
│   │   └── ldam/                         # LDAM Loss (future)
│   │       ├── __init__.py
│   │       ├── loss.py
│   │       └── defaults.py
│   │
│   ├── training/                         # PyTorch Lightning integration
│   │   ├── __init__.py
│   │   ├── pl_module.py                 # LightningModule — dispatches to Method
│   │   └── callbacks.py                 # Custom callbacks (debt curve logger, etc.)
│   │
│   └── utils/                            # General-purpose utilities
│       ├── __init__.py
│       ├── metrics.py                   # balanced_accuracy, macro_f1, minority_recall
│       └── logging.py                   # Logging helpers
│
├── studies/                              # Experiments, runners, output
│   ├── __init__.py
│   │
│   ├── runner/                           # All experiment runners
│   │   ├── __init__.py
│   │   ├── common/                      # Shared runner infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── base_runner.py           # AbstractRunner — lifecycle template method
│   │   │   └── path_utils.py            # Cross-platform path resolution (os, sys only)
│   │   ├── synthetic/                   # Runner: controlled synthetic experiment
│   │   │   ├── __init__.py
│   │   │   └── run.py
│   │   ├── baseline_matrix/             # Runner: compare all methods under equal budget
│   │   │   ├── __init__.py
│   │   │   └── run.py
│   │   ├── ablation/                    # Runner: ablation studies
│   │   │   ├── __init__.py
│   │   │   └── run.py
│   │   └── stress_test/                 # Runner: stress tests
│   │       ├── __init__.py
│   │       └── run.py
│   │
│   └── output/                           # Centralized output generation system
│       ├── __init__.py
│       ├── state_machine.py             # Output lifecycle state machine
│       ├── manager.py                   # OutputManager — single entry point
│       ├── writer.py                    # Base writer + format registry
│       ├── formatters/
│       │   ├── __init__.py
│       │   ├── csv_writer.py
│       │   ├── jsonl_writer.py
│       │   └── markdown_writer.py
│       └── defaults.py                 # Default output configuration
│
├── configs/                              # Hydra configuration tree
│   ├── config.yaml                      # Root config — defaults list, launcher, logger
│   ├── data/                            # Dataset configs
│   │   ├── synthetic.yaml
│   │   └── cifar_lt.yaml
│   ├── model/                           # Model configs
│   │   ├── classifier.yaml
│   │   └── resnet.yaml
│   ├── method/                          # Method configs
│   │   ├── baseline.yaml
│   │   ├── static_bank.yaml
│   │   ├── ed_gb.yaml
│   │   ├── focal_loss.yaml
│   │   └── class_balanced.yaml
│   ├── bank/                            # Bank configs
│   │   ├── static.yaml
│   │   └── ed_gb.yaml
│   ├── training/                        # Training configs
│   │   ├── default.yaml                 # Shared training hparams
│   │   └── lightning.yaml               # PL Trainer config
│   ├── runner/                          # Runner-level experiment configs
│   │   ├── synthetic.yaml               # Preset: data=synthetic, model=classifier, method=all
│   │   ├── baseline.yaml                # Preset: compare all methods
│   │   ├── ablation.yaml                # Preset: vary bank params
│   │   └── stress_test.yaml             # Preset: edge cases
│   └── output/                          # Output configs
│       ├── default.yaml
│       └── formats.yaml
│
├── verification/                         # Formal math verification (standalone, no PL/Hydra)
│   ├── __init__.py
│   ├── verify_formal_definition.py      # Existing — kept as-is
│   └── verify_bank.py                   # Additional property-based tests
│
├── tests/                                # pytest unit tests
│   ├── __init__.py
│   ├── test_exposure.py
│   ├── test_allocator.py
│   ├── test_retrieval.py
│   └── test_output_manager.py
│
├── docs/                                 # Research documentation
│   ├── implemention/                    # Implementation plans
│   │   └── final_implementation_plan.md  # ← This file
│   ├── clean_problem_statement.md
│   ├── ... (other existing docs)
│
├── output/                               # Runtime output (gitignored)
│   └── <experiment_name>/
│       └── <timestamp>/
│           ├── metrics/                 # Training curves, per-step metrics
│           ├── configs/                 # Frozen config snapshot
│           ├── results/                 # Final metrics tables (csv, jsonl)
│           └── artifacts/              # Figures, checkpoints
│
├── pyproject.toml
└── requirements.txt
```

---

## 3. Module Design & Contracts

### 3.1 Data Module Contract

Each dataset directory under `src/data/` must export:

```
src/data/<name>/
├── __init__.py          → exposes Dataset, DataModule
├── dataset.py           → torch.utils.data.Dataset subclass
├── datamodule.py        → pl.LightningDataModule subclass
└── defaults.py          → @dataclass config for Hydra structured configs
```

**Adding a new dataset** = create one new directory with these 4 files. No other file in the codebase is modified.

### 3.2 Model Module Contract

Each model directory under `src/models/` must export:

```
src/models/<name>/
├── __init__.py          → exposes Model class
├── model.py             → nn.Module subclass
└── defaults.py          → @dataclass config
```

### 3.3 Bank Module Contract

```
src/bank/
├── core/
│   ├── base.py          → AbstractGhostBank: store(), query(), update()
│   ├── exposure.py      → ExposureTracker: record(), debt(), reset()
│   ├── allocator.py     → allocate_by_debt(debt, budget) → list[int]
│   └── retrieval.py     → sample_static(), sample_weighted()
└── strategies/
    ├── static.py        → StaticReplayBank(AbstractGhostBank)
    └── ed_gb.py         → ExposureDebtGhostBank(AbstractGhostBank)
```

Key design: `core/` contains pure functions and base classes with **no PyTorch Lightning dependency**. `strategies/` composes core components into concrete bank implementations.

### 3.4 Method Contract

Each method under `src/methods/` implements:

```python
class Method(ABC):
    cfg: MethodConfig  # Hydra structured config
    
    @abstractmethod
    def compute_loss(
        self,
        batch: tuple[Tensor, Tensor],
        bank: AbstractGhostBank | None,
        pl_module: pl.LightningModule,
    ) -> Tensor:
        ...
```

The `LightningModule` calls `method.compute_loss(batch, bank, self)` — the method has full access to the model, current step, and bank state.

**Adding a new method** = create `src/methods/<name>/method.py` implementing this interface + add a YAML to `configs/method/`.

---

## 4. Config Resolution State Machine

Hydra composes configs in a strict order. The resolution pipeline:

```
                         ┌──────────────┐
                         │  config.yaml  │
                         │  (root)       │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  RUNNER_SELECT│  e.g. runner=baseline
                         │  loads from  │
                         │  configs/    │
                         │  runner/     │
                         └──────┬───────┘
                                │  selects:
                                │  data, model, method,
                                │  bank, training defaults
                         ┌──────▼───────┐
                         │  COMPOSE      │  Resolve each component
                         │               │  from configs/<type>/
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  MERGE        │  defaults < runner < CLI
                         │               │  (Hydra native)
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  VALIDATE     │  OmegaConf structured
                         │               │  config validation
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  FINAL        │  Frozen, read-only config
                         │               │  for the experiment
                         └──────────────┘
```

**Usage examples:**

```bash
# Run synthetic experiment with all defaults
python studies/runner/synthetic/run.py

# Run baseline matrix with a specific imbalance ratio
python studies/runner/baseline_matrix/run.py \
    data.synthetic.imbalance_ratio=100

# Override method and bank simultaneously
python studies/runner/baseline_matrix/run.py \
    method=ed_gb bank.ed_gb.retrieval_budget=16

# Multi-run sweep via Hydra
python studies/runner/baseline_matrix/run.py \
    --multirun method=baseline,static_bank,ed_gb \
    data.synthetic.imbalance_ratio=10,50,100
```

---

## 5. Output System Design

### 5.1 Directory Contract

Every run produces:

```
output/<experiment_name>/<YYYYMMDD_HHMMSS>/
├── configs/
│   └── resolved_config.yaml          # Frozen, resolved config snapshot
├── metrics/
│   ├── train_metrics.csv             # Streaming per-step/epoch metrics
│   ├── val_metrics.csv
│   └── debt_curves.csv               # Per-class exposure debt over time
├── results/
│   ├── final_metrics.json            # Aggregated final results
│   ├── comparison_table.csv          # If multi-method run
│   └── summary.jsonl                 # Row-per-run summary
└── artifacts/
    ├── debt_curves.png               # Exposure debt plots
    ├── confusion_matrix.png
    └── checkpoint.ckpt               # Best model checkpoint
```

### 5.2 Output State Machine

```
                    ┌─────────────┐
                    │ INITIALIZED  │  Create output dir, resolve all paths
                    └──────┬──────┘
                           │
                    ┌──────▼───────┐
                    │ CONFIG_SAVED  │  Save frozen config to configs/
                    └──────┬───────┘
                           │  training starts
                    ┌──────▼────────┐
              ┌─────┤ METRICS_OPEN  │◄────┐
              │     │ (streaming)   │     │  checkpoint
              │     └──────┬────────┘     │  every N steps
              │            │  training ends
              │     ┌──────▼──────────┐
              │     │ RESULTS_WRITTEN  │  Final metrics to results/
              │     └──────┬──────────┘
              │            │
              │     ┌──────▼──────────┐
              │     │ ARTIFACTS_SAVED  │  Plots, checkpoints
              │     └──────┬──────────┘
              │            │
              │     ┌──────▼──────┐
              │     │  COMPLETED   │  Success
              │     └─────────────┘
              │
              └── Any state → FAILED → COMPLETED (partial)
```

Each state transition is **atomic and idempotent**. If the process crashes mid-run, re-inspecting the output directory reveals exactly which state was reached.

### 5.3 Writer Registry

```python
# studies/output/writer.py
FORMAT_REGISTRY: dict[str, type[BaseWriter]] = {
    "csv": CSVWriter,
    "jsonl": JSONLWriter,
    "md": MarkdownWriter,
}

class BaseWriter(ABC):
    @abstractmethod
    def write(self, data: dict, path: str) -> None: ...
```

Adding a new output format = create a formatter and register it. No other code changes.

---

## 6. Runner Lifecycle

Each runner in `studies/runner/<name>/run.py` follows this template:

```python
# studies/runner/common/base_runner.py
class AbstractRunner(ABC):
    def run(self) -> None:
        # 1. Initialize Hydra config (via @hydra.main or compose API)
        # 2. Initialize OutputManager → creates output dir, captures config
        # 3. Initialize DataModule, Model, Method, Bank from config
        # 4. Initialize LightningModule + PL Trainer
        # 5. trainer.fit()
        # 6. OutputManager.finalize() → writes results, artifacts
        # 7. OutputManager.complete()
```

**Runners are pure Python** — no shell scripts, no subprocess calls to other scripts. Cross-platform path handling via `os.path.join()`.

---

## 7. Adding a New Component (Extensibility Guide)

### New Dataset
1. Create `src/data/<name>/` with `dataset.py`, `datamodule.py`, `defaults.py`
2. Add `configs/data/<name>.yaml`
3. If existing runner should use it: update runner config default

### New Model
1. Create `src/models/<name>/` with `model.py`, `defaults.py`
2. Add `configs/model/<name>.yaml`

### New Method
1. Create `src/methods/<name>/method.py` implementing `Method.compute_loss()`
2. Add `configs/method/<name>.yaml`

### New Bank Strategy
1. Create strategy in `src/bank/strategies/<name>.py`
2. Add `configs/bank/<name>.yaml`

### New Runner
1. Create `studies/runner/<name>/run.py`
2. Add `configs/runner/<name>.yaml`

---

## 8. Existing Code Migration

| Current File | New Location |
|---|---|
| `empirical/synthetic/run_controlled_synthetic.py` | Split into: `src/data/synthetic/`, `src/bank/`, `src/models/classifier/`, logic for runner goes to `studies/runner/synthetic/` |
| `verification/verify_formal_definition.py` | Keep in place, add `verification/verify_bank.py` |
| `docs/*.md` | Keep in place, add `docs/implemention/` |

---

## 9. Critical Design Decisions (Do Not Lose)

1. **Data + Model are decoupled.** Dataset never imports a model. Model never imports a dataset. The runner wires them together via config.

2. **Bank is PL-agnostic.** `src/bank/core/` has zero PyTorch Lightning imports. Only `src/training/` and `src/methods/` depend on PL.

3. **Methods own the loss computation.** The LightningModule does not hardcode loss logic — it delegates to the selected `Method` object. This lets each method freely compose base loss + bank loss + auxiliary losses.

4. **Runner configs are presets, not code.** A runner config (e.g. `configs/runner/synthetic.yaml`) is just a YAML that sets defaults for `data`, `model`, `method`, `bank`, `training`. It does not import or run code.

5. **Output path is deterministic.** `output/<experiment>/<timestamp>/` where timestamp is generated once at init. All writers receive the same base path. No relative path guessing anywhere.

6. **No shell scripts.** `studies/runner/common/path_utils.py` handles all cross-platform path resolution with `os.path`. All entry points are `python some/runner/run.py`.

7. **Config validation at compose time.** Hydra structured configs (`@dataclass` with `omegaconf.MISSING`) catch missing or invalid parameters before any training code runs.

8. **Every component has a `defaults.py`.** This provides the canonical Hydra structured config for that component, serving as both documentation and validation schema.

---

## 10. Implementation Order

```
Phase 1 — Foundation
  ├── verification/           (already done, keep as-is)
  ├── src/data/base/          + src/data/synthetic/
  ├── src/models/base/        + src/models/classifier/
  ├── src/bank/core/          + src/bank/strategies/
  ├── src/loss/base/
  ├── src/utils/metrics.py
  └── tests/ for core logic

Phase 2 — Training Infrastructure
  ├── src/methods/base/       + baseline/ + static_bank/ + ed_gb/
  ├── src/training/           (LightningModule + callbacks)
  └── configs/                (all YAML files)

Phase 3 — Output System
  ├── studies/output/         (state machine, writers, manager)
  └── tests/test_output_manager.py

Phase 4 — Runners
  ├── studies/runner/common/  (base runner, path utils)
  ├── studies/runner/synthetic/run.py
  └── studies/runner/baseline_matrix/run.py

Phase 5 — Verification
  ├── Run verification/ against synthetic experiment
  ├── Replicate existing run_controlled_synthetic.py results
  └── Validate output system produces correct results
```
