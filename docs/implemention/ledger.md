# Implementation Ledger — Ghost Bank

> Phase-by-phase file manifest. Every directory and file that must be created.
> 7 phases, each produces a self-contained, testable checkpoint.

---

## Phase 1 — Project Skeleton

**Goal**: Package structure, dependencies, and toolchain configuration. Nothing runs yet, but the foundation is solid.

```
ghost-bank/
├── pyproject.toml                  # Project metadata, build system, tool configs
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # __pycache__, output/, *.egg-info, .env
├── .python-version                 # Python version pin (e.g. 3.11)
│
├── src/
│   └── __init__.py                 # Package marker
│
├── studies/
│   └── __init__.py                 # Package marker
│
├── verification/
│   └── __init__.py                 # Package marker
│
├── tests/
│   └── __init__.py                 # Package marker
│
├── docs/
│   └── implemention/
│       ├── final_implementation_plan.md   # (already exists)
│       └── ledger.md                      # ← This file
│
└── output/                         # Created at runtime; add to .gitignore
```

**Total: 9 files**

**Checkpoint**: `pip install -e .` succeeds. `python -c "import src"` works.

---

## Phase 2 — Scientific Core

**Goal**: All reusable logic — datasets, models, bank core, loss functions, utilities.
Zero PyTorch Lightning dependency. Testable in isolation.

```
src/
├── data/
│   ├── __init__.py                         # Dataset registry
│   ├── base/
│   │   ├── __init__.py
│   │   ├── dataset.py                      # BaseDataset (ABC, torch Dataset)
│   │   └── datamodule.py                   # BaseDataModule (ABC, LightningDataModule)
│   └── synthetic/
│       ├── __init__.py
│       ├── dataset.py                      # GaussianDataset — generate_gaussian_data()
│       ├── datamodule.py                   # SyntheticDataModule — train/test split
│       └── defaults.py                     # SyntheticConfig (@dataclass)
│
├── models/
│   ├── __init__.py                         # Model registry
│   ├── base/
│   │   ├── __init__.py
│   │   └── model.py                        # BaseModel (ABC, nn.Module)
│   └── classifier/
│       ├── __init__.py
│       ├── model.py                        # MLPClassifier — predict(), softmax()
│       └── defaults.py                     # MLPConfig (@dataclass)
│
├── bank/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py                         # AbstractGhostBank: store(), query(), update()
│   │   ├── exposure.py                     # ExposureTracker: record(), debt(), reset()
│   │   ├── allocator.py                    # allocate_by_debt(debt, budget) → list[int]
│   │   └── retrieval.py                    # sample_static(), sample_weighted()
│   └── strategies/
│       ├── __init__.py
│       ├── static.py                       # StaticReplayBank — uniform random retrieval
│       └── ed_gb.py                        # ExposureDebtGhostBank — debt-driven retrieval
│
├── loss/
│   ├── __init__.py
│   ├── base.py                             # BaseLoss (ABC)
│   ├── focal/
│   │   ├── __init__.py
│   │   ├── loss.py                         # FocalLoss
│   │   └── defaults.py                     # FocalConfig (@dataclass)
│   ├── class_balanced/
│   │   ├── __init__.py
│   │   ├── loss.py                         # ClassBalancedLoss
│   │   └── defaults.py                     # ClassBalancedConfig (@dataclass)
│   └── ldam/
│       ├── __init__.py
│       ├── loss.py                         # LDAMLoss
│       └── defaults.py                     # LDAMConfig (@dataclass)
│
└── utils/
    ├── __init__.py
    ├── metrics.py                          # balanced_accuracy, macro_f1, minority_recall
    └── logging.py                          # Logger factory
```

**Total: 37 files** (16 dirs, 21 `.py` files incl `__init__`)

**Checkpoint**:
```python
from src.data.synthetic import GaussianDataset, SyntheticDataModule
from src.models.classifier import MLPClassifier
from src.bank.core import ExposureTracker, allocate_by_debt
from src.bank.strategies import StaticReplayBank, ExposureDebtGhostBank
from src.loss.focal import FocalLoss
from src.utils.metrics import balanced_accuracy
```
All imports succeed. Bank core functions produce correct numerical output.

---

## Phase 3 — Training Infrastructure

**Goal**: Methods (the bridge between loss and bank) + PyTorch Lightning module.
Now we can train.

```
src/
├── methods/
│   ├── __init__.py                         # Method registry
│   ├── base.py                             # Abstract Method: compute_loss(batch, bank, pl_module)
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── method.py                       # BaselineMethod — plain CE
│   ├── static_bank/
│   │   ├── __init__.py
│   │   └── method.py                       # StaticBankMethod — CE + static replay
│   ├── ed_gb/
│   │   ├── __init__.py
│   │   └── method.py                       # EDGBMethod — CE + debt-driven bank
│   ├── focal_loss/
│   │   ├── __init__.py
│   │   └── method.py                       # FocalLossMethod — focal loss only
│   └── class_balanced/
│       ├── __init__.py
│       └── method.py                       # ClassBalancedMethod — CB loss only
│
└── training/
    ├── __init__.py
    ├── pl_module.py                        # GhostBankLightningModule
    └── callbacks.py                        # DebtCurveLogger, ExposureTrackerCallback
```

**Total: 17 files** (7 dirs, 10 `.py` files)

**Checkpoint**:
```python
from src.training import GhostBankLightningModule
module = GhostBankLightningModule(
    model=MLPClassifier(...),
    method=EDGBMethod(...),
)
# module can forward a batch
```

---

## Phase 4 — Configuration System

**Goal**: Complete Hydra config tree. Every component has a config file.
Config-driven experiment composition works end-to-end.

```
configs/
├── config.yaml                             # Root — defaults list, launcher, logger
│
├── data/
│   ├── synthetic.yaml                      # GaussianDataset hparams
│   └── cifar_lt.yaml                       # CIFAR-LT hparams (stub for future)
│
├── model/
│   ├── classifier.yaml                     # MLPClassifier hparams
│   └── resnet.yaml                         # ResNet hparams (stub for future)
│
├── method/
│   ├── baseline.yaml
│   ├── static_bank.yaml
│   ├── ed_gb.yaml
│   ├── focal_loss.yaml
│   └── class_balanced.yaml
│
├── bank/
│   ├── static.yaml                         # StaticReplayBank hparams
│   └── ed_gb.yaml                          # ExposureDebtGhostBank hparams
│
├── training/
│   ├── default.yaml                        # Shared: batch_size, lr, epochs, optimizer
│   └── lightning.yaml                      # PL Trainer: accelerator, precision, devices
│
├── runner/
│   ├── synthetic.yaml                      # Preset: data=synthetic, model=classifier, method=ed_gb
│   ├── baseline.yaml                       # Preset: multi-run all methods
│   ├── ablation.yaml                       # Preset: vary bank params
│   └── stress_test.yaml                    # Preset: edge cases (zero budget, full budget, etc.)
│
└── output/
    ├── default.yaml                        # Output root dir, format preferences
    └── formats.yaml                        # Enabled writers: csv, jsonl, md
```

**Total: 20 files** (7 dirs, 13 `.yaml` files)

**Checkpoint**:
```bash
python -c "
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir='configs/'):
    cfg = compose('config', overrides=['method=ed_gb'])
    print(cfg.method.name)  # 'ed_gb'
"
```

---

## Phase 5 — Output System

**Goal**: Centralized output generation with state machine, format registry, and consistent directory structure.

```
studies/
└── output/
    ├── __init__.py
    ├── state_machine.py                    # OutputState enum + transitions
    ├── manager.py                          # OutputManager — single entry point
    ├── writer.py                           # BaseWriter ABC + FORMAT_REGISTRY
    ├── formatters/
    │   ├── __init__.py
    │   ├── csv_writer.py                   # CSV format
    │   ├── jsonl_writer.py                 # JSONL format
    │   └── markdown_writer.py              # Markdown table format
    └── defaults.py                         # OutputConfig (@dataclass)
```

**Total: 10 files** (3 dirs, 7 `.py` files)

**Checkpoint**:
```python
from studies.output import OutputManager
mgr = OutputManager(experiment="test", base_dir="output/")
mgr.initialize()
mgr.save_config({"foo": 1})
mgr.write_metrics({"loss": 0.5})
mgr.finalize({"accuracy": 0.95})
mgr.complete()
# output/test/<timestamp>/ exists with configs/, metrics/, results/
```

---

## Phase 6 — Runners

**Goal**: All experiment entry points. Cross-platform pure Python runners that wire components, configs, and output into complete experiments.

```
studies/
└── runner/
    ├── __init__.py
    ├── common/
    │   ├── __init__.py
    │   ├── base_runner.py                  # AbstractRunner — lifecycle template method
    │   └── path_utils.py                   # Cross-platform path ops (os, sys only)
    ├── synthetic/
    │   ├── __init__.py
    │   └── run.py                          # Single synthetic experiment
    ├── baseline_matrix/
    │   ├── __init__.py
    │   └── run.py                          # Multi-method comparison under equal budget
    ├── ablation/
    │   ├── __init__.py
    │   └── run.py                          # Vary bank/capacity/budget params
    └── stress_test/
        ├── __init__.py
        └── run.py                          # Edge cases: zero minority, full budget, etc.
```

**Total: 14 files** (6 dirs, 8 `.py` files)

**Checkpoint**:
```bash
python studies/runner/synthetic/run.py
# → output/synthetic/<ts>/metrics/*.csv, results/final_metrics.json

python studies/runner/baseline_matrix/run.py --multirun method=baseline,static_bank,ed_gb
# → output/baseline_matrix/<ts>/results/comparison_table.csv
```

---

## Phase 7 — Tests & Verification

**Goal**: Comprehensive test coverage. Additional math verification beyond the original script.

```
tests/
├── test_exposure.py                        # ExposureTracker: record, debt, reset, edge cases
├── test_allocator.py                       # allocate_by_debt: budget bounds, zero debt, ties
├── test_retrieval.py                       # Retrieval: static random, debt-weighted
├── test_bank_strategies.py                 # StaticReplayBank, EDGBank integration
├── test_methods.py                         # Each method: compute_loss returns correct shape
├── test_output_manager.py                  # State machine transitions, crash recovery
└── test_runners.py                         # Runner: end-to-end Hydra composition

verification/
├── verify_formal_definition.py             # (existing — kept as-is)
└── verify_bank.py                          # Property-based: debt monotonicity, allocation invariants
```

**Total: 9 files** (0 dirs, 9 `.py` files)

**Checkpoint**:
```bash
pytest tests/ -v --cov=src --cov=studies
# All tests pass, >90% coverage on bank core
```

---

## Summary — All Phases

| Phase | Scope | Files | Dependencies |
|---|---|---|---|
| 1 | Project skeleton | 9 | None |
| 2 | Scientific core (data, models, bank, loss, utils) | 37 | Phase 1 |
| 3 | Training infrastructure (methods, training) | 17 | Phase 2 |
| 4 | Config system (Hydra YAMLs) | 20 | Phase 2, 3 |
| 5 | Output system (state machine, writers) | 10 | Phase 1 |
| 6 | Runners (experiment entry points) | 14 | Phase 2–5 |
| 7 | Tests & verification | 9 | Phase 2–6 |
| **Total** | | **116 files** | |

**Grand total**: 116 files across 7 phases.

---

##  notes

- **Phase 2 is the most important** — it contains the scientific logic that makes this project novel (exposure tracking, debt computation, budget allocation). Get this right before moving on.
- **Phase 4 and 5 are independent of each other** — they could be developed in parallel.
- **Phase 6 depends on all previous phases** — it ties everything together.
- **Phase 7 is continuous** — write tests as you go, not just at the end. But this phase marks the point where we achieve full coverage.
- **Every file listed must be created** — no file is optional or "add if needed." The architecture is intentional.
