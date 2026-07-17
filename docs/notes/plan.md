# Experiment Commands

> Use `PYTHONIOENCODING=utf-8` on Windows.

## Orchestrator

| Command | Action |
|---------|--------|
| `python run_all.py` | Run all 4 experiments sequentially |

## Runners

| Command | Description |
|---------|-------------|
| `python studies/runner/synthetic/run.py` | Baseline (CE, no bank) |
| `python studies/runner/synthetic/run.py method=ed_gb +bank=ed_gb` | ED-GB |
| `python studies/runner/synthetic/run.py method=static_bank +bank=static` | Static replay |
| `python studies/runner/synthetic/run.py method=focal_loss` | Focal loss |
| `python studies/runner/synthetic/run.py method=class_balanced` | Class-balanced |
| `python studies/runner/synthetic/run.py method=ed_gb +bank=ed_gb training.learning_rate=0.01` | ED-GB custom LR |
| `python studies/runner/baseline_matrix/run.py` | All 5 methods compared |
| `python studies/runner/ablation/run.py` | Sweep capacity/budget/warmup |
| `python studies/runner/stress_test/run.py` | Edge cases (zero/full budget, extreme imbalance, no warmup) |

## Config Overrides

| Key | Values | Default |
|-----|--------|---------|
| `method=` | `baseline`, `static_bank`, `ed_gb`, `focal_loss`, `class_balanced` | `baseline` |
| `+bank=` | `static`, `ed_gb` | *(none)* |
| `data=` | `synthetic` | `synthetic` |
| `model=` | `classifier` | `classifier` |
| `training=` | `default`, `lightning` | `default` |

---

## Go / No-Go Decision Gates

Run in order. Each phase gates the next.

| # | Experiment | Go | No-Go | Dur | Gates Next |
|---|-----------|----|-------|-----|-----------|
| 1 | Synthetic ED-GB | Runs without error, accuracy > baseline | Crashes or accuracy ≤ baseline | 8s | Yes |
| 2 | Baseline Matrix | ED-GB ranks top 2 of 5 methods | ED-GB ranks 4th or 5th | 40s | Yes |
| 3 | Ablation | Accuracy > baseline for ≥75% of parameter grid | Narrow working range only | 8min | Yes |
| 4 | Stress Test | All 4 edge cases pass without crash/NaN | Any test fails | 32s | Final gate |

**Total duration:** ~10 min. Stop at any NO-GO and fix before proceeding.


## GO And No-Go
python studies/runner/baseline_matrix/run.py
python studies/runner/shift_experiment/run.py
​