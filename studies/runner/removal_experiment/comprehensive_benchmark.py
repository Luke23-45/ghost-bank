"""Proper-drift benchmark: bank freeze + shifted test labels.

Uses all three discovered fixes:
  Patch 1 — freeze_bank prevents contamination after shift
  Patch 2 — shift_test_dataset aligns eval with new distribution
  Patch 3 — early removal/forgetting scenario available via removal_experiment

Run:  python studies/runner/removal_experiment/comprehensive_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl

from studies.runner.common.base_runner import (
    AbstractRunner,
    create_datamodule,
    create_model,
    create_bank,
    create_method,
    create_pl_module,
    _run_single_seed,
    _aggregate_metrics,
)
from studies.runner.common.path_utils import get_config_dir
from studies.output import OutputManager
from src.training.callbacks import DistributionShiftCallback
from src.training import DebtCurveLogger, ExposureTrackerCallback, GhostBankProgressBar

BANK_MAP = {"static_bank": "static", "ed_gb": "ed_gb", "pid_gb": "pid_gb", "baseline": None}


def run_proper_drift_trial(method_name, seed, freeze_bank=True, shift_test=True,
                           shift_epoch=5, max_epochs=10):
    """Run a single seed under proper drift (bank freeze + shifted test labels)."""
    pl.seed_everything(seed)

    overrides = [
        "+runner=shift",
        f"method={method_name}",
        "training.enable_progress_bar=false",
        "training.log_every_n_steps=50",
        f"training.max_epochs={max_epochs}",
    ]
    bank_name = BANK_MAP.get(method_name)
    if bank_name is not None:
        overrides += [f"+bank={bank_name}", "bank.exclude_classes=[]"]

    with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
        cfg = compose("config", overrides=overrides)

    cfg.data.seed = seed
    datamodule = create_datamodule(cfg)
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    num_classes = train_dataset.num_classes
    class_counts = train_dataset.class_counts

    model = create_model(cfg, num_classes=num_classes)
    bank = create_bank(cfg, num_classes=num_classes)
    method = create_method(cfg, class_counts=class_counts)
    minority_classes = cfg.data.get("minority_classes", None)
    pl_module = create_pl_module(
        model, method, cfg, bank=bank,
        num_classes=num_classes, minority_classes=minority_classes,
    )

    callbacks = []
    if bank is not None:
        callbacks.append(DebtCurveLogger())
    if pl_module.exposure_tracker is not None:
        callbacks.append(ExposureTrackerCallback())
    callbacks.append(
        DistributionShiftCallback(
            shift_epoch=shift_epoch,
            swap_classes=(0, 2),
            freeze_bank=freeze_bank and bank is not None,
            shift_test_dataset=shift_test,
        )
    )

    from pytorch_lightning.loggers import CSVLogger
    import tempfile
    out_dir = str(Path(tempfile.gettempdir()) / "proper_drift")
    csv_logger = CSVLogger(save_dir=out_dir, name=f"{method_name}_{seed}", version="")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision=32,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=[csv_logger],
        enable_checkpointing=False,
    )

    trainer.fit(pl_module, datamodule=datamodule)
    test_results = trainer.test(pl_module, datamodule=datamodule)

    metrics = {"method": method_name, "seed": seed}
    if test_results:
        for key, val in test_results[0].items():
            metrics[str(key)] = val
    return metrics


def run_standard_vs_proper(methods, seeds, max_epochs=10):
    """Run comparison: standard (original labels) vs proper drift."""
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK: Standard vs Proper Drift")
    print("=" * 70)

    configs = [
        ("Standard (original labels, bank active)", False, False),
        ("Proper drift (shifted labels, bank frozen)", True, True),
        ("Shifted labels + active bank (worst case)", False, True),
    ]

    for label, freeze, shift_test in configs:
        print(f"\n\n  --- {label} ---")
        print(f"  {'Method':<15} {'Bal Acc':>10} {'Min Recall':>12} {'C0 Acc':>8} {'C2 Acc':>8}")
        print("  " + "-" * 53)

        for method_name in methods:
            all_m = []
            for seed in seeds:
                m = run_proper_drift_trial(
                    method_name, seed,
                    freeze_bank=freeze,
                    shift_test=shift_test,
                    max_epochs=max_epochs,
                )
                all_m.append(m)

            agg = _aggregate_metrics(all_m)
            bal = agg.get("test/balanced_acc_mean", 0)
            bal_s = agg.get("test/balanced_acc_std", 0)
            rec = agg.get("test/minority_recall_mean", 0)
            rec_s = agg.get("test/minority_recall_std", 0)
            c0 = agg.get("test/acc_class_0_mean", 0)
            c2 = agg.get("test/acc_class_2_mean", 0)
            print(f"  {method_name:<15} {bal:>7.3f} +/- {bal_s:<.3f}  "
                  f"{rec:>7.3f} +/- {rec_s:<.3f}  "
                  f"{c0:>7.3f}  {c2:>7.3f}")


if __name__ == "__main__":
    methods = ["baseline", "static_bank", "pid_gb"]
    seeds = [13, 42, 73]
    run_standard_vs_proper(methods, seeds, max_epochs=10)
