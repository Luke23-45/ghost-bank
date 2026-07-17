"""Class removal experiment — asymmetric forgetting benchmark.

At removal_epoch, REMOVE the minority class from the training set.
The model can only maintain accuracy on that class via replay.
PID-CR should detect the rising loss and allocate more replay,
beating static_bank's uniform allocation under asymmetric forgetting.

Usage:
    python studies/runner/removal_experiment/run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from studies.runner.common.base_runner import (
    AbstractRunner,
    create_datamodule,
    create_model,
    create_bank,
    create_method,
    create_pl_module,
)
from studies.runner.common.path_utils import get_config_dir
from studies.output import OutputManager


BANK_MAP = {"static_bank": "static", "ed_gb": "ed_gb", "pid_gb": "pid_gb", "baseline": None}


class ClassRemovalCallback(pl.Callback):
    """Remove a class from the training dataset at a given epoch."""

    def __init__(
        self, remove_epoch: int = 5, remove_class: int = 2
    ) -> None:
        super().__init__()
        self.remove_epoch = remove_epoch
        self.remove_class = remove_class
        self._removed = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch == self.remove_epoch and not self._removed:
            ds = trainer.datamodule.train_dataset
            mask = ds._ys != self.remove_class
            ds._xs = ds._xs[mask]
            ds._ys = ds._ys[mask]
            self._removed = True
            if trainer.logger is not None:
                trainer.logger.log_metrics({"class_removed": self.remove_class}, step=trainer.global_step)


def create_pl_module_with_callbacks(cfg, seed, removal_callback=None):
    """Create pl_module and callbacks for removal experiment."""
    import pytorch_lightning as pl

    pl.seed_everything(seed)
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
        num_classes=num_classes,
        minority_classes=minority_classes,
    )

    callbacks = []
    if cfg.training.get("enable_progress_bar", True):
        from src.training import GhostBankProgressBar
        callbacks.append(GhostBankProgressBar(refresh_rate=1))
    if bank is not None:
        from src.training import DebtCurveLogger
        callbacks.append(DebtCurveLogger())
    if pl_module.exposure_tracker is not None:
        from src.training import ExposureTrackerCallback
        callbacks.append(ExposureTrackerCallback())

    if removal_callback is not None:
        callbacks.append(removal_callback)

    return pl_module, datamodule, callbacks


def run_single_seed(cfg, output_root, seed, removal_callback):
    """Run one seed with class removal."""
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    pl_module, datamodule, callbacks = create_pl_module_with_callbacks(
        cfg, seed, removal_callback=removal_callback
    )

    csv_logger = CSVLogger(save_dir=output_root, name=f"seed_{seed}", version="")

    trainer = pl.Trainer(
        accelerator=getattr(cfg.training, "accelerator", "auto"),
        devices=getattr(cfg.training, "devices", 1),
        precision=getattr(cfg.training, "precision", 32),
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        enable_progress_bar=cfg.training.get("enable_progress_bar", True),
        callbacks=callbacks,
        logger=[csv_logger],
        enable_checkpointing=False,
    )

    trainer.fit(pl_module, datamodule=datamodule)
    test_results = trainer.test(pl_module, datamodule=datamodule)

    metrics = {"method": cfg.method.name, "seed": seed}
    if test_results:
        for key, val in test_results[0].items():
            metrics[str(key)] = val

    return metrics


def run_removal_experiment(cfg, output_manager, removal_epoch=5, removal_class=2):
    """Run removal experiment over all configured seeds."""
    seeds = cfg.runner.get("seeds", [13])
    all_metrics = []
    removal_cb = ClassRemovalCallback(remove_epoch=removal_epoch, remove_class=removal_class)

    for seed in seeds:
        metrics = run_single_seed(cfg, str(output_manager.root), seed, removal_cb)
        all_metrics.append(metrics)

    # Aggregate
    aggregated = {}
    if all_metrics:
        aggregated["method"] = all_metrics[0]["method"]
        aggregated["num_seeds"] = len(all_metrics)
        numeric_keys = set()
        for m in all_metrics:
            for k, v in m.items():
                if k not in ("method", "seed") and isinstance(v, (int, float)):
                    numeric_keys.add(k)

        for key in sorted(numeric_keys):
            values = [m[key] for m in all_metrics if key in m]
            if values:
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                aggregated[f"{key}_mean"] = mean
                aggregated[f"{key}_std"] = std

    for m in all_metrics:
        output_manager.write_metrics(m, f"seed_{m['seed']}_metrics.json")
    output_manager.write_metrics(aggregated, "aggregated_metrics.csv")
    output_manager.finalize({
        "aggregated": aggregated,
        "per_seed_metrics": all_metrics,
        "method": cfg.method.name,
        "num_seeds": len(seeds),
    })
    output_manager.complete()

    return aggregated


def run_removal_trial(method_name, removal_epoch=2, removal_class=2) -> dict:
    """Run a single removal trial for a given method."""
    overrides = [
        "+runner=shift",
        f"method={method_name}",
        "runner.seeds=[13,42,73]",
        "training.enable_progress_bar=false",
        "training.log_every_n_steps=50",
        "training.max_epochs=20",
    ]
    bank_name = BANK_MAP.get(method_name)
    if bank_name is not None:
        overrides += [f"+bank={bank_name}", "bank.exclude_classes=[]"]
    with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
        cfg = compose("config", overrides=overrides)

    import tempfile
    out_dir = str(Path(tempfile.gettempdir()) / "removal_experiment")
    label = f"{method_name}_removal_epoch{removal_epoch}"
    mgr = OutputManager(label, out_dir)
    mgr.initialize()
    mgr.save_config(OmegaConf.to_yaml(cfg))

    try:
        metrics = run_removal_experiment(cfg, mgr, removal_epoch=removal_epoch, removal_class=removal_class)
        return metrics
    except Exception as e:
        print(f"  ERROR: {e}")
        mgr.fail()
        return {}


if __name__ == "__main__":
    print("=" * 70)
    print("ASYMMETRIC FORGETTING - CLASS REMOVAL EXPERIMENT (HARDER)")
    print("  Removes minority class (2) at epoch 2, 20 epochs total")
    print("=" * 70)

    methods = ["static_bank", "pid_gb", "ed_gb", "baseline"]
    results = []

    for method in methods:
        print(f"\n--- {method} ---")
        m = run_removal_trial(method, removal_epoch=2, removal_class=2)
        bal = m.get("test/balanced_acc_mean", "N/A")
        bal_s = m.get("test/balanced_acc_std", "N/A")
        rec = m.get("test/minority_recall_mean", "N/A")
        rec_s = m.get("test/minority_recall_std", "N/A")
        c2 = m.get("test/acc_class_2_mean", "N/A")
        c2_s = m.get("test/acc_class_2_std", "N/A")
        print(f"  balanced_acc    = {bal} +/- {bal_s}")
        print(f"  minority_recall = {rec} +/- {rec_s}")
        print(f"  class_2_acc     = {c2} +/- {c2_s}")
        results.append((method, m))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Method':>15} {'Balanced Acc':>15} {'Minority Recall':>18} {'Class 2 Acc':>14}")
    print("  " + "-" * 62)
    for method, m in results:
        bal = m.get("test/balanced_acc_mean", 0)
        bal_s = m.get("test/balanced_acc_std", 0)
        rec = m.get("test/minority_recall_mean", 0)
        rec_s = m.get("test/minority_recall_std", 0)
        c2 = m.get("test/acc_class_2_mean", 0)
        c2_s = m.get("test/acc_class_2_std", 0)
        print(f"  {method:>15} {bal:>8.3f} +/- {bal_s:<.3f}  "
              f"{rec:>8.3f} +/- {rec_s:<.3f}  "
              f"{c2:>7.3f} +/- {c2_s:<.3f}")
