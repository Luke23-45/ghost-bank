from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger

from src.bank.core.base import AbstractGhostBank
from src.bank.core.exposure import ExposureTracker
from src.bank.core.pid_controller import PIDController
from src.methods import Method
from src.models import ResNet
from src.training import (
    GhostBankLightningModule,
    GhostBankProgressBar,
    DebtCurveLogger,
    ExposureTrackerCallback,
)
from studies.output import OutputManager
from studies.runner.cifar100.metrics import average_accuracy, forgetting, backward_transfer
from studies.runner.common.base_runner import (
    AbstractRunner,
    create_datamodule,
    create_model,
    create_bank,
    create_method,
    create_pl_module,
)
from studies.runner.common.path_utils import get_config_dir

BANK_MAP = {"static_bank": "static", "ed_gb": "ed_gb", "pid_gb": "pid_gb"}


def _aggregate_metrics(all_metrics: list[dict]) -> dict:
    if not all_metrics:
        return {}

    aggregated: dict = {
        "method": all_metrics[0]["method"],
        "num_seeds": len(all_metrics),
    }

    numeric_keys: set[str] = set()
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

    return aggregated


class CIFAR100Runner(AbstractRunner):
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            base_cfg = compose("config", overrides=self.overrides + ["+runner=cifar100"])

        base_overrides = [
            "+runner=cifar100",
            "data=cifar100",
            "model=cifar_resnet",
            "training=cifar100",
        ]

        pairs: list[tuple[DictConfig, str | None]] = []
        for method_name in base_cfg.runner.methods:
            method_overrides = [f"method={method_name}"]
            if method_name in BANK_MAP:
                method_overrides.append(f"+bank={BANK_MAP[method_name]}")
                method_overrides.append("bank.exclude_classes=[]")
            with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
                cfg = compose(
                    "config",
                    overrides=self.overrides + base_overrides + method_overrides,
                )
            pairs.append((cfg, method_name))
        return pairs

    def run_experiment(
        self,
        cfg: DictConfig,
        output_manager: OutputManager,
    ) -> dict:
        seeds: list[int] = cfg.runner.get("seeds", [13])

        all_metrics: list[dict] = []
        for seed in seeds:
            metrics = self._run_single_seed(cfg, str(output_manager.root), seed)
            all_metrics.append(metrics)

        aggregated = _aggregate_metrics(all_metrics)

        try:
            for m in all_metrics:
                output_manager.write_metrics(m, f"seed_{m['seed']}_metrics.json")

            output_manager.write_metrics(aggregated, "aggregated_metrics.csv")
            output_manager.finalize(
                {
                    "aggregated": aggregated,
                    "per_seed_metrics": all_metrics,
                    "method": cfg.method.name,
                    "num_seeds": len(seeds),
                }
            )
            output_manager.complete()
        except BaseException:
            output_manager.fail()
            raise

        return aggregated

    def _run_single_seed(self, cfg: DictConfig, output_root: str, seed: int) -> dict:
        cfg.data.seed = seed
        pl.seed_everything(seed, workers=True)

        dm = create_datamodule(cfg)
        dm.setup("fit")

        classes_per_task = dm.classes_per_task
        num_tasks = dm.num_tasks
        total_classes = num_tasks * classes_per_task

        model = create_model(cfg, num_classes=classes_per_task)
        bank = create_bank(cfg, num_classes=total_classes)
        method = create_method(cfg, class_counts=None)

        exposure_tracker: ExposureTracker | None = None
        if getattr(method, "needs_exposure_tracker", False):
            exposure_tracker = ExposureTracker(total_classes)

        pid_controller: PIDController | None = None
        if getattr(method, "needs_pid_controller", False):
            pid_controller = PIDController(
                num_classes=total_classes,
                K_p=getattr(method, "K_p", 1.0),
                K_i=getattr(method, "K_i", 0.1),
                K_d=getattr(method, "K_d", 0.5),
                decay=getattr(method, "pid_decay", 0.99),
                smooth=getattr(method, "pid_smooth", 0.9),
                temperature=getattr(method, "temperature", 1.0),
                class_weights=None,
            )

        accuracy_matrix: list[list[float]] = []

        for task_id in range(num_tasks):
            if task_id > 0:
                model.expand_head(classes_per_task)
                if bank is not None:
                    bank.expand(classes_per_task)

            train_loader, _ = dm.get_task_loaders(task_id)
            current_num_classes = (task_id + 1) * classes_per_task

            pl_module = create_pl_module(
                model, method, cfg,
                bank=bank,
                num_classes=current_num_classes,
                exposure_tracker=exposure_tracker,
                pid_controller=pid_controller,
            )

            callbacks: list[pl.Callback] = []
            if cfg.training.get("enable_progress_bar", True):
                callbacks.append(
                    GhostBankProgressBar(
                        refresh_rate=cfg.training.get("progress_refresh_rate", 1),
                        leave=True,
                    )
                )
            if bank is not None:
                callbacks.append(DebtCurveLogger())
            if pl_module.exposure_tracker is not None:
                callbacks.append(ExposureTrackerCallback())

            csv_logger = CSVLogger(
                save_dir=output_root,
                name=f"seed_{seed}_task_{task_id}",
                version="",
            )

            trainer = pl.Trainer(
                accelerator=getattr(cfg.training, "accelerator", "auto"),
                devices=getattr(cfg.training, "devices", 1),
                precision=getattr(cfg.training, "precision", 32),
                max_epochs=cfg.runner.get("epochs_per_task", 70),
                log_every_n_steps=cfg.training.log_every_n_steps,
                gradient_clip_val=cfg.training.get("gradient_clip_val", None),
                enable_progress_bar=cfg.training.get("enable_progress_bar", True),
                callbacks=callbacks,
                logger=[csv_logger],
                enable_checkpointing=False,
            )

            trainer.fit(pl_module, train_dataloaders=train_loader)

            with torch.no_grad():
                model.eval()
                row = [0.0] * num_tasks
                for prev_task in range(task_id + 1):
                    task_test_loader = dm.get_task_test_loader(prev_task)
                    test_results = trainer.test(
                        pl_module, dataloaders=task_test_loader, verbose=False,
                    )
                    task_acc = 0.0
                    if test_results and "test/acc" in test_results[0]:
                        task_acc = test_results[0]["test/acc"]
                    row[prev_task] = task_acc
                accuracy_matrix.append(row)

        final_avg_acc = average_accuracy(accuracy_matrix)
        forget = forgetting(accuracy_matrix) if num_tasks > 1 else 0.0
        bwt = backward_transfer(accuracy_matrix) if num_tasks > 1 else 0.0

        metrics: dict = {
            "method": cfg.method.name,
            "seed": seed,
            "test/avg_acc": final_avg_acc,
            "test/forgetting": forget,
            "test/backward_transfer": bwt,
        }

        for t in range(num_tasks):
            if t < len(accuracy_matrix):
                col = [accuracy_matrix[row][t] for row in range(t, num_tasks)]
                final = col[-1] if col else 0.0
                metrics[f"test/task_{t}_final_acc"] = final

        return metrics


if __name__ == "__main__":
    runner = CIFAR100Runner()
    runner.run()
