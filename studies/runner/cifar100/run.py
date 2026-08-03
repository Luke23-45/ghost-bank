from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.bank.core.base import AbstractGhostBank
from src.bank.core.exposure import ExposureTracker
from src.bank.core.pid_controller import PIDController
from src.methods import Method
from src.models import ResNet
from src.training import (
    ConsoleEpochCallback,
    GhostBankLightningModule,
    GhostBankProgressBar,
    DebtCurveLogger,
    ExposureTrackerCallback,
)
from src.utils.logging import setup_logging
from studies.output import OutputManager
from studies.runner.cifar100.metrics import (
    aggregate_matrices,
    average_accuracy,
    backward_transfer,
    forgetting,
    matrix_to_csv,
)
from src.bank.core.allocator import allocate_uniform_fixed_total
from studies.runner.common.base_runner import (
    AbstractRunner,
    create_datamodule,
    create_model,
    create_bank,
    create_method,
    create_pl_module,
)
from src.data.cifar100.transforms import make_train_transform_from_rng
from src.data.cifar100.transforms import make_eval_transform
from studies.runner.common.path_utils import get_config_dir

BANK_MAP = {"static_bank": "static", "uniform_herding": "herding", "icarl": "herding"}


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


_RUN_README = """# Experiment run output

This directory holds the full persistence trail of one experiment run
(config x seed sweep) so every number in a paper table can be traced to
raw files.

## Layout

- `configs/resolved_config.yaml`  - exact resolved Hydra config (seed, budget, method, ...).
- `run_meta.json`                 - git commit / dirty flag, python & library versions, device, wall clock.
- `README.md`                     - this file.
- `metrics/seed_{s}_metrics.json` - per-seed scalar metrics (avg acc, forgetting, BWT,
                                    per-task final acc, epochs used, wall time).
- `metrics/seed_{s}_accuracy_matrix.csv` - accuracy matrix (see below).
- `metrics/seed_{s}_task_classes.json`   - original class ids assigned to each task
                                          (the per-seed class permutation).
- `metrics/seed_{s}_bank_sizes.json`     - exemplar memory size per class at the end of
                                          each task (absent for methods without a bank).
- `metrics/aggregated_metrics.csv`       - mean / std across seeds for every scalar metric.
- `metrics/aggregated_accuracy_matrix.csv` / `_std.csv` - element-wise mean / population
                                          std of the accuracy matrix across seeds.
- `results/final_results.json`    - aggregated + per-seed results.
- `seed_{s}_task_{t}/`            - PyTorch Lightning CSVLogger output for every (seed, task):
  - `metrics.csv`    - per-epoch train/val/test metrics.
  - `hparams.yaml`   - hyperparameters passed to the Lightning module.
  - `task_meta.yaml` - task id, class ids, bank name/budget, epochs actually used.
- `artifacts/`       - optional persisted final models (`save_checkpoint: true`).

## Accuracy matrix format

`seed_{s}_accuracy_matrix.csv` has one row per task; row i is measured
at the end of task i.  Cell [i][j] is the accuracy on task j's test set.
Cells for tasks not yet seen (j > i) are `nan` and are excluded from all
statistics.  The final row holds the final-state per-task accuracies, so
`test/avg_acc` is the mean of that row.

## Metric definitions (matching studies/runner/cifar100/metrics.py)

- `avg_acc`            - mean over tasks of the final-row accuracy (per seed)
                         / mean across seeds (aggregated).
- `forgetting`         - Chaudhry et al.: mean over tasks 0..T-2 of
                         (peak accuracy - final accuracy) per task.
- `backward_transfer`  - mean over tasks 0..T-2 of (final - first-evaluated
                         accuracy) per task.
"""


def _build_run_meta(cfg: DictConfig, seeds: list[int], started_at: datetime) -> dict:
    """Environment and provenance metadata for a run directory."""
    import subprocess

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except BaseException:
            return None
        return None

    return {
        "experiment": cfg.runner.experiment_name,
        "method": cfg.method.name,
        "seeds": [int(s) for s in seeds],
        "num_seeds": len(seeds),
        "num_tasks": int(cfg.data.get("num_tasks", 10)),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time_s": (datetime.now() - started_at).total_seconds(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


class CIFAR100Runner(AbstractRunner):
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        base_overrides_leaf = [
            o for o in self.overrides if not o.startswith("method.")
        ]
        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            base_cfg = compose(
                "config",
                overrides=base_overrides_leaf + ["+runner=cifar100"],
            )

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
                method_overrides.append("++bank.exclude_classes=[]")

            with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
                cfg = compose(
                    "config",
                    overrides=method_overrides + self.overrides + base_overrides,
                )
            pairs.append((cfg, method_name))
        return pairs

    def run_experiment(
        self,
        cfg: DictConfig,
        output_manager: OutputManager,
    ) -> dict:
        log_cfg = cfg.training.get("logging", {})
        setup_logging(level=log_cfg.get("level", "info"))
        seeds: list[int] = cfg.runner.get("seeds", [13])
        started_at = datetime.now()

        all_metrics: list[dict] = []
        for seed in seeds:
            metrics = self._run_single_seed(cfg, str(output_manager.root), seed)
            all_metrics.append(metrics)

        aggregated = _aggregate_metrics(all_metrics)

        try:
            matrices: list[list[list[float]]] = []
            for m in all_metrics:
                matrix = m.pop("accuracy_matrix", None)
                class_ids = m.pop("task_class_ids", None)
                bank_sizes = m.pop("bank_sizes", None)
                if matrix is not None:
                    matrices.append(matrix)
                    output_manager.write_file(
                        f"metrics/seed_{m['seed']}_accuracy_matrix.csv",
                        matrix_to_csv(matrix),
                    )
                if class_ids is not None:
                    output_manager.write_file(
                        f"metrics/seed_{m['seed']}_task_classes.json",
                        json.dumps(class_ids, indent=2),
                    )
                if bank_sizes:
                    output_manager.write_file(
                        f"metrics/seed_{m['seed']}_bank_sizes.json",
                        json.dumps(bank_sizes, indent=2),
                    )
                output_manager.write_metrics(m, f"seed_{m['seed']}_metrics.json")

            if matrices:
                mean_matrix, std_matrix = aggregate_matrices(matrices)
                output_manager.write_file(
                    "metrics/aggregated_accuracy_matrix.csv",
                    matrix_to_csv(mean_matrix),
                )
                output_manager.write_file(
                    "metrics/aggregated_accuracy_matrix_std.csv",
                    matrix_to_csv(std_matrix),
                )

            output_manager.write_file(
                "run_meta.json",
                json.dumps(_build_run_meta(cfg, seeds, started_at), indent=2),
            )
            output_manager.write_file("README.md", _RUN_README)

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

    @staticmethod
    def _imprint_head(
        model: ResNet,
        train_loader: DataLoader,
        task_id: int,
        classes_per_task: int,
    ) -> None:
        """LUCIR/PODNet-style weight imprinting for newly added head rows.

        Runs one full pass of the new task's training data through the frozen
        backbone and initializes the current task's head rows with L2-normalized
        class-mean features, so new prototypes start from the data instead of
        random directions.  Only rows of the current task's classes are touched;
        task 0 keeps its random initialization (as in LUCIR).
        """
        if task_id == 0:
            return
        device = next(model.parameters()).device
        class_ids = torch.arange(
            task_id * classes_per_task,
            (task_id + 1) * classes_per_task,
            dtype=torch.long,
        )
        feats: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for _, x, y in train_loader:
                mask = torch.isin(y, class_ids)
                if bool(mask.any()):
                    feats.append(model.extract_features(x.to(device))[mask.to(device)].detach().cpu())
                    labels.append(y[mask])
        if feats:
            model.fc.imprint(torch.cat(feats), torch.cat(labels))

    def _run_single_seed(self, cfg: DictConfig, output_root: str, seed: int) -> dict:
        cfg.data.seed = seed
        pl.seed_everything(seed, workers=True)
        _seed_t0 = time.time()
        debug = bool(cfg.get("debug", False))

        dm = create_datamodule(cfg)
        dm.setup("fit")

        classes_per_task = dm.classes_per_task
        num_tasks = dm.num_tasks
        total_classes = num_tasks * classes_per_task
        task_class_ids = {t: dm.task_class_ids(t) for t in range(num_tasks)}

        model = create_model(cfg, num_classes=classes_per_task)
        bank = create_bank(cfg, num_classes=classes_per_task, run_seed=seed)
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
        epochs_used: list[int] = []
        bank_sizes: list[dict] = []

        for task_id in range(num_tasks):
            if task_id > 0:
                if hasattr(method, "on_task_start"):
                    method.on_task_start(model, task_id)
                model.expand_head(classes_per_task)
                if bank is not None:
                    bank.expand(classes_per_task)
            if bank is not None and hasattr(bank, "start_task"):
                bank.start_task()

            if bank is not None and hasattr(bank, "set_quotas"):
                quota_alloc = allocate_uniform_fixed_total(
                    num_classes=(task_id + 1) * classes_per_task,
                    total_budget=cfg.data.memory_total,
                    floor=cfg.bank.get("floor", 1),
                )
                if debug:
                    print(f"[RUNNER] task={task_id} set_quotas allocation={quota_alloc}", flush=True)
                bank.set_quotas(quota_alloc)

            train_loader, _ = dm.get_task_loaders(task_id)
            if cfg.model.get("imprint", True) and hasattr(model.fc, "imprint"):
                self._imprint_head(model, train_loader, task_id, classes_per_task)
            current_num_classes = (task_id + 1) * classes_per_task

            augment_rng = torch.Generator(device="cpu")
            augment_rng.manual_seed(cfg.data.seed + task_id)
            train_transform = make_train_transform_from_rng(
                mean=dm.config.mean,
                std=dm.config.std,
                rng=augment_rng,
            )

            pl_module = create_pl_module(
                model, method, cfg,
                bank=bank,
                num_classes=current_num_classes,
                exposure_tracker=exposure_tracker,
                pid_controller=pid_controller,
                train_transform=train_transform,
                augment_generator=augment_rng,
                raw_dataset=dm.train_dataset,
            )

            # Determine quiet / verbose mode from training.logging.level --
            log_cfg = cfg.training.get("logging", {})
            log_level = log_cfg.get("level", "info")
            _quiet = log_level in ("warning", "error", "critical", "none")

            show_progress = cfg.training.get("enable_progress_bar", True)
            if _quiet:
                show_progress = False

            callbacks: list[pl.Callback] = []
            if _quiet:
                callbacks.append(
                    ConsoleEpochCallback(prefix=f"seed={seed} task={task_id}")
                )
            elif show_progress:
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

            # Early stopping --------------------------------------------------
            es_cfg = cfg.training.get("early_stopping")
            if es_cfg is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor=es_cfg.get("monitor", "val/acc"),
                        mode=es_cfg.get("mode", "max"),
                        patience=es_cfg.get("patience", 10),
                        min_delta=es_cfg.get("min_delta", 0.001),
                        stopping_threshold=es_cfg.get("stopping_threshold", None),
                        verbose=False,
                    )
                )

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
                enable_progress_bar=show_progress,
                enable_model_summary=not _quiet,
                callbacks=callbacks,
                logger=[csv_logger],
                enable_checkpointing=False,
            )

            val_loader = dm.get_val_task_loader(task_id)
            if val_loader is None:
                print("[WARNING] Val splits not found! Early stopping will monitor test set.", flush=True)
                val_loader = dm.get_task_test_loader(task_id)
            if debug:
                print(f"[RUNNER] task={task_id} trainer.fit starting...", flush=True)
            import time as _time
            _t0 = _time.time()
            trainer.fit(
                pl_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
            if debug:
                print(f"[RUNNER] task={task_id} trainer.fit done in {_time.time()-_t0:.1f}s", flush=True)
            epochs_used.append(trainer.current_epoch + 1)

            task_meta = {
                "method": cfg.method.name,
                "seed": seed,
                "task_id": task_id,
                "class_ids": dm.task_class_ids(task_id),
                "epochs_used": epochs_used[-1],
                "max_epochs": cfg.runner.get("epochs_per_task", 70),
                "memory_total": cfg.data.get("memory_total", 2000),
                "retrieval_budget": cfg.method.get("retrieval_budget", 64),
                "warmup_steps": cfg.method.get("warmup_steps", 0),
                "kd_weight": cfg.method.get("kd_weight", 0.0),
                "kd_temperature": cfg.method.get("kd_temperature", 2.0),
                "bank": cfg.bank.name if "bank" in cfg else None,
                "num_classes": current_num_classes,
            }
            task_meta_path = os.path.join(
                output_root, f"seed_{seed}_task_{task_id}", "task_meta.yaml"
            )
            os.makedirs(os.path.dirname(task_meta_path), exist_ok=True)
            with open(task_meta_path, "w", encoding="utf-8") as f:
                f.write(OmegaConf.to_yaml(task_meta))
            if bank is not None and hasattr(bank, "rebuild_selected"):
                # PyTorch Lightning moves the model to CPU after trainer.fit() finishes.
                # We must manually move it back to the accelerator device for fast feature extraction.
                accelerator_device = trainer.strategy.root_device
                model.to(accelerator_device)

                eval_transform = make_eval_transform(
                    mean=dm.config.mean,
                    std=dm.config.std,
                )
                allocation = allocate_uniform_fixed_total(
                    num_classes=current_num_classes,
                    total_budget=cfg.data.memory_total,
                    floor=cfg.bank.get("floor", 1),
                )
                if debug:
                    print(f"[RUNNER] task={task_id} rebuild_selected starting... allocation={allocation}", flush=True)
                _t1 = _time.time()
                bank.rebuild_selected(
                    model=model,
                    allocation=allocation,
                    eval_transform=eval_transform,
                    device=accelerator_device,
                    verbose=debug,
                )
                if debug:
                    print(f"[RUNNER] task={task_id} rebuild_selected done in {_time.time()-_t1:.1f}s", flush=True)

            if debug:
                print(f"[RUNNER] task={task_id} testing loop starting...", flush=True)
            _t2 = _time.time()
            with torch.no_grad():
                model.eval()
                row = [float("nan")] * num_tasks
                for prev_task in range(task_id + 1):
                    _t3 = _time.time()
                    task_test_loader = dm.get_task_test_loader(prev_task)
                    test_results = trainer.test(
                        pl_module, dataloaders=task_test_loader, verbose=False,
                    )
                    task_acc = 0.0
                    if test_results and "test/acc" in test_results[0]:
                        task_acc = test_results[0]["test/acc"]
                    row[prev_task] = task_acc
                    if debug:
                        print(f"[RUNNER] task={task_id} test prev_task={prev_task} acc={task_acc:.4f} in {_time.time()-_t3:.1f}s", flush=True)
                accuracy_matrix.append(row)
            if debug:
                print(f"[RUNNER] task={task_id} testing loop done in {_time.time()-_t2:.1f}s", flush=True)

            if bank is not None:
                bank_state = bank.state_dict()
                pool_map = bank_state.get("selected", bank_state["bank"])
                bank_sizes.append({int(c): len(pool) for c, pool in pool_map.items()})

        if cfg.output.get("save_checkpoint", False):
            ckpt_path = os.path.join(
                output_root, "artifacts", f"final_model_seed_{seed}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)

        final_avg_acc = average_accuracy(accuracy_matrix)
        forget = forgetting(accuracy_matrix) if num_tasks > 1 else 0.0
        bwt = backward_transfer(accuracy_matrix) if num_tasks > 1 else 0.0

        metrics: dict = {
            "method": cfg.method.name,
            "seed": seed,
            "test/avg_acc": final_avg_acc,
            "test/forgetting": forget,
            "test/backward_transfer": bwt,
            "wall_time_s": time.time() - _seed_t0,
            "accuracy_matrix": accuracy_matrix,
            "task_class_ids": task_class_ids,
            "bank_sizes": bank_sizes,
        }

        for t in range(num_tasks):
            if t < len(accuracy_matrix):
                col = [accuracy_matrix[row][t] for row in range(t, num_tasks)]
                final = col[-1] if col else 0.0
                metrics[f"test/task_{t}_final_acc"] = final
            if t < len(epochs_used):
                metrics[f"train/epochs_task_{t}"] = epochs_used[t]

        return metrics


if __name__ == "__main__":
    runner = CIFAR100Runner(overrides=sys.argv[1:])
    runner.run()
