from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import spearmanr

from src.bank.core.probe import ProbeScorer
from src.data.cifar100 import CIFAR100Config, CIFAR100DataModule
from src.data.cifar100.transforms import make_eval_transform, make_train_transform_from_rng
from src.methods.probe_guided import ProbeGuidedMethod
from src.models.ptm import PTModel
from src.training import GhostBankLightningModule
from src.training.callbacks import ConsoleEpochCallback
from src.utils.logging import setup_logging
from studies.output import OutputManager
from studies.runner.cifar100.metrics import average_accuracy, forgetting, backward_transfer
from studies.runner.common.path_utils import get_config_dir


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


def _compute_per_class_forgetting(
    acc_matrix: list[list[float]],
    num_tasks: int,
    classes_per_task: int,
) -> list[float]:
    """Compute per-class forgetting from an accuracy matrix.

    For each task column, forgetting = max(row acc) - final row acc.
    Returns a list of length num_tasks * classes_per_task with per-class
    forgetting values (repeated within each task block).
    """
    if len(acc_matrix) < 2:
        return []

    n_cols = max(len(row) for row in acc_matrix)
    rect = np.full((len(acc_matrix), n_cols), np.nan)
    for i, row in enumerate(acc_matrix):
        rect[i, :len(row)] = row

    per_task_forgetting = []
    for t in range(n_cols):
        col = rect[:, t]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            per_task_forgetting.append(float(np.nanmax(valid) - valid[-1]))
        else:
            per_task_forgetting.append(0.0)

    per_class = []
    for t in range(len(per_task_forgetting)):
        for _ in range(classes_per_task):
            per_class.append(per_task_forgetting[t])
    return per_class


def _compute_spearman_correlation(
    probe_scorer: ProbeScorer,
    acc_matrix: list[list[float]],
    num_tasks: int,
    classes_per_task: int,
) -> float | None:
    """Compute Spearman rank correlation between probe scores and forgetting."""
    if not probe_scorer.history:
        return None

    per_class_forgetting = _compute_per_class_forgetting(
        acc_matrix, num_tasks, classes_per_task,
    )
    if not per_class_forgetting:
        return None

    last_probe_scores = probe_scorer.raw_scores
    if not last_probe_scores:
        return None

    min_len = min(len(per_class_forgetting), len(last_probe_scores))
    if min_len < 2:
        return None

    f_arr = np.array(per_class_forgetting[:min_len], dtype=np.float64)
    p_arr = np.array(last_probe_scores[:min_len], dtype=np.float64)

    f_var = np.var(f_arr)
    p_var = np.var(p_arr)
    if f_var < 1e-12 or p_var < 1e-12:
        return None

    r, _ = spearmanr(f_arr, p_arr)
    return float(r) if not np.isnan(r) else None


def _transform_raw_batch(raw_images: torch.Tensor, eval_transform) -> torch.Tensor:
    batch_list = []
    for i in range(raw_images.shape[0]):
        raw = raw_images[i]
        if raw.dim() == 3 and raw.shape[-1] == 3 and raw.shape[0] != 3:
            img_nchw = raw.permute(2, 0, 1).contiguous()
        else:
            img_nchw = raw.contiguous()
        if eval_transform is not None:
            batch_list.append(eval_transform(img_nchw))
        else:
            img = img_nchw.float() / 255.0
            if img.dim() == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1).contiguous()
            batch_list.append(img)
    return torch.stack(batch_list, dim=0)


def _compute_nme_prototypes(
    model: PTModel,
    exemplar_bank: dict[int, list],
    num_classes: int,
    eval_transform,
    device: torch.device,
) -> torch.Tensor:
    """Compute class prototypes from the current exemplar bank."""
    feat_dim = model.embedding_dim
    prototypes = torch.zeros(num_classes, feat_dim, device=device)
    for class_id in range(num_classes):
        if class_id % 10 == 0:
            print(
                f"    [nme] building prototypes {class_id + 1}/{num_classes}",
                flush=True,
            )
        pool = exemplar_bank.get(class_id, [])
        if not pool:
            continue

        raw_images = []
        for item in pool:
            raw = item[0]
            if not torch.is_tensor(raw):
                raw = torch.as_tensor(raw)
            raw_images.append(raw)
        raw_batch = torch.stack(raw_images, dim=0)
        images_t = _transform_raw_batch(raw_batch, eval_transform).to(device)
        with torch.no_grad():
            feats = model.extract_features(images_t)
        prototypes[class_id] = feats.mean(dim=0)
    return prototypes


def _evaluate_with_nme(
    model: PTModel,
    exemplar_bank: dict[int, list],
    dm: CIFAR100DataModule,
    num_seen_classes: int,
    current_task_id: int,
    eval_transform,
    device: torch.device,
) -> list[float]:
    """Evaluate seen tasks with a nearest-prototype classifier."""
    prototypes = _compute_nme_prototypes(
        model,
        exemplar_bank,
        num_seen_classes,
        eval_transform,
        device,
    )
    proto_norm = F.normalize(prototypes, dim=-1)

    row = [0.0] * (current_task_id + 1)
    model.eval()
    with torch.no_grad():
        for task_id in range(current_task_id + 1):
            print(
                f"    [nme] evaluating task {task_id + 1}/{current_task_id + 1}",
                flush=True,
            )
            test_loader = dm.get_task_test_loader(task_id)
            correct = 0
            total = 0
            for _, x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                feats = model.extract_features(x)
                feat_norm = F.normalize(feats, dim=-1)
                logits = torch.mm(feat_norm, proto_norm[:num_seen_classes].t())
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()

            row[task_id] = float(correct / total) if total > 0 else 0.0
    return row


class ProbeGuidedCIFAR100Runner:
    """End-to-end experiment runner for the PTM/probe-guided method.

    Orchestrates:
        1. CIFAR-100 data loading with held-out probe splits,
        2. Pre-trained backbone initialisation (frozen or partially tuned),
        3. Task-by-task training with probe-guided replay,
        4. Post-hoc classifier calibration,
        5. Final evaluation with full metric reporting.
    """

    def __init__(self, overrides: list[str] | None = None) -> None:
        self.overrides = overrides or []

    def compose_configs(self) -> list[tuple[DictConfig, str]]:
        base_overrides = self.overrides.copy()
        if not any(o.startswith("+runner=") for o in base_overrides):
            base_overrides.insert(0, "+runner=probe_guided")

        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            base_cfg = compose("config", overrides=base_overrides)

        configs: list[tuple[DictConfig, str]] = []

        runner_methods = OmegaConf.select(base_cfg, "runner.methods")
        method_names = list(
            runner_methods if runner_methods is not None
            else ["probe_guided", "uniform_replay", "frozen_baseline"]
        )
        for method_name in method_names:
            with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
                overrides = base_overrides + [
                    "data=cifar100",
                    "model=ptm_resnet",
                    "training=cifar100",
                    f"method={method_name}",
                ]
                cfg = compose("config", overrides=overrides)
            configs.append((cfg, method_name))

        return configs

    def run(self) -> list[dict]:
        configs = self.compose_configs()
        if configs:
            log_cfg = configs[0][0].training.get("logging", {})
            setup_logging(level=log_cfg.get("level", "info"))

        all_metrics: list[dict] = []
        for cfg, run_name in configs:
            mgr = OutputManager(
                experiment=f"cifar100/probe_guided/{run_name}",
                base_dir=cfg.output.base_dir,
            )
            mgr.initialize()
            mgr.save_config(OmegaConf.to_yaml(cfg))
            try:
                metrics = self.run_experiment(cfg, output_manager=mgr)
                all_metrics.append(metrics)
            except BaseException:
                mgr.fail()
                raise
        return all_metrics

    def run_experiment(
        self,
        cfg: DictConfig,
        output_manager: OutputManager,
    ) -> dict:
        log_cfg = cfg.training.get("logging", {})
        setup_logging(level=log_cfg.get("level", "info"))
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
                    "method": cfg.method.get("name", "probe_guided"),
                    "num_seeds": len(seeds),
                }
            )
            output_manager.complete()
        except BaseException:
            output_manager.fail()
            raise

        return aggregated

    def _run_single_seed(
        self,
        cfg: DictConfig,
        output_root: str,
        seed: int,
    ) -> dict:
        method_name = cfg.method.get("name", "probe_guided")
        pl.seed_everything(seed, workers=True)

        num_tasks = cfg.data.get("num_tasks", 10)
        classes_per_task = cfg.data.get("classes_per_task", 10)
        total_classes = num_tasks * classes_per_task

        data_config = CIFAR100Config(
            root=cfg.data.root,
            seed=seed,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.get("num_workers", 4),
            pin_memory=cfg.data.get("pin_memory", True),
            num_tasks=num_tasks,
            classes_per_task=classes_per_task,
            mean=tuple(cfg.data.get("mean", [0.5071, 0.4867, 0.4408])),
            std=tuple(cfg.data.get("std", [0.2675, 0.2565, 0.2761])),
            probe_split_size=cfg.data.get("probe_split_size", 30),
            val_split_size=cfg.data.get("val_split_size", 20),
            split_seed=cfg.data.get("split_seed", 13),
            memory_total=cfg.data.get("memory_total", 2000),
            probe_enabled=True,
        )

        dm = CIFAR100DataModule(data_config)
        dm.setup("fit")

        model = PTModel(
            backbone=cfg.model.get("backbone", "resnet18"),
            pretrained=cfg.model.get("pretrained", True),
            num_classes=classes_per_task,
            freeze_backbone=cfg.model.get("freeze_backbone", True),
            embedding_dim=cfg.model.get("embedding_dim", 512),
            classifier_mode=cfg.model.get("classifier_mode", "linear"),
        )

        method = ProbeGuidedMethod(
            seed=seed,
            retrieval_budget=cfg.method.get("retrieval_budget", 64),
            warmup_steps=cfg.method.get("warmup_steps", 0),
            memory_total=cfg.data.get("memory_total", 2000),
            memory_floor=cfg.method.get("memory_floor", 1),
            gamma=cfg.method.get("gamma", 0.5),
            beta=cfg.method.get("beta", 1.0),
            probe_smoothing=cfg.method.get("probe_smoothing", 0.0),
            distillation_weight=cfg.method.get("distillation_weight", 0.0),
            calibrate=cfg.method.get("calibrate", False),
        )

        probe_scorer = ProbeScorer(
            num_classes=total_classes,
            smoothing=cfg.method.get("probe_smoothing", 0.0),
        )

        eval_transform = make_eval_transform(
            mean=data_config.mean,
            std=data_config.std,
        )

        accuracy_matrix: list[list[float]] = []
        nme_accuracy_matrix: list[list[float]] = []

        for task_id in range(num_tasks):
            print(f"\n{'=' * 60}")
            print(f"  Task {task_id + 1}/{num_tasks}  ({method_name})")
            print(f"{'=' * 60}")

            if task_id > 0:
                model.expand_head(classes_per_task)

            train_loader, _ = dm.get_task_loaders(task_id)
            current_num_classes = (task_id + 1) * classes_per_task
            replay_class_count = task_id * classes_per_task
            method.set_replay_class_count(replay_class_count)

            # --- Phase 1: compute probe scores for all seen classes ---
            if task_id == 0:
                probe_scores = None
            else:
                probe_scores = self._compute_probe_scores(
                    model, dm, probe_scorer,
                    current_num_classes, eval_transform,
                )

            # --- Phase 2: update memory allocation from probe scores ---
            method.update_allocation(probe_scores)
            method.prune_memory()

            # --- Phase 3: train on current task ---
            augment_rng = torch.Generator(device="cpu")
            augment_rng.manual_seed(seed + task_id)
            train_transform = make_train_transform_from_rng(
                mean=data_config.mean,
                std=data_config.std,
                rng=augment_rng,
            )

            pl_module = GhostBankLightningModule(
                model=model,
                method=method,
                bank=None,
                learning_rate=cfg.training.learning_rate,
                num_classes=current_num_classes,
                optimizer_name=cfg.training.get("optimizer", "sgd"),
                momentum=cfg.training.get("momentum", 0.9),
                weight_decay=cfg.training.get("weight_decay", 0.0005),
                train_transform=train_transform,
                augment_generator=augment_rng,
            )

            log_level = cfg.training.logging.level
            _quiet = log_level in ("warning", "error", "critical", "none")
            callbacks: list[pl.Callback] = []
            if _quiet:
                callbacks.append(
                    ConsoleEpochCallback(prefix=f"seed={seed} task={task_id}")
                )

            csv_logger = CSVLogger(
                save_dir=output_root,
                name=f"seed_{seed}_task_{task_id}",
                version="",
            )

            trainer = pl.Trainer(
                accelerator=cfg.training.get("accelerator", "auto"),
                devices=cfg.training.get("devices", 1),
                precision=cfg.training.get("precision", 32),
                max_epochs=cfg.runner.get("epochs_per_task", 70),
                log_every_n_steps=cfg.training.log_every_n_steps,
                enable_progress_bar=not _quiet,
                enable_model_summary=not _quiet,
                callbacks=callbacks,
                logger=[csv_logger],
                enable_checkpointing=False,
            )

            val_loader = dm.get_task_test_loader(task_id)
            trainer.fit(
                pl_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            # --- Phase 4: post-hoc calibration ---
            device = next(model.parameters()).device
            print(f"  [{method_name}] Task {task_id + 1}: calibration", flush=True)
            self._run_calibration(
                model, method, dm, current_num_classes, device, eval_transform,
            )

            # --- Phase 5: snapshot the calibrated model for the next task ---
            print(f"  [{method_name}] Task {task_id + 1}: snapshot", flush=True)
            method.snapshot_model(pl_module)

            # --- Phase 6: evaluate on all seen classes ---
            print(f"  [{method_name}] Task {task_id + 1}: linear eval", flush=True)
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

            print(f"  [{method_name}] Task {task_id + 1}: NME eval", flush=True)
            nme_row = _evaluate_with_nme(
                model,
                method._exemplar_bank,
                dm,
                num_seen_classes=current_num_classes,
                current_task_id=task_id,
                eval_transform=eval_transform,
                device=device,
            )
            nme_accuracy_matrix.append(nme_row)

        # --- Final metrics ---
        final_avg_acc = average_accuracy(accuracy_matrix)
        forget = forgetting(accuracy_matrix) if num_tasks > 1 else 0.0
        bwt = backward_transfer(accuracy_matrix) if num_tasks > 1 else 0.0
        nme_final_avg_acc = average_accuracy(nme_accuracy_matrix)
        nme_forget = forgetting(nme_accuracy_matrix) if num_tasks > 1 else 0.0
        nme_bwt = backward_transfer(nme_accuracy_matrix) if num_tasks > 1 else 0.0

        metrics: dict = {
            "method": method_name,
            "seed": seed,
            "linear/test/avg_acc": final_avg_acc,
            "linear/test/forgetting": forget,
            "linear/test/backward_transfer": bwt,
            "test/avg_acc": nme_final_avg_acc,
            "test/forgetting": nme_forget,
            "test/backward_transfer": nme_bwt,
        }

        for t in range(num_tasks):
            if t < len(accuracy_matrix):
                col = [accuracy_matrix[row][t] for row in range(t, num_tasks)]
                final_acc = col[-1] if col else 0.0
                metrics[f"test/task_{t}_final_acc"] = final_acc

        # --- Probe-to-forgetting correlation ---
        probe_history = probe_scorer.history
        if len(probe_history) >= 2:
            metrics["probe/history_depth"] = len(probe_history)
            corr = _compute_spearman_correlation(
                probe_scorer, accuracy_matrix, num_tasks, classes_per_task,
            )
            if corr is not None:
                metrics["probe/spearman_r"] = corr

        # --- Write rich output artifacts ---
        for t, row_acc in enumerate(accuracy_matrix):
            metrics[f"task_{t}/accuracy_row"] = json.dumps(row_acc)
        for t, row_acc in enumerate(nme_accuracy_matrix):
            metrics[f"nme/task_{t}/accuracy_row"] = json.dumps(row_acc)
        metrics["allocation_history"] = json.dumps(method.allocation_history)
        metrics["probe/score_history"] = json.dumps(probe_history)

        print(f"\n  Linear avg accuracy: {final_avg_acc * 100:.2f}%")
        print(f"  Linear forgetting: {forget:.2f}")
        print(f"  Linear backward transfer: {bwt:.2f}")
        print(f"  NME avg accuracy: {nme_final_avg_acc * 100:.2f}%")
        print(f"  NME forgetting: {nme_forget:.2f}")
        print(f"  NME backward transfer: {nme_bwt:.2f}")
        if "probe/spearman_r" in metrics:
            print(f"  Probe-forgetting Spearman r: {metrics['probe/spearman_r']:.3f}")

        return metrics

    def _run_calibration(
        self,
        model: PTModel,
        method: ProbeGuidedMethod,
        dm: CIFAR100DataModule,
        num_seen_classes: int,
        device: torch.device,
        eval_transform,
    ) -> None:
        """Post-hoc calibration: freeze backbone, fine-tune classifier on balanced val data."""
        if not method.calibrate:
            return

        model.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(
            model.classifier.parameters(),
            lr=method.calibration_lr,
            momentum=0.9,
            weight_decay=0.0,
        )

        val_loaders: dict[int, tuple[torch.Tensor, torch.Tensor]] = dm.get_val_loaders()
        if not val_loaders:
            return

        num_val_per_class = min(
            min(v[0].shape[0] for v in [val_loaders[c] for c in range(num_seen_classes) if c in val_loaders]),
            method.retrieval_budget,
        )

        for epoch in range(method.calibration_epochs):
            total_loss = 0.0
            total_batches = 0
            for c in range(num_seen_classes):
                if c not in val_loaders:
                    continue
                images, targets = val_loaders[c]
                n = min(images.shape[0], num_val_per_class)
                idx = torch.randperm(images.shape[0])[:n]
                images = images[idx].to(device)
                targets = targets[idx].to(device)

                if eval_transform is not None:
                    batch_list = []
                    for i in range(images.shape[0]):
                        img_nhwc = images[i]
                        img_nchw = img_nhwc.permute(2, 0, 1).contiguous()
                        batch_list.append(eval_transform(img_nchw))
                    images_t = torch.stack(batch_list, dim=0)
                else:
                    images_t = images.float() / 255.0
                    if images_t.dim() == 4 and images_t.shape[-1] == 3 and images_t.shape[1] != 3:
                        images_t = images_t.permute(0, 3, 1, 2).contiguous()

                logits = model(images_t)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

            if total_batches > 0:
                avg_loss = total_loss / total_batches
                print(f"    Calibration epoch {epoch+1}/{method.calibration_epochs}: loss={avg_loss:.4f}")

        for param in model.classifier.parameters():
            param.requires_grad = False

    def _compute_probe_scores(
        self,
        model: PTModel,
        dm: CIFAR100DataModule,
        probe_scorer: ProbeScorer,
        num_seen_classes: int,
        eval_transform,
    ) -> list[float]:
        """Compute held-out probe loss for all seen classes.

        Returns a list of probe scores (length = num_seen_classes).
        """
        model.eval()
        device = next(model.parameters()).device
        probe_loaders = dm.get_probe_loaders()
        scores: list[float] = []

        for c in range(num_seen_classes):
            if c not in probe_loaders:
                scores.append(0.0)
                continue

            probe_images, probe_targets = probe_loaders[c]
            loss = probe_scorer.compute_probe_loss(
                model=model,
                probe_images=probe_images,
                probe_targets=probe_targets,
                class_id=c,
                device=device,
                transform=eval_transform,
            )
            scores.append(loss)

        probe_scorer.update(scores)
        return scores


if __name__ == "__main__":
    runner = ProbeGuidedCIFAR100Runner(overrides=sys.argv[1:])
    runner.run()
