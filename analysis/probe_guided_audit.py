from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bank.core.probe import ProbeScorer
from src.bank.strategies.static import StaticReplayBank
from src.data.cifar100 import CIFAR100Config, CIFAR100DataModule
from src.data.cifar100.transforms import make_eval_transform, make_train_transform_from_rng
from src.methods import BaselineMethod, ProbeGuidedMethod, StaticBankMethod
from src.models.ptm import PTModel
from src.training import GhostBankLightningModule
from src.training.callbacks import ConsoleEpochCallback
from studies.runner.cifar100.metrics import average_accuracy, backward_transfer, forgetting


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 13
    tasks: int = 3
    classes_per_task: int = 10
    epochs: int = 5
    batch_size: int = 128
    num_workers: int = 4
    memory_total: int = 2000
    retrieval_budget: int = 64
    probe_split_size: int = 30
    val_split_size: int = 20
    split_seed: int = 13
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "16-mixed"


def _setup_device(accelerator: str) -> torch.device:
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but torch.cuda.is_available() is false.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"[device] Using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    return torch.device("cpu")


def _make_datamodule(cfg: AuditConfig) -> CIFAR100DataModule:
    data_cfg = CIFAR100Config(
        root="./data/cifar100",
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        num_tasks=cfg.tasks,
        classes_per_task=cfg.classes_per_task,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        probe_split_size=cfg.probe_split_size,
        val_split_size=cfg.val_split_size,
        split_seed=cfg.split_seed,
        memory_total=cfg.memory_total,
        probe_enabled=True,
    )
    dm = CIFAR100DataModule(data_cfg)
    dm.setup("fit")
    return dm


def _make_model(classes_per_task: int) -> PTModel:
    return PTModel(
        backbone="resnet18",
        pretrained=True,
        num_classes=classes_per_task,
        freeze_backbone=True,
        embedding_dim=512,
        classifier_mode="linear",
    )


def _make_method(name: str, cfg: AuditConfig) -> tuple[object, object | None]:
    if name == "probe_guided":
        return ProbeGuidedMethod(
            seed=cfg.seed,
            retrieval_budget=cfg.retrieval_budget,
            warmup_steps=0,
            memory_total=cfg.memory_total,
            memory_floor=1,
            gamma=0.5,
            beta=1.0,
            probe_smoothing=0.0,
            distillation_weight=0.0,
            calibrate=False,
        ), None
    if name == "uniform_replay":
        bank = StaticReplayBank(
            num_classes=cfg.tasks * cfg.classes_per_task,
            capacity_per_class=max(1, cfg.memory_total // (cfg.tasks * cfg.classes_per_task)),
            seed=cfg.seed,
        )
        return StaticBankMethod(retrieval_budget=cfg.retrieval_budget, warmup_steps=0), bank
    if name == "frozen_baseline":
        return BaselineMethod(), None
    raise ValueError(f"Unknown method: {name}")


def _transform_raw_batch(raw_images: torch.Tensor, eval_transform) -> torch.Tensor:
    if raw_images.dim() != 4:
        raise ValueError(f"Expected a 4D batch, got {tuple(raw_images.shape)}")
    if raw_images.shape[-1] == 3 and raw_images.shape[1] != 3:
        batch = raw_images.permute(0, 3, 1, 2).contiguous()
    else:
        batch = raw_images.contiguous()
    if eval_transform is not None:
        return eval_transform(batch)
    return batch.float().div(255.0)


def _gini(values: list[int]) -> float:
    xs = [v for v in values if v >= 0]
    if not xs:
        return 0.0
    total = sum(xs)
    if total <= 0:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    cum = 0.0
    for i, v in enumerate(xs, 1):
        cum += i * v
    return (2.0 * cum) / (n * total) - (n + 1) / n


def _entropy(values: list[int]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in values:
        if v <= 0:
            continue
        p = v / total
        ent -= p * math.log(p + 1e-12)
    return ent


def _compute_probe_scores(
    model: PTModel,
    dm: CIFAR100DataModule,
    probe_scorer: ProbeScorer,
    num_seen_classes: int,
    eval_transform,
    device: torch.device,
) -> list[float]:
    model.eval()
    probe_loaders = dm.get_probe_loaders()
    scores: list[float] = []
    with torch.inference_mode():
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


def _allocation_stats(allocation: list[int]) -> dict:
    if not allocation:
        return {"sum": 0, "min": 0, "max": 0, "entropy": 0.0, "gini": 0.0}
    return {
        "sum": int(sum(allocation)),
        "min": int(min(allocation)),
        "max": int(max(allocation)),
        "entropy": float(_entropy(allocation)),
        "gini": float(_gini(allocation)),
    }


def _bank_stats(bank: dict[int, list]) -> dict:
    counts = [len(pool) for pool in bank.values()]
    if not counts:
        return {
            "classes": 0,
            "total": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
        }
    return {
        "classes": len(counts),
        "total": int(sum(counts)),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "mean": float(mean(counts)),
    }


def _nme_evaluate(
    model: PTModel,
    exemplar_bank: dict[int, list],
    dm: CIFAR100DataModule,
    num_seen_classes: int,
    eval_transform,
    device: torch.device,
) -> tuple[list[float], dict]:
    from studies.runner.cifar100.probe_guided.run import _compute_nme_prototypes

    prototypes = _compute_nme_prototypes(
        model=model,
        exemplar_bank=exemplar_bank,
        num_classes=num_seen_classes,
        eval_transform=eval_transform,
        device=device,
    )
    proto_norm = F.normalize(prototypes, dim=-1)
    row = [0.0] * dm.num_tasks
    per_task_proto_norms: list[float] = []

    model.eval()
    with torch.inference_mode():
        for task_id in range(min(dm.num_tasks, num_seen_classes // dm.classes_per_task)):
            test_loader = dm.get_task_test_loader(task_id)
            correct = 0
            total = 0
            for _, x, y in test_loader:
                x = _transform_raw_batch(x.to(device), eval_transform)
                y = y.to(device)
                feats = model.extract_features(x)
                feat_norm = F.normalize(feats, dim=-1)
                logits = torch.mm(feat_norm, proto_norm[:num_seen_classes].t())
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
            row[task_id] = float(correct / total) if total > 0 else 0.0

    for class_id in range(num_seen_classes):
        per_task_proto_norms.append(float(prototypes[class_id].norm().item()))

    return row, {
        "prototype_count": int((prototypes.abs().sum(dim=1) > 0).sum().item()),
        "prototype_norm_mean": float(mean(per_task_proto_norms)) if per_task_proto_norms else 0.0,
        "prototype_norm_std": float(pstdev(per_task_proto_norms)) if len(per_task_proto_norms) > 1 else 0.0,
    }


def _run_short_audit(method_name: str, cfg: AuditConfig) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    device = _setup_device(cfg.accelerator)
    dm = _make_datamodule(cfg)
    model = _make_model(cfg.classes_per_task)
    method, bank = _make_method(method_name, cfg)
    probe_scorer = ProbeScorer(num_classes=cfg.tasks * cfg.classes_per_task, smoothing=0.0)
    eval_transform = make_eval_transform(dm.config.mean, dm.config.std)
    train_log: list[dict] = []
    accuracy_matrix: list[list[float]] = []

    for task_id in range(cfg.tasks):
        print(f"\n[analysis] task {task_id + 1}/{cfg.tasks} :: {method_name}", flush=True)
        if task_id > 0:
            model.expand_head(cfg.classes_per_task)
            if bank is not None:
                bank.expand(cfg.classes_per_task)

        train_loader, val_loader = dm.get_task_loaders(task_id)
        current_num_classes = (task_id + 1) * cfg.classes_per_task
        replay_class_count = task_id * cfg.classes_per_task

        if hasattr(method, "set_replay_class_count"):
            method.set_replay_class_count(replay_class_count)

        if task_id > 0:
            probe_scores = _compute_probe_scores(
                model=model,
                dm=dm,
                probe_scorer=probe_scorer,
                num_seen_classes=current_num_classes,
                eval_transform=eval_transform,
                device=device,
            )
        else:
            probe_scores = None

        if hasattr(method, "update_allocation"):
            method.update_allocation(probe_scores)
        if hasattr(method, "prune_memory"):
            method.prune_memory()

        allocation = list(getattr(method, "_current_allocation", []) or [])
        bank_obj = getattr(method, "_exemplar_bank", None) if bank is None else bank._bank
        if bank_obj is None:
            bank_stats = {"classes": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0}
        else:
            bank_stats = _bank_stats(bank_obj)

        augment_rng = torch.Generator(device="cpu")
        augment_rng.manual_seed(cfg.seed + task_id)
        train_transform = make_train_transform_from_rng(
            mean=dm.config.mean,
            std=dm.config.std,
            rng=augment_rng,
        )
        pl_module = GhostBankLightningModule(
            model=model,
            method=method,
            bank=bank,
            learning_rate=0.1,
            num_classes=current_num_classes,
            optimizer_name="sgd",
            momentum=0.9,
            weight_decay=5e-4,
            train_transform=train_transform,
            augment_generator=augment_rng,
        )

        quiet = True
        trainer = pl.Trainer(
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            max_epochs=cfg.epochs,
            log_every_n_steps=10,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[ConsoleEpochCallback(prefix=f"seed={cfg.seed} task={task_id}")],
            logger=[CSVLogger(save_dir=str(ROOT / "analysis" / "outputs"), name=f"{method_name}_seed{cfg.seed}_task{task_id}", version="")],
            enable_checkpointing=False,
        )
        trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        device = next(model.parameters()).device
        if method_name == "probe_guided" and hasattr(method, "snapshot_model"):
            method.snapshot_model(pl_module)

        task_row = [0.0] * cfg.tasks
        model.eval()
        with torch.no_grad():
            for prev_task in range(task_id + 1):
                loader = dm.get_task_test_loader(prev_task)
                results = trainer.test(pl_module, dataloaders=loader, verbose=False)
                task_row[prev_task] = float(results[0]["test/acc"]) if results and "test/acc" in results[0] else 0.0
        accuracy_matrix.append(task_row)

        task_probe = {
            "task": task_id,
            "probe_scores": probe_scores,
            "allocation": allocation,
            "allocation_stats": _allocation_stats(allocation),
            "bank_stats": bank_stats,
        }
        if probe_scores is not None:
            task_probe["probe_mean"] = float(mean(probe_scores))
            task_probe["probe_std"] = float(pstdev(probe_scores)) if len(probe_scores) > 1 else 0.0
            task_probe["probe_min"] = float(min(probe_scores))
            task_probe["probe_max"] = float(max(probe_scores))
        train_log.append(task_probe)

        print(f"  allocation: {task_probe['allocation_stats']}", flush=True)
        print(f"  bank: {task_probe['bank_stats']}", flush=True)
        if probe_scores is not None:
            print(
                f"  probe: mean={task_probe['probe_mean']:.4f} std={task_probe['probe_std']:.4f} "
                f"min={task_probe['probe_min']:.4f} max={task_probe['probe_max']:.4f}",
                flush=True,
            )

    linear_avg = average_accuracy(accuracy_matrix)
    linear_forgetting = forgetting(accuracy_matrix) if cfg.tasks > 1 else 0.0
    linear_bwt = backward_transfer(accuracy_matrix) if cfg.tasks > 1 else 0.0
    if hasattr(method, "prune_memory"):
        method.prune_memory()
    nme_row, nme_stats = _nme_evaluate(
        model=model,
        exemplar_bank=getattr(method, "_exemplar_bank", bank._bank if bank is not None else {}),
        dm=dm,
        num_seen_classes=cfg.tasks * cfg.classes_per_task,
        eval_transform=eval_transform,
        device=device,
    )
    nme_avg = float(sum(nme_row) / len(nme_row)) if nme_row else 0.0

    result = {
        "method": method_name,
        "seed": cfg.seed,
        "tasks": cfg.tasks,
        "epochs": cfg.epochs,
        "allocation_history": train_log,
        "accuracy_matrix": accuracy_matrix,
        "nme_row": nme_row,
        "nme_stats": nme_stats,
        "nme_avg": nme_avg,
        "linear_avg": linear_avg,
        "linear_forgetting": linear_forgetting,
        "linear_backward_transfer": linear_bwt,
        "probe_history": probe_scorer.history,
        "probe_raw_scores": probe_scorer.raw_scores,
        "final_bank_stats": _bank_stats(getattr(method, "_exemplar_bank", bank._bank if bank is not None else {})),
    }
    print(json.dumps({
        "method": method_name,
        "nme_avg": nme_avg,
        "final_bank_stats": result["final_bank_stats"],
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Short diagnostic run for probe-guided failure analysis.")
    parser.add_argument("--method", default="probe_guided", choices=["probe_guided", "uniform_replay", "frozen_baseline"])
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--memory-total", type=int, default=2000)
    parser.add_argument("--retrieval-budget", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "outputs")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    args = parser.parse_args()

    if args.accelerator == "cpu" and args.precision == "16-mixed":
        args.precision = "32-true"

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = AuditConfig(
        seed=args.seed,
        tasks=args.tasks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        memory_total=args.memory_total,
        retrieval_budget=args.retrieval_budget,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )
    result = _run_short_audit(args.method, cfg)
    out_path = args.output / f"{args.method}_seed{args.seed}_tasks{args.tasks}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[analysis] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
