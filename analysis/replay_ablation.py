from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.bank.core.probe import ProbeScorer

from analysis.herding_bic_pilot import (
    _allocation_stats,
    _bank_stats,
    _build_allocation,
    _compute_probe_scores,
    _evaluate_linear_matrix,
    _evaluate_nme,
    _herding_select,
    _make_datamodule,
    _setup_device,
    _to_chw,
    _transform_raw_batch,
)
from src.bank.core.retrieval import sample_uniform
from src.methods.base import Method, MethodContext
from src.methods.static_bank.method import _augment_replay
from src.models.ptm import PTModel
from src.training import GhostBankLightningModule
from src.training.callbacks import ConsoleEpochCallback
from studies.runner.cifar100.metrics import average_accuracy, backward_transfer, forgetting


@dataclass(frozen=True)
class AblationConfig:
    seed: int = 13
    tasks: int = 3
    classes_per_task: int = 10
    epochs: int = 5
    batch_size: int = 128
    num_workers: int = 2
    memory_total: int = 2000
    retrieval_budget: int = 64
    probe_split_size: int = 30
    val_split_size: int = 20
    split_seed: int = 13
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "16-mixed"


class ReplayBank:
    """Fixed-memory replay bank with interchangeable selection policy."""

    def __init__(self, seed: int, selection_policy: str) -> None:
        self._pool: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._selected: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._seen_indices: set[int] = set()
        self._rng = random.Random(seed)
        self._selection_policy = selection_policy

    def start_task(self) -> None:
        self._seen_indices.clear()

    def store(self, examples: list, raw_indices: torch.Tensor | None = None) -> None:
        indices = raw_indices.tolist() if raw_indices is not None else None
        for pos, (img, label) in enumerate(examples):
            if indices is not None:
                sample_idx = int(indices[pos])
                if sample_idx in self._seen_indices:
                    continue
                self._seen_indices.add(sample_idx)
            class_id = int(label) if torch.is_tensor(label) else int(label)
            self._pool.setdefault(class_id, []).append((img, label))

    def query(self, budget: int, **kwargs) -> list:
        if budget <= 0:
            return []
        return sample_uniform(self._selected, budget, self._rng)

    def expand(self, num_new_classes: int) -> None:
        return

    def rebuild_selected(
        self,
        model: PTModel,
        allocation: list[int],
        eval_transform,
        device: torch.device,
        chunk_size: int = 256,
    ) -> dict[str, float]:
        selected: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
            class_id: [] for class_id in range(len(allocation))
        }
        total_selected = 0
        model.eval()
        model_device = next(model.parameters()).device
        with torch.inference_mode():
            for class_id, quota in enumerate(allocation):
                pool = self._pool.get(class_id, [])
                if quota <= 0 or not pool:
                    continue
                if quota >= len(pool):
                    selected[class_id] = list(pool)
                    total_selected += len(pool)
                    continue

                if self._selection_policy == "random":
                    pick = sorted(self._rng.sample(range(len(pool)), k=quota))
                elif self._selection_policy == "herding":
                    feats_chunks: list[torch.Tensor] = []
                    for start in range(0, len(pool), chunk_size):
                        end = min(start + chunk_size, len(pool))
                        raw_batch = torch.stack(
                            [_to_chw(item[0]) for item in pool[start:end]],
                            dim=0,
                        )
                        images_t = _transform_raw_batch(raw_batch, eval_transform).to(model_device)
                        feats_chunks.append(model.extract_features(images_t).detach().cpu())
                    feats_all = torch.cat(feats_chunks, dim=0)
                    pick = _herding_select(feats_all, quota)
                else:
                    raise ValueError(f"Unknown selection policy: {self._selection_policy}")

                selected[class_id] = [pool[i] for i in pick]
                total_selected += len(selected[class_id])

        self._selected = selected
        return _bank_stats(self)

    @property
    def selected(self) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        return self._selected


class ReplayMethod(Method):
    def __init__(self, retrieval_budget: int = 64, warmup_steps: int = 0) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: ReplayBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch
        if bank is None:
            return F.cross_entropy(pl_module(x), y)

        if context is not None and context.raw_x is not None and context.raw_y is not None:
            bank.store(
                list(zip(context.raw_x, context.raw_y.tolist())),
                raw_indices=context.raw_indices,
            )
        else:
            bank.store([(x[i], y[i]) for i in range(len(y))])

        if pl_module.global_step >= self.warmup_steps:
            replay_items = bank.query(self.retrieval_budget)
            replay_x = _augment_replay(
                replay_items,
                transform=context.train_transform if context is not None else None,
                rng=context.augment_rng if context is not None else None,
                device=y.device,
            )
            replay_y = (
                torch.tensor(
                    [int(item[1]) for item in replay_items],
                    device=y.device,
                    dtype=torch.long,
                )
                if replay_items
                else None
            )
            if replay_x is not None and replay_y is not None and replay_y.numel() > 0:
                x = torch.cat([x, replay_x], dim=0)
                y = torch.cat([y, replay_y], dim=0)

        return F.cross_entropy(pl_module(x), y)


def _build_variants(method: str) -> list[tuple[str, str, str]]:
    variants = [
        ("uniform_random", "uniform", "random"),
        ("probe_random", "probe", "random"),
        ("uniform_herding", "uniform", "herding"),
        ("probe_herding", "probe", "herding"),
    ]
    if method == "compare":
        return variants
    for name, allocation_mode, selection_policy in variants:
        if name == method:
            return [(name, allocation_mode, selection_policy)]
    raise ValueError(f"Unknown method: {method}")


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman_correlation(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n < 2:
        return None
    a_arr = np.asarray(a[:n], dtype=np.float64)
    b_arr = np.asarray(b[:n], dtype=np.float64)
    if np.var(a_arr) < 1e-12 or np.var(b_arr) < 1e-12:
        return None
    a_rank = _rank_values(a_arr)
    b_rank = _rank_values(b_arr)
    corr = np.corrcoef(a_rank, b_rank)[0, 1]
    if not np.isfinite(corr):
        return None
    return float(corr)


def _compute_per_class_forgetting(
    acc_matrix: list[list[float]],
    classes_per_task: int,
) -> list[float]:
    if len(acc_matrix) < 2:
        return []

    n_cols = max(len(row) for row in acc_matrix)
    rect = np.full((len(acc_matrix), n_cols), np.nan)
    for i, row in enumerate(acc_matrix):
        rect[i, : len(row)] = row

    per_task_forgetting: list[float] = []
    for t in range(n_cols):
        col = rect[:, t]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            per_task_forgetting.append(float(np.nanmax(valid) - valid[-1]))
        else:
            per_task_forgetting.append(0.0)

    per_class: list[float] = []
    for value in per_task_forgetting:
        per_class.extend([value] * classes_per_task)
    return per_class


def _compute_probe_correlation(
    probe_scores: list[float],
    acc_matrix: list[list[float]],
    classes_per_task: int,
) -> float | None:
    per_class_forgetting = _compute_per_class_forgetting(acc_matrix, classes_per_task)
    return _spearman_correlation(per_class_forgetting, probe_scores)


def _run_variant(
    variant_name: str,
    allocation_mode: str,
    selection_policy: str,
    cfg: AblationConfig,
    output_dir: Path,
) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    device = _setup_device(cfg.accelerator)
    dm = _make_datamodule(
        SimpleNamespace(
            seed=cfg.seed,
            tasks=cfg.tasks,
            classes_per_task=cfg.classes_per_task,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            memory_total=cfg.memory_total,
            probe_split_size=cfg.probe_split_size,
            val_split_size=cfg.val_split_size,
            split_seed=cfg.split_seed,
        )
    )
    model = PTModel(
        backbone="resnet18",
        pretrained=True,
        num_classes=cfg.classes_per_task,
        freeze_backbone=True,
        embedding_dim=512,
        classifier_mode="linear",
    )
    method = ReplayMethod(retrieval_budget=cfg.retrieval_budget, warmup_steps=0)
    bank = ReplayBank(seed=cfg.seed, selection_policy=selection_policy)
    probe_scorer = ProbeScorer(num_classes=cfg.tasks * cfg.classes_per_task, smoothing=0.0)
    eval_transform = dm._shared_eval_transform() if hasattr(dm, "_shared_eval_transform") else None
    if eval_transform is None:
        from src.data.cifar100.transforms import make_eval_transform

        eval_transform = make_eval_transform(dm.config.mean, dm.config.std)

    accuracy_matrix: list[list[float]] = []
    nme_matrix: list[list[float]] = []
    allocation_history: list[dict] = []
    probe_history: list[list[float]] = []
    probe_scores: list[float] | None = None

    for task_id in range(cfg.tasks):
        print(f"\n[ablation] task {task_id + 1}/{cfg.tasks} :: {variant_name}", flush=True)
        bank.start_task()
        if task_id > 0:
            model.expand_head(cfg.classes_per_task)

        train_loader, val_loader = dm.get_task_loaders(task_id)
        current_num_classes = (task_id + 1) * cfg.classes_per_task

        allocation = _build_allocation(
            allocation_mode,
            probe_scores,
            current_num_classes,
            cfg.memory_total,
            floor=1,
            gamma=0.5,
            beta=1.0,
        )
        allocation_history.append(
            {
                "task": task_id,
                "allocation": allocation,
                "allocation_stats": _allocation_stats(allocation),
                "probe_scores": probe_scores,
            }
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
            train_transform=dm._shared_train_transform() if hasattr(dm, "_shared_train_transform") else None,
            augment_generator=torch.Generator(device="cpu").manual_seed(cfg.seed + task_id),
        )

        csv_logger = CSVLogger(
            save_dir=str(output_dir),
            name=f"{variant_name}_seed{cfg.seed}_task{task_id}",
            version="",
        )
        trainer = pl.Trainer(
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            max_epochs=cfg.epochs,
            log_every_n_steps=10,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[ConsoleEpochCallback(prefix=f"seed={cfg.seed} task={task_id}")],
            logger=[csv_logger],
            enable_checkpointing=False,
        )
        trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        model.to(device)

        selected_stats = bank.rebuild_selected(
            model=model,
            allocation=allocation,
            eval_transform=eval_transform,
            device=device,
        )
        allocation_history[-1]["selected_stats"] = selected_stats
        allocation_history[-1]["bank_stats"] = _bank_stats(bank)

        raw_row = _evaluate_linear_matrix(
            model=model,
            dm=dm,
            task_id=task_id,
            device=device,
            eval_transform=eval_transform,
            bias_params=None,
        )
        nme_row = _evaluate_nme(
            model=model,
            exemplar_bank=bank.selected,
            dm=dm,
            num_seen_classes=current_num_classes,
            current_task_id=task_id,
            eval_transform=eval_transform,
            device=device,
        )
        accuracy_matrix.append(raw_row)
        nme_matrix.append(nme_row)

        probe_scores = _compute_probe_scores(
            model=model,
            dm=dm,
            probe_scorer=probe_scorer,
            num_seen_classes=current_num_classes,
            eval_transform=eval_transform,
            device=device,
        )
        probe_history.append(probe_scores)
        model.train()

        print(f"  allocation: {allocation_history[-1]['allocation_stats']}", flush=True)
        print(f"  selected: {selected_stats}", flush=True)

    raw_final = average_accuracy(accuracy_matrix)
    raw_forgetting = forgetting(accuracy_matrix) if cfg.tasks > 1 else 0.0
    raw_bwt = backward_transfer(accuracy_matrix) if cfg.tasks > 1 else 0.0
    nme_final = average_accuracy(nme_matrix)
    probe_r = _compute_probe_correlation(
        probe_scores=probe_history[-1] if probe_history else [],
        acc_matrix=accuracy_matrix,
        classes_per_task=cfg.classes_per_task,
    )

    result = {
        "variant": variant_name,
        "allocation_mode": allocation_mode,
        "selection_policy": selection_policy,
        "seed": cfg.seed,
        "tasks": cfg.tasks,
        "epochs": cfg.epochs,
        "raw_accuracy_matrix": accuracy_matrix,
        "nme_accuracy_matrix": nme_matrix,
        "raw/test/avg_acc": raw_final,
        "raw/test/forgetting": raw_forgetting,
        "raw/test/backward_transfer": raw_bwt,
        "nme/test/avg_acc": nme_final,
        "probe/spearman_r": probe_r,
        "allocation_history": allocation_history,
        "probe_history": probe_history,
        "selected_bank_stats": _bank_stats(bank),
    }
    print(
        json.dumps(
            {
                "variant": variant_name,
                "raw/test/avg_acc": raw_final,
                "nme/test/avg_acc": nme_final,
                "probe/spearman_r": probe_r,
                "selected_bank_stats": result["selected_bank_stats"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: uniform vs probe-guided allocation, random vs herding selection.",
    )
    parser.add_argument(
        "--method",
        default="compare",
        choices=[
            "compare",
            "uniform_random",
            "probe_random",
            "uniform_herding",
            "probe_herding",
        ],
    )
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--memory-total", type=int, default=2000)
    parser.add_argument("--retrieval-budget", type=int, default=64)
    parser.add_argument("--probe-split-size", type=int, default=30)
    parser.add_argument("--val-split-size", type=int, default=20)
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "outputs")
    args = parser.parse_args()

    if args.accelerator == "cpu" and args.precision == "16-mixed":
        args.precision = "32-true"

    cfg = AblationConfig(
        seed=args.seed,
        tasks=args.tasks,
        classes_per_task=10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        memory_total=args.memory_total,
        retrieval_budget=args.retrieval_budget,
        probe_split_size=args.probe_split_size,
        val_split_size=args.val_split_size,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    variants = _build_variants(args.method)
    results: dict[str, dict] = {}
    for variant_name, allocation_mode, selection_policy in variants:
        results[variant_name] = _run_variant(
            variant_name=variant_name,
            allocation_mode=allocation_mode,
            selection_policy=selection_policy,
            cfg=cfg,
            output_dir=args.output,
        )

    out_path = args.output / f"replay_ablation_seed{args.seed}_tasks{args.tasks}_{args.method}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[analysis] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
