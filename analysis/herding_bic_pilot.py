from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bank.core.allocator import allocate_fixed_total, allocate_uniform_fixed_total
from src.bank.core.probe import ProbeScorer
from src.bank.core.retrieval import sample_uniform
from src.data.cifar100 import CIFAR100Config, CIFAR100DataModule
from src.methods.baseline import BaselineMethod
from src.methods.base import Method, MethodContext
from src.methods.static_bank.method import _augment_replay
from src.models.ptm import PTModel
from src.training import GhostBankLightningModule
from src.training.callbacks import ConsoleEpochCallback
from studies.runner.cifar100.metrics import average_accuracy, backward_transfer, forgetting


@dataclass(frozen=True)
class PilotConfig:
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
    bias_steps: int = 50
    bias_samples_per_class: int = 20
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


def _make_datamodule(cfg: PilotConfig) -> CIFAR100DataModule:
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


def _to_chw(raw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    if raw.dim() == 3 and raw.shape[-1] == 3 and raw.shape[0] != 3:
        return raw.permute(2, 0, 1).contiguous()
    return raw.contiguous()


def _herding_select(features: torch.Tensor, budget: int) -> list[int]:
    """Greedy herding that matches the class mean."""
    n = features.shape[0]
    if n == 0 or budget <= 0:
        return []
    budget = min(budget, n)
    class_mean = features.mean(dim=0)
    selected: list[int] = []
    selected_sum = torch.zeros_like(class_mean)
    chosen: set[int] = set()
    for _ in range(budget):
        best_idx = -1
        best_dist = float("inf")
        for i in range(n):
            if i in chosen:
                continue
            cand_mean = (selected_sum + features[i]) / (len(selected) + 1)
            dist = torch.norm(class_mean - cand_mean).item()
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        chosen.add(best_idx)
        selected_sum = selected_sum + features[best_idx]
    return selected


def _balanced_validation_batch(
    dm: CIFAR100DataModule,
    num_seen_classes: int,
    samples_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    val_loaders = dm.get_val_loaders()
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for c in range(num_seen_classes):
        if c not in val_loaders:
            continue
        cls_images, cls_targets = val_loaders[c]
        n = min(samples_per_class, cls_images.shape[0])
        if n <= 0:
            continue
        idx = torch.randperm(cls_images.shape[0], generator=generator)[:n]
        images.append(cls_images[idx])
        targets.append(cls_targets[idx])
    if not images:
        raise RuntimeError("No validation samples available for bias correction.")
    return torch.cat(images, dim=0), torch.cat(targets, dim=0)


def _apply_bias_correction(
    logits: torch.Tensor,
    old_classes: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    if old_classes <= 0:
        return logits
    corrected = logits.clone()
    corrected[:, :old_classes] = alpha * corrected[:, :old_classes] + beta
    return corrected


class HerdingReplayBank:
    """Full pool + selected pool replay bank.

    The pool keeps all unique task samples seen so far.
    The selected pool stores the current herding selection used for replay.
    """

    def __init__(self, seed: int) -> None:
        self._pool: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._selected: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._seen_indices: set[int] = set()
        self._rng = random.Random(seed)

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
            cid = int(label) if torch.is_tensor(label) else int(label)
            if cid not in self._pool:
                self._pool[cid] = []
            self._pool[cid].append((img, label))

    def query(self, budget: int, **kwargs) -> list:
        if budget <= 0:
            return []
        return sample_uniform(self._selected, budget, self._rng)

    def expand(self, num_new_classes: int) -> None:
        # No-op: pools grow dynamically by class id.
        return

    def rebuild_selected(
        self,
        model: PTModel,
        allocation: list[int],
        eval_transform,
        device: torch.device,
        chunk_size: int = 256,
    ) -> dict[str, float]:
        selected: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        total_selected = 0
        classes_used = 0
        model.eval()
        with torch.inference_mode():
            for class_id, quota in enumerate(allocation):
                pool = self._pool.get(class_id, [])
                if quota <= 0 or not pool:
                    selected[class_id] = []
                    continue
                classes_used += 1
                if quota >= len(pool):
                    selected[class_id] = list(pool)
                    total_selected += len(pool)
                    continue

                feats_chunks: list[torch.Tensor] = []
                for start in range(0, len(pool), chunk_size):
                    end = min(start + chunk_size, len(pool))
                    raw_batch = torch.stack([_to_chw(item[0]) for item in pool[start:end]], dim=0)
                    images_t = _transform_raw_batch(raw_batch, eval_transform).to(device)
                    feats = model.extract_features(images_t).detach().cpu()
                    feats_chunks.append(feats)
                feats_all = torch.cat(feats_chunks, dim=0)
                pick = _herding_select(feats_all, quota)
                selected[class_id] = [pool[i] for i in pick]
                total_selected += len(selected[class_id])

        self._selected = selected
        return {
            "classes": classes_used,
            "total": total_selected,
            "min": min((len(v) for v in selected.values()), default=0),
            "max": max((len(v) for v in selected.values()), default=0),
            "mean": float(total_selected / max(1, classes_used)),
        }

    def selected_stats(self) -> dict:
        counts = [len(pool) for pool in self._selected.values()]
        if not counts:
            return {"classes": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0}
        return {
            "classes": len(counts),
            "total": int(sum(counts)),
            "min": int(min(counts)),
            "max": int(max(counts)),
            "mean": float(mean(counts)),
        }

    @property
    def selected(self) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        return self._selected


class HerdingReplayMethod(Method):
    def __init__(self, retrieval_budget: int = 64, warmup_steps: int = 0) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: HerdingReplayBank | None = None,
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


def _build_allocation(
    mode: str,
    probe_scores: list[float] | None,
    num_seen_classes: int,
    memory_total: int,
    floor: int = 1,
    gamma: float = 0.5,
    beta: float = 1.0,
) -> list[int]:
    if num_seen_classes <= 0:
        return []
    if mode == "uniform" or probe_scores is None:
        return allocate_uniform_fixed_total(
            num_classes=num_seen_classes,
            total_budget=memory_total,
            floor=floor,
        )
    active_scores = list(probe_scores[:num_seen_classes])
    if active_scores:
        min_v = min(active_scores)
        max_v = max(active_scores)
        if max_v - min_v >= 1e-12:
            active_scores = [
                (score - min_v) / (max_v - min_v) for score in active_scores
            ]
        else:
            active_scores = [0.0] * len(active_scores)
    return allocate_fixed_total(
        num_classes=num_seen_classes,
        total_budget=memory_total,
        probe_scores=active_scores,
        gamma=gamma,
        beta=beta,
        floor=floor,
    )


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


def _evaluate_linear_matrix(
    model: PTModel,
    dm: CIFAR100DataModule,
    task_id: int,
    device: torch.device,
    eval_transform,
    bias_params: tuple[int, float, float] | None = None,
) -> list[float]:
    row = [0.0] * dm.num_tasks
    model.eval()
    with torch.inference_mode():
        for prev_task in range(task_id + 1):
            loader = dm.get_task_test_loader(prev_task)
            correct = 0
            total = 0
            for _, x, y in loader:
                x = _transform_raw_batch(x.to(device), eval_transform)
                y = y.to(device)
                logits = model(x)
                if bias_params is not None:
                    old_classes, alpha, beta = bias_params
                    logits = _apply_bias_correction(logits, old_classes, alpha, beta)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
            row[prev_task] = float(correct / total) if total > 0 else 0.0
    return row


def _compute_nme_prototypes(
    model: PTModel,
    exemplar_bank: dict[int, list],
    num_classes: int,
    eval_transform,
    device: torch.device,
) -> torch.Tensor:
    feat_dim = model.embedding_dim
    prototypes = torch.zeros(num_classes, feat_dim, device=device)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    exemplars: list[tuple[int, torch.Tensor]] = []
    for class_id in range(num_classes):
        for item in exemplar_bank.get(class_id, []):
            raw = item[0]
            exemplars.append((class_id, _to_chw(raw)))

    if not exemplars:
        return prototypes

    batch_size = 256
    for start in range(0, len(exemplars), batch_size):
        end = min(start + batch_size, len(exemplars))
        chunk = exemplars[start:end]
        labels = torch.tensor([cid for cid, _ in chunk], device=device, dtype=torch.long)
        raw_batch = torch.stack([raw for _, raw in chunk], dim=0)
        images_t = _transform_raw_batch(raw_batch, eval_transform).to(device)
        with torch.inference_mode():
            feats = model.extract_features(images_t)
        prototypes.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones(labels.shape[0], device=device, dtype=torch.long))

    nonzero = counts > 0
    if nonzero.any():
        prototypes[nonzero] = prototypes[nonzero] / counts[nonzero].unsqueeze(1).to(prototypes.dtype)
    return prototypes


def _evaluate_nme(
    model: PTModel,
    exemplar_bank: dict[int, list],
    dm: CIFAR100DataModule,
    num_seen_classes: int,
    current_task_id: int,
    eval_transform,
    device: torch.device,
) -> list[float]:
    prototypes = _compute_nme_prototypes(
        model=model,
        exemplar_bank=exemplar_bank,
        num_classes=num_seen_classes,
        eval_transform=eval_transform,
        device=device,
    )
    proto_norm = F.normalize(prototypes, dim=-1)
    row = [0.0] * dm.num_tasks
    model.eval()
    with torch.inference_mode():
        for task_id in range(current_task_id + 1):
            loader = dm.get_task_test_loader(task_id)
            correct = 0
            total = 0
            for _, x, y in loader:
                x = _transform_raw_batch(x.to(device), eval_transform)
                y = y.to(device)
                feats = model.extract_features(x)
                feat_norm = F.normalize(feats, dim=-1)
                logits = torch.mm(feat_norm, proto_norm[:num_seen_classes].t())
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
            row[task_id] = float(correct / total) if total > 0 else 0.0
    return row


def _fit_bias_correction(
    model: PTModel,
    dm: CIFAR100DataModule,
    num_seen_classes: int,
    old_classes: int,
    eval_transform,
    device: torch.device,
    steps: int,
    samples_per_class: int,
    seed: int,
) -> tuple[float, float]:
    if old_classes <= 0:
        return 1.0, 0.0

    images, targets = _balanced_validation_batch(
        dm,
        num_seen_classes,
        samples_per_class,
        seed=seed,
    )
    images = images.to(device)
    targets = targets.to(device)
    images_t = _transform_raw_batch(images, eval_transform)

    alpha = torch.nn.Parameter(torch.tensor(1.0, device=device))
    beta = torch.nn.Parameter(torch.tensor(0.0, device=device))
    optim = torch.optim.Adam([alpha, beta], lr=0.05)

    model.eval()
    with torch.inference_mode():
        base_logits = model(images_t).detach()

    for _ in range(steps):
        corrected = base_logits.clone()
        corrected[:, :old_classes] = alpha * corrected[:, :old_classes] + beta
        loss = F.cross_entropy(corrected, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return float(alpha.detach().item()), float(beta.detach().item())


def _bank_stats(bank: HerdingReplayBank) -> dict:
    counts = [len(pool) for pool in bank.selected.values()]
    if not counts:
        return {"classes": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "classes": len(counts),
        "total": int(sum(counts)),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "mean": float(mean(counts)),
    }


def _allocation_stats(allocation: list[int]) -> dict:
    if not allocation:
        return {"sum": 0, "min": 0, "max": 0, "entropy": 0.0, "gini": 0.0}
    total = sum(allocation)
    if total <= 0:
        return {"sum": 0, "min": 0, "max": 0, "entropy": 0.0, "gini": 0.0}
    probs = [a / total for a in allocation if a > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    xs = sorted(a for a in allocation if a >= 0)
    n = len(xs)
    cum = 0.0
    for i, v in enumerate(xs, 1):
        cum += i * v
    gini = (2.0 * cum) / (n * total) - (n + 1) / n if n > 0 else 0.0
    return {
        "sum": int(total),
        "min": int(min(allocation)),
        "max": int(max(allocation)),
        "entropy": float(entropy),
        "gini": float(gini),
    }


def _run_variant(
    variant: str,
    cfg: PilotConfig,
    output_dir: Path,
) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    device = _setup_device(cfg.accelerator)
    dm = _make_datamodule(cfg)
    model = PTModel(
        backbone="resnet18",
        pretrained=True,
        num_classes=cfg.classes_per_task,
        freeze_backbone=True,
        embedding_dim=512,
        classifier_mode="linear",
    )
    replay_method = HerdingReplayMethod(retrieval_budget=cfg.retrieval_budget, warmup_steps=0)
    baseline_method = BaselineMethod()
    bank = HerdingReplayBank(seed=cfg.seed)
    probe_scorer = ProbeScorer(num_classes=cfg.tasks * cfg.classes_per_task, smoothing=0.0)
    eval_transform = dm._shared_eval_transform() if hasattr(dm, "_shared_eval_transform") else None
    if eval_transform is None:
        from src.data.cifar100.transforms import make_eval_transform
        eval_transform = make_eval_transform(dm.config.mean, dm.config.std)

    accuracy_matrix: list[list[float]] = []
    bic_matrix: list[list[float]] = []
    nme_matrix: list[list[float]] = []
    allocation_history: list[dict] = []
    bias_history: list[dict] = []
    probe_history: list[list[float]] = []
    last_bias: tuple[int, float, float] | None = None

    for task_id in range(cfg.tasks):
        print(f"\n[groundup] task {task_id + 1}/{cfg.tasks} :: {variant}", flush=True)
        bank.start_task()
        if task_id > 0:
            model.expand_head(cfg.classes_per_task)

        train_loader, val_loader = dm.get_task_loaders(task_id)
        current_num_classes = (task_id + 1) * cfg.classes_per_task

        if variant == "probe_guided_herding_bic" and task_id > 0:
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
        if probe_scores is not None:
            probe_history.append(probe_scores)

        allocation_mode = "probe" if variant == "probe_guided_herding_bic" and probe_scores is not None else "uniform"
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
            method=replay_method if variant != "frozen_baseline" else baseline_method,
            bank=bank if variant != "frozen_baseline" else None,
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
            name=f"{variant}_seed{cfg.seed}_task{task_id}",
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

        select_stats = bank.rebuild_selected(
            model=model,
            allocation=allocation,
            eval_transform=eval_transform,
            device=device,
        )
        allocation_history[-1]["selected_stats"] = select_stats
        allocation_history[-1]["bank_stats"] = _bank_stats(bank)

        old_classes = task_id * cfg.classes_per_task
        bias_params = _fit_bias_correction(
            model=model,
            dm=dm,
            num_seen_classes=current_num_classes,
            old_classes=old_classes,
            eval_transform=eval_transform,
            device=device,
            steps=cfg.bias_steps,
            samples_per_class=cfg.bias_samples_per_class,
            seed=cfg.seed + task_id,
        )
        last_bias = (old_classes, bias_params[0], bias_params[1])
        bias_history.append(
            {
                "task": task_id,
                "old_classes": old_classes,
                "alpha": bias_params[0],
                "beta": bias_params[1],
            }
        )

        raw_row = _evaluate_linear_matrix(
            model=model,
            dm=dm,
            task_id=task_id,
            device=device,
            eval_transform=eval_transform,
            bias_params=None,
        )
        bic_row = _evaluate_linear_matrix(
            model=model,
            dm=dm,
            task_id=task_id,
            device=device,
            eval_transform=eval_transform,
            bias_params=last_bias,
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
        bic_matrix.append(bic_row)
        nme_matrix.append(nme_row)

        print(f"  allocation: {allocation_history[-1]['allocation_stats']}", flush=True)
        print(f"  selected: {select_stats}", flush=True)
        print(f"  bias: alpha={bias_params[0]:.4f} beta={bias_params[1]:.4f}", flush=True)

    raw_final = average_accuracy(accuracy_matrix)
    raw_forgetting = forgetting(accuracy_matrix) if cfg.tasks > 1 else 0.0
    raw_bwt = backward_transfer(accuracy_matrix) if cfg.tasks > 1 else 0.0
    bic_final = average_accuracy(bic_matrix)
    bic_forgetting = forgetting(bic_matrix) if cfg.tasks > 1 else 0.0
    bic_bwt = backward_transfer(bic_matrix) if cfg.tasks > 1 else 0.0
    nme_final = average_accuracy(nme_matrix)

    result = {
        "variant": variant,
        "seed": cfg.seed,
        "tasks": cfg.tasks,
        "epochs": cfg.epochs,
        "raw_accuracy_matrix": accuracy_matrix,
        "bic_accuracy_matrix": bic_matrix,
        "nme_accuracy_matrix": nme_matrix,
        "raw/test/avg_acc": raw_final,
        "raw/test/forgetting": raw_forgetting,
        "raw/test/backward_transfer": raw_bwt,
        "bic/test/avg_acc": bic_final,
        "bic/test/forgetting": bic_forgetting,
        "bic/test/backward_transfer": bic_bwt,
        "nme/test/avg_acc": nme_final,
        "allocation_history": allocation_history,
        "bias_history": bias_history,
        "probe_history": probe_history,
        "selected_bank_stats": _bank_stats(bank),
    }
    print(
        json.dumps(
            {
                "variant": variant,
                "raw/test/avg_acc": raw_final,
                "bic/test/avg_acc": bic_final,
                "nme/test/avg_acc": nme_final,
                "selected_bank_stats": result["selected_bank_stats"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-up pilot: herding + BiC on PTM CIFAR-100.")
    parser.add_argument("--method", default="compare", choices=[
        "compare",
        "probe_guided_herding_bic",
        "uniform_herding_bic",
        "frozen_baseline",
    ])
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--memory-total", type=int, default=2000)
    parser.add_argument("--retrieval-budget", type=int, default=64)
    parser.add_argument("--probe-split-size", type=int, default=30)
    parser.add_argument("--val-split-size", type=int, default=20)
    parser.add_argument("--bias-steps", type=int, default=50)
    parser.add_argument("--bias-samples-per-class", type=int, default=20)
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "outputs")
    args = parser.parse_args()

    if args.accelerator == "cpu" and args.precision == "16-mixed":
        args.precision = "32-true"

    cfg = PilotConfig(
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
        bias_steps=args.bias_steps,
        bias_samples_per_class=args.bias_samples_per_class,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    variants = (
        ["probe_guided_herding_bic", "uniform_herding_bic", "frozen_baseline"]
        if args.method == "compare"
        else [args.method]
    )
    results: dict[str, dict] = {}
    for variant in variants:
        results[variant] = _run_variant(variant, cfg, args.output)

    out_path = args.output / f"herding_bic_seed{args.seed}_tasks{args.tasks}_{args.method}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[analysis] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
