"""
Full 10-task CIFAR-100 benchmark: SDWR-v2 comparison.

Compares 3 methods under the original experimental protocol:
  1. Baseline        : no replay
  2. Uniform + CE    : uniform retrieval + CE (StaticBank)
  3. Uniform + Distill: uniform retrieval + distillation (α=0)

Settings (from configs/training/cifar100.yaml):
  - 10 tasks × 10 classes = 100 classes total
  - R=64, S=200, 70 epochs/task
  - SGD LR=0.1, momentum=0.9, weight_decay=5e-4
  - Flat LR (cosine scheduler removed)

Run from repo root:
  python studies/analysis/benchmark_sdwr_v2.py

Expected runtime: ~7 hours on a single GPU.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.cifar100 import CIFAR100DataModule, CIFAR100Config
from src.data.cifar100.transforms import make_train_transform
from src.models import ResNet
from src.bank.core.retrieval import sample_by_allocation

SEED = 13
N_TASKS = 10
N_CLASSES_PER_TASK = 10
N_CLASSES_TOTAL = 100
EPOCHS_PER_TASK = 70
BATCH_SIZE = 128
RETRIEVAL_BUDGET = 64
CAPACITY_PER_CLASS = 200
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
GRADIENT_CLIP = 1.0

# Distillation hyperparameters
LAMBDA = 1.0
TEMPERATURE = 2.0


# ── data ──────────────────────────────────────────────────────────────────
def load_cifar100_data() -> tuple[list, list, list]:
    cfg = CIFAR100Config(
        root="./data/cifar100",
        seed=SEED,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batch_size=BATCH_SIZE,
        num_tasks=10,
        classes_per_task=10,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )
    dm = CIFAR100DataModule(cfg)
    dm.setup("fit")
    train_images = dm._train_images
    train_targets = dm._train_targets
    class_images: list[list[torch.Tensor]] = [[] for _ in range(100)]
    for i in range(len(train_targets)):
        c = int(train_targets[i])
        class_images[c].append(train_images[i])
    return class_images, dm.config.mean, dm.config.std


def prepare_val_data(class_images: list[list[torch.Tensor]]) -> list:
    val_data = []
    for t in range(N_TASKS):
        start = t * N_CLASSES_PER_TASK
        end = start + N_CLASSES_PER_TASK
        xs, ys = [], []
        for c in range(start, end):
            imgs = class_images[c]
            split = int(len(imgs) * 0.8)
            xs.extend(imgs[split:])
            ys.extend([c] * (len(imgs) - split))
        val_data.append(TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in xs]),
            torch.tensor(ys, dtype=torch.long),
        ))
    return val_data


# ── model ─────────────────────────────────────────────────────────────────
def create_model(num_classes: int) -> ResNet:
    return ResNet(num_classes=num_classes, base_filters=64)


# ── bank ──────────────────────────────────────────────────────────────────
class ReplayBank:
    def __init__(self, num_classes: int, capacity: int, seed: int):
        self._bank: dict[int, list] = {}
        self._capacity = capacity
        self._rng = random.Random(seed)

    def store(self, examples: list):
        for x, y in examples:
            c = int(y)
            if c not in self._bank:
                self._bank[c] = []
            pool = self._bank[c]
            if len(pool) < self._capacity:
                pool.append((x, y))
            else:
                idx = self._rng.randint(0, len(pool) - 1)
                pool[idx] = (x, y)

    def query_uniform(self, budget: int) -> list:
        classes = [c for c, pool in self._bank.items() if pool]
        if not classes:
            return []
        pool_sizes = [len(self._bank[c]) for c in classes]
        total = sum(pool_sizes)
        if total == 0:
            return []
        selected = self._rng.choices(classes, weights=pool_sizes, k=budget)
        return [self._rng.choice(self._bank[cls]) for cls in selected]

    def query_uniform_by_class(self, budget: int, num_classes: int) -> list:
        if num_classes == 0:
            return []
        base = budget // num_classes
        extra = budget - base * num_classes
        alloc = [base + (1 if i < extra else 0) for i in range(num_classes)]
        return sample_by_allocation(self._bank, alloc, self._rng)

    def snapshot_logits(self, model, transform, device):
        for c in list(self._bank.keys()):
            pool = self._bank[c]
            new_pool = []
            for item in pool:
                raw, label = item[0], item[1]
                if raw.dim() == 3 and raw.shape[-1] == 3:
                    t = raw.permute(2, 0, 1).contiguous().float() / 255.0
                elif raw.dtype == torch.uint8:
                    t = raw.float() / 255.0
                else:
                    t = raw.float() if torch.is_tensor(raw) else torch.as_tensor(raw).float()
                x_aug = transform(t).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit = model(x_aug).cpu().squeeze(0)
                new_pool.append((raw, label, logit))
            self._bank[c] = new_pool


# ── helpers ───────────────────────────────────────────────────────────────
def _augment(items: list, transform, device):
    if not items:
        return None, None, None
    xs, ys, tlogits = [], [], []
    for item in items:
        if len(item) == 3:
            raw, label, logit = item
            tlogits.append(logit)
        else:
            raw, label = item
            tlogits.append(None)
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        if raw.dim() == 3 and raw.shape[-1] == 3:
            t = raw.permute(2, 0, 1).contiguous().float() / 255.0
        else:
            t = raw.float() / 255.0 if raw.dtype == torch.uint8 else raw
        if transform is not None:
            t = transform(t)
        xs.append(t)
        ys.append(int(label))
    xs_s = torch.stack(xs).to(device)
    ys_s = torch.tensor(ys, device=device, dtype=torch.long)
    have_logits = any(lt is not None for lt in tlogits)
    if have_logits:
        logit_list = []
        for lt in tlogits:
            if lt is not None:
                logit_list.append(lt.to(device))
            else:
                logit_list.append(torch.zeros(1, device=device))
        tlogits_s = torch.stack(logit_list)
    else:
        tlogits_s = None
    return xs_s, ys_s, tlogits_s


def _distillation_loss(current_logits, target_logits, temperature):
    return F.kl_div(
        F.log_softmax(current_logits / temperature, dim=1),
        F.softmax(target_logits / temperature, dim=1),
        reduction='none',
    ).sum(dim=1) * (temperature ** 2)


def _make_optim(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


def _eval_all(model, val_data, up_to_task, device):
    """Evaluate on all classes seen so far."""
    model.eval()
    all_x, all_y = [], []
    for t in range(up_to_task + 1):
        for x, y in DataLoader(val_data[t], 256, shuffle=False):
            all_x.append(x); all_y.append(y)
    if not all_x:
        model.train()
        return 0.0
    ds = TensorDataset(torch.cat(all_x), torch.cat(all_y))
    correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(ds, 256, shuffle=False):
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    model.train()
    return correct / total if total > 0 else 0.0


def _per_task_acc(model, val_data, device):
    """Return per-class accuracies across all tasks seen so far."""
    model.eval()
    correct, total = {}, {}
    all_seen = set()
    for t in range(N_TASKS):
        for x, y in DataLoader(val_data[t], 256, shuffle=False):
            all_seen.update(y.tolist())
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for i in range(len(y)):
                c = int(y[i])
                correct[c] = correct.get(c, 0) + (preds[i] == y[i]).item()
                total[c] = total.get(c, 0) + 1
    accs = [correct[c] / total[c] if total.get(c, 0) > 0 else 0.0 for c in sorted(all_seen)]
    model.train()
    return accs


# ──────────────────────────────────────────────────────────────────────────
# Method 1: Baseline (no replay)
# ──────────────────────────────────────────────────────────────────────────

def run_baseline(device, train_data_class, val_data):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    history = []

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)

        start = task_id * N_CLASSES_PER_TASK
        end = start + N_CLASSES_PER_TASK
        xs, ys = [], []
        for c in range(start, end):
            xs.extend(train_data_class[c])
            ys.extend([c] * len(train_data_class[c]))
        ds = TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in xs]),
            torch.tensor(ys, dtype=torch.long),
        )
        loader = DataLoader(ds, BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                opt.step()

        acc = _eval_all(model, val_data, task_id, device)
        history.append(acc)
        print(f"  Task {task_id}: acc={acc:.4f}", flush=True)

    accs = _per_task_acc(model, val_data, device)
    return accs, history


# ──────────────────────────────────────────────────────────────────────────
# Method 2: Uniform + CE (class-balanced retrieval, cross-entropy loss)
# ──────────────────────────────────────────────────────────────────────────

def run_uniform_ce(device, train_data_class, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = ReplayBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)
    history = []

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)

        start = task_id * N_CLASSES_PER_TASK
        end = start + N_CLASSES_PER_TASK
        xs, ys = [], []
        for c in range(start, end):
            xs.extend(train_data_class[c])
            ys.extend([c] * len(train_data_class[c]))
        ds = TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in xs]),
            torch.tensor(ys, dtype=torch.long),
        )
        loader = DataLoader(ds, BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                if task_id > 0:
                    num_old = task_id * N_CLASSES_PER_TASK
                    replay = bank.query_uniform_by_class(RETRIEVAL_BUDGET, num_old)
                    rx, ry, _ = _augment(replay, transform, device)
                    cx = torch.cat([x, rx]) if rx is not None else x
                    cy = torch.cat([y, ry]) if ry is not None else y
                else:
                    cx, cy = x, y

                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                opt.step()

        acc = _eval_all(model, val_data, task_id, device)
        history.append(acc)
        print(f"  Task {task_id}: acc={acc:.4f}", flush=True)

    accs = _per_task_acc(model, val_data, device)
    return accs, history


# ──────────────────────────────────────────────────────────────────────────
# Method 3: Uniform + Distill (class-balanced retrieval, distillation loss)
# ──────────────────────────────────────────────────────────────────────────

def run_uniform_distill(device, train_data_class, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = ReplayBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)
    history = []

    for task_id in range(N_TASKS):
        if task_id > 0:
            # Snapshot BEFORE expansion
            model.eval()
            bank.snapshot_logits(model, transform, device)
            model.expand_head(N_CLASSES_PER_TASK)
            model.train()

        start = task_id * N_CLASSES_PER_TASK
        end = start + N_CLASSES_PER_TASK
        xs, ys = [], []
        for c in range(start, end):
            xs.extend(train_data_class[c])
            ys.extend([c] * len(train_data_class[c]))
        ds = TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in xs]),
            torch.tensor(ys, dtype=torch.long),
        )
        loader = DataLoader(ds, BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                if task_id > 0:
                    num_old = task_id * N_CLASSES_PER_TASK
                    replay = bank.query_uniform_by_class(RETRIEVAL_BUDGET, num_old)
                    rx, ry, rlogits = _augment(replay, transform, device)
                else:
                    rx, ry, rlogits = None, None, None

                if rx is not None and rlogits is not None:
                    logits_all = model(torch.cat([x, rx]))
                    ce_loss = F.cross_entropy(logits_all[:len(y)], y, reduction='sum')
                    current_old = logits_all[len(y):, :num_old]
                    distill = _distillation_loss(current_old, rlogits, TEMPERATURE)
                    # α=0 → uniform weights
                    weighted_distill = distill.sum()
                    total_items = len(y) + len(ry)
                    loss = (ce_loss + LAMBDA * weighted_distill) / total_items
                else:
                    loss = F.cross_entropy(model(x), y)

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                opt.step()

        acc = _eval_all(model, val_data, task_id, device)
        history.append(acc)
        print(f"  Task {task_id}: acc={acc:.4f}", flush=True)

    accs = _per_task_acc(model, val_data, device)
    return accs, history


# ── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    print("Loading CIFAR-100 data...", flush=True)
    class_images, mean, std = load_cifar100_data()
    print(f"Loaded {sum(len(c) for c in class_images)} images in {time.time()-t0:.1f}s", flush=True)

    # 80/20 train/val split: train_data_class[c] = list of training images
    train_data_class = [imgs[:int(len(imgs) * 0.8)] for imgs in class_images]
    val_data = prepare_val_data(class_images)
    print(f"Train: {sum(len(c) for c in train_data_class)} / Val: {sum(len(d) for d in val_data)}", flush=True)

    methods = [
        ("Baseline", run_baseline),
        ("Uniform + CE", run_uniform_ce),
        ("Uniform + Distill", run_uniform_distill),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"  SDWR-v2 full benchmark  (10 tasks, 100 classes)", flush=True)
    print(f"  epochs={EPOCHS_PER_TASK}, budget={RETRIEVAL_BUDGET}, capacity={CAPACITY_PER_CLASS}", flush=True)
    print(f"  lam={LAMBDA}, tau={TEMPERATURE}", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = {}
    for name, fn in methods:
        t1 = time.time()
        print(f"\n--- {name} ---", flush=True)
        if name == "Baseline":
            accs, history = fn(device, train_data_class, val_data)
        else:
            accs, history = fn(device, train_data_class, val_data, mean, std)
        elapsed = time.time() - t1
        all_results[name] = (accs, history)
        final_acc = np.mean(accs)
        print(f"  Final avg accuracy: {final_acc:.4f}  [{elapsed/60:.1f} min]", flush=True)

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Method':<35s} {'Final Avg':>10s} {'Time':>8s}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for name, (accs, history) in all_results.items():
        final_acc = np.mean(accs)
        elapsed_str = f"{(time.time() - t0)/60:.1f}m"  # approximate
        print(f"  {name:<35s} {final_acc:>10.4f} {elapsed_str:>8s}", flush=True)
    print(f"{'='*70}", flush=True)

    # Task-level accuracy curves
    print(f"\n  Task-level accuracy progression:", flush=True)
    print(f"  {'Task':>5s}", end="", flush=True)
    for name in all_results:
        print(f"  {name:<35s}", end="", flush=True)
    print(flush=True)
    for t in range(N_TASKS):
        print(f"  {t:>5d}", end="", flush=True)
        for name, (accs, history) in all_results.items():
            val = history[t] if t < len(history) else 0.0
            print(f"  {val:<35.4f}", end="", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
