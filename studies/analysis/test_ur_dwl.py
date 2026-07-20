"""
Verification: UR-DWL (Uniform Retrieval + Debt-Weighted Loss)

Uses the SAME CIFAR-100 data pipeline as the main benchmark
(processed tensors from CIFAR100Ingestor).

Compares 3 methods on a 2-task (10-class) scenario:
  - StaticBank  : uniform retrieval, no weighting
  - PID-GB      : debt-concentrated retrieval
  - UR-DWL      : uniform retrieval + debt-weighted loss (proposed)

Run from repo root:
  python studies/analysis/test_ur_dwl.py
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bank.core.pid_controller import PIDController
from src.bank.strategies.static import StaticReplayBank
from src.bank.strategies.ed_gb import ExposureDebtGhostBank
from src.data.cifar100 import CIFAR100DataModule, CIFAR100Config
from src.data.cifar100.transforms import make_eval_transform, make_train_transform, make_train_transform_from_rng
from src.models import ResNet
from src.bank.core.allocator import allocate_by_debt
from src.bank.core.retrieval import sample_by_allocation

SEED = 13
N_TASKS = 2                          # 2 tasks for fast verification
N_CLASSES_PER_TASK = 5               # 5 classes each → 10 total
N_CLASSES_TOTAL = N_TASKS * N_CLASSES_PER_TASK
EPOCHS_PER_TASK = 15
BATCH_SIZE = 32
RETRIEVAL_BUDGET = 16
CAPACITY_PER_CLASS = 50
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
ALPHA = 2.0                          # UR-DWL debt weight multiplier
BANK_PROBE_SIZE = 16
USE_SCHEDULER = False                # keep it simple: flat LR


# ── data: re-use the project CIFAR-100 pipeline ─────────────────────────
def load_cifar100_data() -> tuple[list, list, list]:
    """Load CIFAR-100 raw tensors using the project's ingestor, then
    return (class_images, class_targets) each a list-of-lists keyed by
    global class ID (0-99)."""
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

    # Raw NHWC uint8 tensors: [N, 32, 32, 3]
    train_images = dm._train_images   # [50000, 32, 32, 3] uint8
    train_targets = dm._train_targets # [50000] int64

    # Split into per-class lists
    class_images: list[list[torch.Tensor]] = [[] for _ in range(100)]
    class_targets: list[list[int]] = [[] for _ in range(100)]
    for i in range(len(train_targets)):
        c = int(train_targets[i])
        class_images[c].append(train_images[i])
        class_targets[c].append(c)

    return class_images, class_targets, dm.config.mean, dm.config.std


def prepare_task_data(
    class_images: list[list[torch.Tensor]],
    class_order: list[int],
    val_split: float = 0.2,
) -> tuple[list, list]:
    train_data, val_data = [], []
    for task_id in range(N_TASKS):
        classes = class_order[task_id * N_CLASSES_PER_TASK: (task_id + 1) * N_CLASSES_PER_TASK]
        xs, ys, vxs, vys = [], [], [], []
        for c in classes:
            imgs = class_images[c]
            n = len(imgs)
            split = int(n * (1 - val_split))
            xs.extend(imgs[:split])
            ys.extend([c] * split)
            vxs.extend(imgs[split:])
            vys.extend([c] * (n - split))
        train_data.append(TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in xs]),
            torch.tensor(ys, dtype=torch.long),
        ))
        val_data.append(TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in vxs]),
            torch.tensor(vys, dtype=torch.long),
        ))
    return train_data, val_data


# ── model (same as main benchmark) ──────────────────────────────────────
def create_model(num_classes: int) -> ResNet:
    return ResNet(num_classes=num_classes, base_filters=64)


# ── banks ───────────────────────────────────────────────────────────────
class UniformBank:
    """Wrapper that matches the main benchmark's StaticReplayBank query."""
    def __init__(self, num_classes: int, capacity: int, seed: int):
        self._bank = StaticReplayBank(num_classes or N_CLASSES_TOTAL, capacity, seed)
        self._rng = random.Random(seed)

    def store(self, examples: list):
        self._bank.store(examples)

    def expand(self, num_new: int):
        self._bank.expand(num_new)

    def query(self, budget: int, **kwargs) -> list:
        return self._bank.query(budget)

    @property
    def bank(self):
        return self._bank._bank


class DebtBank:
    """Wrapper matching the main benchmark's ExposureDebtGhostBank."""
    def __init__(self, num_classes: int, capacity: int, seed: int):
        self._bank = ExposureDebtGhostBank(num_classes or N_CLASSES_TOTAL, capacity, seed)
        self._rng = random.Random(seed)
        self._last_allocation = []

    def store(self, examples: list):
        self._bank.store(examples)

    def expand(self, num_new: int):
        self._bank.expand(num_new)

    def query(self, budget: int, *, debt: list[float] | None = None,
              temperature: float = 1.0, **kwargs) -> list:
        if debt is not None:
            debt_list = [d if c in self._bank._bank else 0.0 for c, d in enumerate(debt)]
            allocation = allocate_by_debt(debt_list, budget, temperature=temperature)
        else:
            allocation = [0] * len(debt) if debt else [0] * len(self._bank._bank)
        self._last_allocation = list(allocation)
        return sample_by_allocation(self._bank._bank, allocation, self._rng)

    @property
    def bank(self):
        return self._bank._bank


# ── helpers ─────────────────────────────────────────────────────────────
def _augment(items: list, transform, device):
    """Match main benchmark's _augment_replay: NHWC uint8 → NCHW → transform."""
    if not items:
        return None, None
    xs, ys = [], []
    for raw, label in items:
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        # NHWC uint8 [H, W, 3] → NCHW float32 [3, H, W]
        if raw.dim() == 3 and raw.shape[-1] == 3:
            t = raw.permute(2, 0, 1).contiguous().float() / 255.0
        else:
            t = raw.float() / 255.0 if raw.dtype == torch.uint8 else raw
        if transform is not None:
            t = transform(t)
        xs.append(t)
        ys.append(int(label))
    return torch.stack(xs).to(device), torch.tensor(ys, device=device, dtype=torch.long)


def _probe_loss(model, pool, n_probe, device, transform):
    """Probe loss for absent classes (matches PID-GB's bank probe)."""
    rng = random.Random(SEED)
    items = rng.sample(pool, min(n_probe, len(pool)))
    xs, ys = _augment(items, transform, device)
    if xs is None:
        return None
    with torch.no_grad():
        logits = model(xs)
    return F.cross_entropy(logits, ys, reduction="mean").item()


def _eval(model, val_data, up_to_task, device):
    all_x, all_y = [], []
    for t in range(up_to_task + 1):
        for x, y in DataLoader(val_data[t], 128, shuffle=False):
            all_x.append(x); all_y.append(y)
    if not all_x:
        return 0.0
    ds = TensorDataset(torch.cat(all_x), torch.cat(all_y))
    loader = DataLoader(ds, 128, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    model.train()
    return correct / total if total > 0 else 0.0


# ── method implementations ──────────────────────────────────────────────

def _make_optim(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# ────────────────────────────────────────────────────────────────────────
# 1. StaticBank — uniform retrieval, no weighting
# ────────────────────────────────────────────────────────────────────────

def run_static(device, train_data, val_data, class_order, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = UniformBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)
            bank.expand(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # Store raw NHWC uint8
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))
                # Retrieve uniform
                replay = bank.query(RETRIEVAL_BUDGET)
                rx, ry = _augment(replay, transform, device)
                cx = torch.cat([x, rx]) if rx is not None else x
                cy = torch.cat([y, ry]) if ry is not None else y
                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward(); opt.step()

    # Per-class accuracy after final task
    return _per_class_acc(model, val_data, device)


# ────────────────────────────────────────────────────────────────────────
# 2. PID-GB — debt-concentrated retrieval
# ────────────────────────────────────────────────────────────────────────

def run_pid_gb(device, train_data, val_data, class_order, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = DebtBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)
    pid = PIDController(N_CLASSES_TOTAL, K_p=1.0, K_i=0.1, K_d=0.5,
                        decay=0.99, smooth=0.9, temperature=1.0)

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)
            bank.expand(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                # PID signal
                with torch.no_grad():
                    logits = model(x)
                pcls = []
                for c in range(pid.num_classes):
                    mask = y == c
                    if mask.sum() > 0:
                        pcls.append(F.cross_entropy(logits[mask], y[mask], reduction="mean").item())
                    elif c in bank.bank and len(bank.bank[c]) > 0:
                        pcls.append(_probe_loss(model, bank.bank[c], BANK_PROBE_SIZE, device, transform))
                    else:
                        pcls.append(None)
                debt = pid.update(pcls)

                # Store
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                # Query with debt
                replay = bank.query(RETRIEVAL_BUDGET, debt=debt, temperature=1.0)
                rx, ry = _augment(replay, transform, device)
                cx = torch.cat([x, rx]) if rx is not None else x
                cy = torch.cat([y, ry]) if ry is not None else y
                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


# ────────────────────────────────────────────────────────────────────────
# 3. UR-DWL — Uniform Retrieval + Debt-Weighted Loss (proposed)
# ────────────────────────────────────────────────────────────────────────

def run_ur_dwl(device, train_data, val_data, class_order, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = UniformBank(0, CAPACITY_PER_CLASS, SEED)   # ← uniform retrieval!
    transform = make_train_transform(mean, std)
    pid = PIDController(N_CLASSES_TOTAL, K_p=1.0, K_i=0.1, K_d=0.5,
                        decay=0.99, smooth=0.9, temperature=1.0)

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)
            bank.expand(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                # PID signal (same as PID-GB)
                with torch.no_grad():
                    logits = model(x)
                pcls = []
                for c in range(pid.num_classes):
                    mask = y == c
                    if mask.sum() > 0:
                        pcls.append(F.cross_entropy(logits[mask], y[mask], reduction="mean").item())
                    elif c in bank.bank and len(bank.bank[c]) > 0:
                        pcls.append(_probe_loss(model, bank.bank[c], BANK_PROBE_SIZE, device, transform))
                    else:
                        pcls.append(None)
                debt = pid.update(pcls)

                # Store
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                # ── UNIFORM retrieval (same as StaticBank) ──
                replay = bank.query(RETRIEVAL_BUDGET)
                rx, ry = _augment(replay, transform, device)

                if rx is not None:
                    cx = torch.cat([x, rx])
                    cy = torch.cat([y, ry])
                    logits_all = model(cx)

                    # Debt-weighted cross-entropy
                    ce = F.cross_entropy(logits_all, cy, reduction="none")
                    w = torch.ones(len(cy), device=device)
                    for j, c_val in enumerate(ry):
                        w[len(y) + j] = 1.0 + ALPHA * max(0.0, debt[c_val.item()])
                    loss = (ce * w).sum() / w.sum()
                else:
                    loss = F.cross_entropy(model(x), y)

                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


# ── per-class accuracy ──────────────────────────────────────────────────

def _per_class_acc(model, val_data, device):
    model.eval()
    correct, total = {}, {}
    for t in range(N_TASKS):
        for x, y in DataLoader(val_data[t], 128, shuffle=False):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for i in range(len(y)):
                c = int(y[i])
                correct[c] = correct.get(c, 0) + (preds[i] == y[i]).item()
                total[c] = total.get(c, 0) + 1
    accs = []
    for c in sorted(correct):
        accs.append(correct[c] / total[c] if total[c] > 0 else 0.0)
    model.train()
    return accs


# ── main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    print("Loading CIFAR-100 data...", flush=True)
    class_images, class_targets, mean, std = load_cifar100_data()
    print(f"Loaded {sum(len(c) for c in class_images)} total images in {time.time()-t0:.1f}s", flush=True)

    # Use only the first N_CLASSES_TOTAL classes for our 2-task experiment
    class_order = list(range(N_CLASSES_TOTAL))
    train_data, val_data = prepare_task_data(class_images[:N_CLASSES_TOTAL], class_order)
    print(f"Train samples per task: {[len(d) for d in train_data]}", flush=True)

    methods = [
        ("StaticBank (uniform retrieval)", run_static),
        ("PID-GB (debt-concentrated retrieval)", run_pid_gb),
        ("UR-DWL (uniform + debt-weighted loss)", run_ur_dwl),
    ]

    print(f"\n{'='*65}", flush=True)
    print(f"  Method comparison (2 tasks, {N_CLASSES_TOTAL} classes)", flush=True)
    print(f"{'='*65}", flush=True)

    all_accs = {}
    for name, fn in methods:
        t1 = time.time()
        accs = fn(device, train_data, val_data, class_order, mean, std)
        elapsed = time.time() - t1
        all_accs[name] = accs
        avg = np.mean(accs)
        prior_avg = np.mean([a for i, a in enumerate(accs) if i >= N_CLASSES_PER_TASK])

        print(f"\n  {name}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for i, c in enumerate(range(N_CLASSES_TOTAL)):
            marker = " [prior]" if i >= N_CLASSES_PER_TASK else " [task-0]"
            print(f"    class {c:2d}: {accs[i]:.3f}{marker}", flush=True)
        print(f"    average: {avg:.3f}  (prior avg: {prior_avg:.3f})  [{elapsed:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
