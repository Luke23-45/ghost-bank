"""
Verification: SDWR (Stratified Debt-Weighted Replay)

Uses the SAME CIFAR-100 data pipeline as the main benchmark
(processed tensors from CIFAR100Ingestor).

Compares 5 methods on a 2-task (10-class) scenario:
  - Baseline      : no replay
  - StaticBank    : uniform retrieval, no weighting
  - PID-GB        : debt-concentrated retrieval (original)
  - SDWR-base     : stratified allocation (coverage-first), uniform loss
  - SDWR-full     : stratified allocation + debt-weighted loss (\alpha=1.0)

Run from repo root:
  python studies/analysis/test_sdwr.py
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

from src.bank.core.pid_controller import PIDController
from src.bank.strategies.static import StaticReplayBank
from src.bank.strategies.ed_gb import ExposureDebtGhostBank
from src.data.cifar100 import CIFAR100DataModule, CIFAR100Config
from src.data.cifar100.transforms import make_train_transform
from src.models import ResNet
from src.bank.core.allocator import allocate_by_debt
from src.bank.core.retrieval import sample_by_allocation

SEED = 13
N_TASKS = 2
N_CLASSES_PER_TASK = 5
N_CLASSES_TOTAL = N_TASKS * N_CLASSES_PER_TASK
EPOCHS_PER_TASK = 15
BATCH_SIZE = 32
RETRIEVAL_BUDGET = 16
CAPACITY_PER_CLASS = 50
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
ALPHA = 1.0                        # SDWR-full debt weight multiplier
BANK_PROBE_SIZE = 16
SDWR_EPSILON = 1e-6


# ── data: re-use the project CIFAR-100 pipeline ─────────────────────────
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


# ── model ────────────────────────────────────────────────────────────────
def create_model(num_classes: int) -> ResNet:
    return ResNet(num_classes=num_classes, base_filters=64)


# ── banks ────────────────────────────────────────────────────────────────
class UniformBank:
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


# ── SDWR allocation ─────────────────────────────────────────────────────
def allocate_sdwr(
    debt: list[float],
    budget: int,
    num_classes: int,
    epsilon: float = SDWR_EPSILON,
) -> list[int]:
    """Coverage-first allocation (Eq. 4.1 in suggestions.md).

    Each class receives ``budget // num_classes`` base items.
    The remainder is distributed by largest-remainder on
    ``max(0, debt) + epsilon`` weights.
    """
    b = budget // num_classes
    remaining = budget - b * num_classes
    if remaining <= 0:
        return [b] * num_classes

    w = [max(0.0, d) + epsilon for d in debt]
    total_w = sum(w)
    raw = [remaining * w_i / total_w for w_i in w]
    base = [int(v) for v in raw]

    leftover = remaining - sum(base)
    if leftover > 0:
        frac = [raw[i] - base[i] for i in range(num_classes)]
        order = sorted(range(num_classes), key=lambda i: (frac[i], w[i]), reverse=True)
        for i in order[:leftover]:
            base[i] += 1

    return [b + base[i] for i in range(num_classes)]


# ── zero-allocation tracker ─────────────────────────────────────────────
class ZeroAllocTracker:
    """Records fraction of prior classes receiving zero replay per step."""
    def __init__(self):
        self._rates: list[float] = []

    def record(self, allocation: list[int], num_prior: int):
        if num_prior > 0:
            zeros = sum(1 for a in allocation[:num_prior] if a == 0)
            self._rates.append(zeros / num_prior)

    @property
    def avg_rate(self) -> float:
        return float(np.mean(self._rates)) if self._rates else 0.0


# ── helpers ──────────────────────────────────────────────────────────────
def _augment(items: list, transform, device):
    if not items:
        return None, None
    xs, ys = [], []
    for raw, label in items:
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
    return torch.stack(xs).to(device), torch.tensor(ys, device=device, dtype=torch.long)


def _probe_loss(model, pool, n_probe, device, transform):
    rng = random.Random(SEED)
    items = rng.sample(pool, min(n_probe, len(pool)))
    xs, ys = _augment(items, transform, device)
    if xs is None:
        return None
    with torch.no_grad():
        logits = model(xs)
    return F.cross_entropy(logits, ys, reduction="mean").item()


def _make_optim(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


def _zero_alloc_from_items(replay_items: list, num_prior: int) -> list[int]:
    """Convert a list of (raw, label) replay items to a per-class allocation vector."""
    counts = [0] * num_prior
    for _, label in replay_items:
        c = int(label)
        if c < num_prior:
            counts[c] += 1
    return counts


# ────────────────────────────────────────────────────────────────────────
# 1. Baseline — no replay
# ────────────────────────────────────────────────────────────────────────

def run_baseline(device, train_data, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device), None


# ────────────────────────────────────────────────────────────────────────
# 2. StaticBank — uniform retrieval, no weighting
# ────────────────────────────────────────────────────────────────────────

def run_static(device, train_data, val_data, mean, std):
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
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))
                replay = bank.query(RETRIEVAL_BUDGET)
                rx, ry = _augment(replay, transform, device)
                cx = torch.cat([x, rx]) if rx is not None else x
                cy = torch.cat([y, ry]) if ry is not None else y
                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device), None


# ────────────────────────────────────────────────────────────────────────
# 3. PID-GB — debt-concentrated retrieval (original)
# ────────────────────────────────────────────────────────────────────────

def run_pid_gb(device, train_data, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = DebtBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)
    pid = PIDController(N_CLASSES_TOTAL, K_p=1.0, K_i=0.1, K_d=0.5,
                        decay=0.99, smooth=0.9, temperature=1.0)
    za_tracker = ZeroAllocTracker()

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)
            bank.expand(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

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

                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                replay = bank.query(RETRIEVAL_BUDGET, debt=debt, temperature=1.0)
                rx, ry = _augment(replay, transform, device)
                cx = torch.cat([x, rx]) if rx is not None else x
                cy = torch.cat([y, ry]) if ry is not None else y
                loss = F.cross_entropy(model(cx), cy)

                za_tracker.record(bank._last_allocation, N_CLASSES_PER_TASK * task_id)

                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device), za_tracker


# ────────────────────────────────────────────────────────────────────────
# 4. SDWR-base — stratified allocation, uniform loss
# ────────────────────────────────────────────────────────────────────────

def run_sdwr_base(device, train_data, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = UniformBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)
    pid = PIDController(N_CLASSES_TOTAL, K_p=1.0, K_i=0.1, K_d=0.5,
                        decay=0.99, smooth=0.9, temperature=1.0)
    za_tracker = ZeroAllocTracker()

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)
            bank.expand(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

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

                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                # ── SDWR stratified allocation ──
                n_classes = pid.num_classes
                debt_for_alloc = [d if c in bank.bank else 0.0 for c, d in enumerate(debt)]
                allocation = allocate_sdwr(debt_for_alloc, RETRIEVAL_BUDGET, n_classes)
                replay = sample_by_allocation(bank.bank, allocation, random.Random(SEED))
                rx, ry = _augment(replay, transform, device)

                za_tracker.record(allocation, N_CLASSES_PER_TASK * task_id)

                # ── uniform loss (no debt weighting) ──
                cx = torch.cat([x, rx]) if rx is not None else x
                cy = torch.cat([y, ry]) if ry is not None else y
                loss = F.cross_entropy(model(cx), cy)

                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device), za_tracker


# ────────────────────────────────────────────────────────────────────────
# 5. SDWR-full — stratified allocation + debt-weighted loss
# ────────────────────────────────────────────────────────────────────────

def run_sdwr_full(device, train_data, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = UniformBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)
    pid = PIDController(N_CLASSES_TOTAL, K_p=1.0, K_i=0.1, K_d=0.5,
                        decay=0.99, smooth=0.9, temperature=1.0)
    za_tracker = ZeroAllocTracker()

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)
            bank.expand(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

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

                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                # ── SDWR stratified allocation ──
                n_classes = pid.num_classes
                debt_for_alloc = [d if c in bank.bank else 0.0 for c, d in enumerate(debt)]
                allocation = allocate_sdwr(debt_for_alloc, RETRIEVAL_BUDGET, n_classes)
                replay = sample_by_allocation(bank.bank, allocation, random.Random(SEED))
                rx, ry = _augment(replay, transform, device)

                za_tracker.record(allocation, N_CLASSES_PER_TASK * task_id)

                # ── debt-weighted loss ──
                if rx is not None:
                    cx = torch.cat([x, rx])
                    cy = torch.cat([y, ry])
                    logits_all = model(cx)

                    ce = F.cross_entropy(logits_all, cy, reduction="none")
                    w = torch.ones(len(cy), device=device)
                    for j, c_val in enumerate(ry):
                        w[len(y) + j] = 1.0 + ALPHA * max(0.0, debt[c_val.item()])
                    loss = (ce * w).sum() / w.sum()
                else:
                    loss = F.cross_entropy(model(x), y)

                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device), za_tracker


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
    global ALPHA
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="SDWR-full debt weight multiplier")
    args = p.parse_args()

    ALPHA = args.alpha

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    print("Loading CIFAR-100 data...", flush=True)
    class_images, class_targets, mean, std = load_cifar100_data()
    print(f"Loaded {sum(len(c) for c in class_images)} total images in {time.time()-t0:.1f}s", flush=True)

    class_order = list(range(N_CLASSES_TOTAL))
    train_data, val_data = prepare_task_data(class_images[:N_CLASSES_TOTAL], class_order)
    print(f"Train samples per task: {[len(d) for d in train_data]}", flush=True)

    methods = [
        ("Baseline (no replay)", run_baseline),
        ("StaticBank (uniform retrieval)", run_static),
        ("PID-GB (debt-concentrated retrieval)", run_pid_gb),
        ("SDWR-base (stratified allocation, uniform loss)", run_sdwr_base),
        ("SDWR-full (stratified + debt-weighted loss, alpha=%.1f)" % ALPHA, run_sdwr_full),
    ]

    print(f"\n{'='*80}", flush=True)
    print(f"  Method comparison (2 tasks, {N_CLASSES_TOTAL} classes)", flush=True)
    print(f"  SDWR alpha={ALPHA}, budget={RETRIEVAL_BUDGET}, capacity={CAPACITY_PER_CLASS}", flush=True)
    print(f"{'='*80}", flush=True)

    all_accs = {}
    for name, fn in methods:
        t1 = time.time()
        result = fn(device, train_data, val_data, mean, std)
        elapsed = time.time() - t1
        if isinstance(result, tuple) and len(result) == 2:
            accs, za_tracker = result
        else:
            accs, za_tracker = result, None
        all_accs[name] = accs
        avg = np.mean(accs)
        prior_avg = np.mean([a for i, a in enumerate(accs) if i >= N_CLASSES_PER_TASK])
        za_str = ""
        if za_tracker is not None:
            za_str = f"  zero-alloc: {za_tracker.avg_rate*100:.1f}% of prior classes/step"

        print(f"\n  {name}", flush=True)
        print(f"  {'-'*55}", flush=True)
        for i, c in enumerate(range(N_CLASSES_TOTAL)):
            marker = " [prior]" if i >= N_CLASSES_PER_TASK else " [task-0]"
            print(f"    class {c:2d}: {accs[i]:.3f}{marker}", flush=True)
        print(f"    average: {avg:.3f}  (prior avg: {prior_avg:.3f})  [{elapsed:.0f}s]{za_str}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("  Note: with R >= N in this 2-task setup (N=10, R=16),", flush=True)
    print("  all methods satisfy the coverage floor. The R < N regime", flush=True)
    print("  (where budget concentration causes starvation) requires", flush=True)
    print("  a full 10-task CIFAR-100 benchmark to evaluate.", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
