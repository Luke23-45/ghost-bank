"""
Verification: SDWR-v2 (Slow Distillation-Weighted Replay)

Implements the professor's architecture from docs/suggestion1.md.

Compares 5 methods on a 2-task (10-class) CIFAR-100 subset:
  1. Baseline         : no replay
  2. Uniform + CE     : uniform retrieval, cross-entropy loss (StaticBank)
  3. Uniform + Distill: uniform retrieval, distillation loss, alpha=0
  4. SDWR-v2 alpha=1.0: uniform retrieval, weighted distillation
  5. SDWR-v2 alpha=2.0: uniform retrieval, weighted distillation

Run from repo root:
  python studies/analysis/test_sdwr_v2.py
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

from src.bank.strategies.static import StaticReplayBank
from src.data.cifar100 import CIFAR100DataModule, CIFAR100Config
from src.data.cifar100.transforms import make_train_transform
from src.models import ResNet
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

# Distillation hyperparameters (from §4 of suggestion1.md)
LAMBDA = 1.0
TEMPERATURE = 2.0
RHO_EMA_GAMMA = 0.9


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
    for i in range(len(train_targets)):
        c = int(train_targets[i])
        class_images[c].append(train_images[i])

    return class_images, dm.config.mean, dm.config.std


def prepare_task_data(
    class_images: list[list[torch.Tensor]],
    val_split: float = 0.2,
) -> tuple[list, list]:
    train_data, val_data = [], []
    for task_id in range(N_TASKS):
        start = task_id * N_CLASSES_PER_TASK
        end = start + N_CLASSES_PER_TASK
        xs, ys, vxs, vys = [], [], [], []
        for c in range(start, end):
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


# ── uniform replay bank (class-balanced) ─────────────────────────────────
class ReplayBank:
    """Stores exemplars and supports both uniform and class-balanced retrieval.

    Items stored as (raw_image, label) pairs.
    After snapshotting, items become (raw_image, label, snapshot_logit) triples.
    """
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

    def expand(self, num_new: int):
        pass

    def query_uniform(self, budget: int) -> list:
        """Uniform retrieval weighted by pool size (like StaticBank)."""
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
        """Class-balanced uniform retrieval — each class gets equal items."""
        if num_classes == 0:
            return []
        base = budget // num_classes
        extra = budget - base * num_classes
        # Build allocation: each class gets base; first `extra` classes get +1
        alloc = [base + (1 if i < extra else 0) for i in range(num_classes)]
        return sample_by_allocation(self._bank, alloc, self._rng)

    def snapshot_logits(self, model, transform, device):
        """Replace (raw, label) with (raw, label, logit) for every exemplar."""
        for c in list(self._bank.keys()):
            pool = self._bank[c]
            new_pool = []
            for item in pool:
                raw, label = item[0], item[1]
                # NHWC uint8 → NCHW float [0,1] (same conversion as _augment)
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


# ── helpers ──────────────────────────────────────────────────────────────
def _augment(items: list, transform, device):
    """Convert stored items to batched tensors.

    Items may be (raw, label) pairs or (raw, label, logit) triples.
    Returns (xs, ys, tlogits) where tlogits is None if no logits available.
    """
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

    # Check if any logits are available
    have_logits = any(lt is not None for lt in tlogits)
    if have_logits:
        # Use stored logits where available, zeros elsewhere
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


def _compute_drift(
    bank: ReplayBank,
    model,
    transform,
    device,
    old_classes: list[int],
    temperature: float,
) -> list[float]:
    """Compute drift delta_c for each old class (§3.2).

    Returns list of delta_c values indexed by position in old_classes.
    """
    deltas = []
    for c in old_classes:
        pool = bank._bank.get(c, [])
        if not pool:
            deltas.append(0.0)
            continue
        drift_sum = 0.0
        for item in pool:
            if len(item) == 3:
                raw, _, snapshot_logit = item
            else:
                raw, _ = item
                continue  # no snapshot to compare against

            # NHWC uint8 → NCHW float [0,1] (same conversion as _augment)
            if raw.dim() == 3 and raw.shape[-1] == 3:
                t = raw.permute(2, 0, 1).contiguous().float() / 255.0
            elif raw.dtype == torch.uint8:
                t = raw.float() / 255.0
            else:
                t = raw.float() if torch.is_tensor(raw) else torch.as_tensor(raw).float()
            x_aug = transform(t).unsqueeze(0).to(device)
            with torch.no_grad():
                current_logit = model(x_aug)
            # Only compare the old-class portion matching snapshot dimensionality
            num_snapshot = snapshot_logit.shape[0]
            current_old = current_logit[0, :num_snapshot].cpu()
            # L1 distance on softmax probabilities
            drift = (
                F.softmax(current_old / temperature, dim=0)
                - F.softmax(snapshot_logit / temperature, dim=0)
            ).abs().sum().item()
            drift_sum += drift
        deltas.append(drift_sum / max(len(pool), 1))
    return deltas


def _distillation_loss(current_logits: torch.Tensor,
                       target_logits: torch.Tensor,
                       temperature: float) -> torch.Tensor:
    """KL divergence with temperature scaling (§3.4)."""
    return F.kl_div(
        F.log_softmax(current_logits / temperature, dim=1),
        F.softmax(target_logits / temperature, dim=1),
        reduction='none',
    ).sum(dim=1) * (temperature ** 2)


def _make_optim(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


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
        for _ in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


# ────────────────────────────────────────────────────────────────────────
# 2. Uniform + CE — class-balanced uniform retrieval, CE loss
# ────────────────────────────────────────────────────────────────────────

def run_uniform_ce(device, train_data, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = ReplayBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)

    for task_id in range(N_TASKS):
        if task_id > 0:
            model.expand_head(N_CLASSES_PER_TASK)

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for _ in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                if task_id > 0:
                    replay = bank.query_uniform_by_class(RETRIEVAL_BUDGET, N_CLASSES_PER_TASK)
                    rx, ry, _ = _augment(replay, transform, device)
                    cx = torch.cat([x, rx]) if rx is not None else x
                    cy = torch.cat([y, ry]) if ry is not None else y
                else:
                    cx, cy = x, y

                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


# ────────────────────────────────────────────────────────────────────────
# 3. Uniform + Distill — uniform retrieval + distillation (alpha=0)
# 4 & 5. SDWR-v2 — uniform retrieval + weighted distillation
# ────────────────────────────────────────────────────────────────────────

def run_distill_method(device, train_data, val_data, mean, std, *,
                       alpha: float):
    """Run a distillation-based method.

    alpha=0  → Uniform + Distill (method 3)
    alpha=1  → SDWR-v2 (method 4)
    alpha=2  → SDWR-v2 (method 5)
    """
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = ReplayBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)

    # Per-task importance factors (§3.2)
    rho: dict[int, float] = {}  # class_id -> importance factor

    for task_id in range(N_TASKS):
        if task_id > 0:
            # ── snapshot BEFORE expansion (§3.1) ──
            model.eval()
            bank.snapshot_logits(model, transform, device)

            # ── expand head ──
            model.expand_head(N_CLASSES_PER_TASK)

            # ── compute drift with EXPANDED model (§3.2) ──
            old_classes = list(range(task_id * N_CLASSES_PER_TASK))
            deltas = _compute_drift(bank, model, transform, device,
                                    old_classes, TEMPERATURE)
            for i, c in enumerate(old_classes):
                prev = rho.get(c, 0.0)
                rho[c] = RHO_EMA_GAMMA * prev + (1.0 - RHO_EMA_GAMMA) * deltas[i]

            model.train()

        loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
        for _ in range(EPOCHS_PER_TASK):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                raw = (x * 255).byte().permute(0, 2, 3, 1).contiguous()
                bank.store(list(zip(raw, y.tolist())))

                if task_id > 0:
                    num_old = N_CLASSES_PER_TASK
                    replay = bank.query_uniform_by_class(RETRIEVAL_BUDGET, num_old)
                    rx, ry, rlogits = _augment(replay, transform, device)
                else:
                    rx, ry, rlogits = None, None, None

                if rx is not None and rlogits is not None:
                    # Combined loss: CE on current batch + distillation on replay
                    logits_all = model(torch.cat([x, rx]))
                    # CE on current batch only
                    ce_loss = F.cross_entropy(logits_all[:len(y)], y, reduction='sum')
                    # Distillation on replay items
                    current_old = logits_all[len(y):, :num_old]
                    distill = _distillation_loss(current_old, rlogits, TEMPERATURE)
                    # Per-class weighting
                    weights = torch.ones(len(ry), device=device)
                    for j, c_val in enumerate(ry):
                        weights[j] = 1.0 + alpha * rho.get(c_val.item(), 0.0)
                    weighted_distill = (distill * weights).sum()
                    # Normalize by total number of items (|B_t| + |R_t|)
                    total_items = len(y) + len(ry)
                    loss = (ce_loss + LAMBDA * weighted_distill) / total_items
                else:
                    loss = F.cross_entropy(model(x), y)

                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


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
    class_images, mean, std = load_cifar100_data()
    print(f"Loaded {sum(len(c) for c in class_images)} total images in {time.time()-t0:.1f}s", flush=True)

    train_data, val_data = prepare_task_data(class_images)
    print(f"Train samples per task: {[len(d) for d in train_data]}", flush=True)

    methods = [
        ("1. Baseline (no replay)", lambda: run_baseline(device, train_data, val_data, mean, std)),
        ("2. Uniform + CE (class-balanced)", lambda: run_uniform_ce(device, train_data, val_data, mean, std)),
        ("3. Uniform + Distill (alpha=0)", lambda: run_distill_method(device, train_data, val_data, mean, std, alpha=0.0)),
        ("4. SDWR-v2 (alpha=1.0)", lambda: run_distill_method(device, train_data, val_data, mean, std, alpha=1.0)),
        ("5. SDWR-v2 (alpha=2.0)", lambda: run_distill_method(device, train_data, val_data, mean, std, alpha=2.0)),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"  SDWR-v2 verification  (2 tasks, {N_CLASSES_TOTAL} classes)", flush=True)
    print(f"  lam={LAMBDA}, tau={TEMPERATURE}, rho_gamma={RHO_EMA_GAMMA}", flush=True)
    print(f"  budget={RETRIEVAL_BUDGET}, capacity={CAPACITY_PER_CLASS}", flush=True)
    print(f"{'='*70}", flush=True)

    all_accs = {}
    for name, fn in methods:
        t1 = time.time()
        accs = fn()
        elapsed = time.time() - t1
        all_accs[name] = accs
        avg = np.mean(accs)
        task0_avg = np.mean(accs[:N_CLASSES_PER_TASK])
        prior_avg = np.mean(accs[N_CLASSES_PER_TASK:])

        print(f"\n  {name}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for i, c in enumerate(range(N_CLASSES_TOTAL)):
            marker = " [prior]" if i >= N_CLASSES_PER_TASK else " [task-0]"
            print(f"    class {c:2d}: {accs[i]:.3f}{marker}", flush=True)
        print(f"    avg: {avg:.3f}  task-0: {task0_avg:.3f}  prior: {prior_avg:.3f}  [{elapsed:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
