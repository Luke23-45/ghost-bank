"""
Verification: DHMDR (Disjoint-Head Masked Distillation Replay)

Corrects the SDWR-v2 bug: CE is computed ONLY on new-class logits,
distillation ONLY on old-class logits, so gradients do not interfere
at the classifier head level.

Compares 4 methods on a 2-task (10-class) CIFAR-100 subset:
  1. Baseline        : no replay
  2. Uniform + CE    : class-balanced retrieval, CE on all logits (flawed)
  3. DHMDR (alpha=0) : uniform retrieval, disjoint-head distillation
  4. DHMDR (alpha=1) : same + per-class importance weighting

Run from repo root:
  python studies/analysis/test_dhmdr.py
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

# Distillation hyperparameters
LAMBDA = 1.0
TEMPERATURE = 2.0
RHO_EMA_GAMMA = 0.9


# ── data ──────────────────────────────────────────────────────────────────
def load_cifar100_data() -> tuple[list, list, list]:
    cfg = CIFAR100Config(
        root="./data/cifar100", seed=SEED, num_workers=0, pin_memory=False,
        persistent_workers=False, batch_size=BATCH_SIZE, num_tasks=10,
        classes_per_task=10, mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )
    dm = CIFAR100DataModule(cfg)
    dm.setup("fit")
    train_images = dm._train_images
    train_targets = dm._train_targets
    class_images: list[list[torch.Tensor]] = [[] for _ in range(100)]
    for i in range(len(train_targets)):
        class_images[int(train_targets[i])].append(train_images[i])
    return class_images, dm.config.mean, dm.config.std


def prepare_task_data(class_images: list[list[torch.Tensor]],
                      val_split: float = 0.2) -> tuple[list, list]:
    train_data, val_data = [], []
    for task_id in range(N_TASKS):
        start = task_id * N_CLASSES_PER_TASK
        end = start + N_CLASSES_PER_TASK
        xs, ys, vxs, vys = [], [], [], []
        for c in range(start, end):
            imgs = class_images[c]
            n = len(imgs)
            split = int(n * (1 - val_split))
            xs.extend(imgs[:split]); ys.extend([c] * split)
            vxs.extend(imgs[split:]); vys.extend([c] * (n - split))
        train_data.append(TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in xs]),
            torch.tensor(ys, dtype=torch.long),
        ))
        val_data.append(TensorDataset(
            torch.stack([x.float().permute(2, 0, 1) / 255.0 for x in vxs]),
            torch.tensor(vys, dtype=torch.long),
        ))
    return train_data, val_data


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
                self._bank[c][self._rng.randint(0, len(pool) - 1)] = (x, y)

    def query_uniform_by_class(self, budget: int, num_classes: int) -> list:
        if num_classes == 0:
            return []
        base = budget // num_classes
        extra = budget - base * num_classes
        alloc = [base + (1 if i < extra else 0) for i in range(num_classes)]
        return sample_by_allocation(self._bank, alloc, self._rng)

    def snapshot_logits(self, model, transform, device):
        for c in list(self._bank.keys()):
            new_pool = []
            for item in self._bank[c]:
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
        logit_list = [lt.to(device) if lt is not None else torch.zeros(1, device=device) for lt in tlogits]
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


def _compute_drift(bank, model, transform, device, old_classes, temperature):
    deltas = []
    for c in old_classes:
        pool = bank._bank.get(c, [])
        if not pool:
            deltas.append(0.0)
            continue
        drift_sum = 0.0
        for item in pool:
            if len(item) != 3:
                continue
            raw, _, snapshot_logit = item
            if raw.dim() == 3 and raw.shape[-1] == 3:
                t = raw.permute(2, 0, 1).contiguous().float() / 255.0
            elif raw.dtype == torch.uint8:
                t = raw.float() / 255.0
            else:
                t = raw.float() if torch.is_tensor(raw) else torch.as_tensor(raw).float()
            x_aug = transform(t).unsqueeze(0).to(device)
            with torch.no_grad():
                current_logit = model(x_aug)
            num_snapshot = snapshot_logit.shape[0]
            current_old = current_logit[0, :num_snapshot].cpu()
            drift = (F.softmax(current_old / temperature, dim=0) - F.softmax(snapshot_logit / temperature, dim=0)).abs().sum().item()
            drift_sum += drift
        deltas.append(drift_sum / max(len(pool), 1))
    return deltas


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
    accs = [correct[c] / total[c] if total.get(c, 0) > 0 else 0.0 for c in sorted(correct)]
    model.train()
    return accs


# ──────────────────────────────────────────────────────────────────────────
# 1. Baseline (no replay)
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# 2. Uniform + CE (class-balanced, CE on all logits — the flawed baseline)
# ──────────────────────────────────────────────────────────────────────────

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
                    num_old = N_CLASSES_PER_TASK
                    replay = bank.query_uniform_by_class(RETRIEVAL_BUDGET, num_old)
                    rx, ry, _ = _augment(replay, transform, device)
                    cx = torch.cat([x, rx]) if rx is not None else x
                    cy = torch.cat([y, ry]) if ry is not None else y
                else:
                    cx, cy = x, y

                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


# ──────────────────────────────────────────────────────────────────────────
# 3 & 4. DHMDR — disjoint-head masked distillation replay
# ──────────────────────────────────────────────────────────────────────────

def run_dhmdr(device, train_data, val_data, mean, std, *, alpha: float):
    """DHMDR with optional per-class importance weighting.

    alpha=0 → DHMDR without importance weights
    alpha=1 → DHMDR with per-class weighting
    """
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = ReplayBank(0, CAPACITY_PER_CLASS, SEED)
    transform = make_train_transform(mean, std)

    rho: dict[int, float] = {}

    for task_id in range(N_TASKS):
        if task_id > 0:
            # Snapshot BEFORE expansion
            model.eval()
            bank.snapshot_logits(model, transform, device)

            # Expand head
            model.expand_head(N_CLASSES_PER_TASK)

            # Compute drift with expanded model (eval mode)
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
                    num_old = N_CLASSES_PER_TASK * task_id
                    replay = bank.query_uniform_by_class(RETRIEVAL_BUDGET, num_old)
                    rx, ry, rlogits = _augment(replay, transform, device)
                else:
                    rx, ry, rlogits = None, None, None

                if rx is not None and rlogits is not None:
                    # ── DHMDR: disjoint-head losses ──
                    logits_all = model(torch.cat([x, rx]))
                    batch_sz = len(y)

                    # 1) New-task CE on NEW-CLASS logits only (§3.3 step 3)
                    #    Labels are 5-9 for task 1; remap to 0-4.
                    new_logits = logits_all[:batch_sz, num_old:]
                    ce_loss = F.cross_entropy(new_logits, y - num_old, reduction='mean')

                    # 2) Distillation on OLD-CLASS logits only (§3.3 step 4)
                    old_logits = logits_all[batch_sz:, :num_old]
                    distill = _distillation_loss(old_logits, rlogits, TEMPERATURE)

                    # Optional per-class weighting
                    if alpha > 0:
                        weights = torch.ones(len(ry), device=device)
                        for j, c_val in enumerate(ry):
                            weights[j] = 1.0 + alpha * rho.get(c_val.item(), 0.0)
                        distill = distill * weights
                    replay_loss = distill.mean()

                    # 3) Combined loss: L_new + λ·L_replay (§3.3 step 5)
                    loss = ce_loss + LAMBDA * replay_loss
                else:
                    # Task 0: standard CE over all learned classes
                    loss = F.cross_entropy(model(x), y)

                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


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

    train_data, val_data = prepare_task_data(class_images)
    print(f"Train per task: {[len(d) for d in train_data]}", flush=True)

    methods = [
        ("1. Baseline (no replay)", lambda: run_baseline(device, train_data, val_data, mean, std)),
        ("2. Uniform + CE (class-balanced)", lambda: run_uniform_ce(device, train_data, val_data, mean, std)),
        ("3. DHMDR (alpha=0)", lambda: run_dhmdr(device, train_data, val_data, mean, std, alpha=0.0)),
        ("4. DHMDR (alpha=1.0)", lambda: run_dhmdr(device, train_data, val_data, mean, std, alpha=1.0)),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"  DHMDR verification (2 tasks, {N_CLASSES_TOTAL} classes)", flush=True)
    print(f"  lam={LAMBDA}, tau={TEMPERATURE}, budget={RETRIEVAL_BUDGET}", flush=True)
    print(f"{'='*70}", flush=True)

    all_accs = {}
    for name, fn in methods:
        t1 = time.time()
        accs = fn()
        elapsed = time.time() - t1
        all_accs[name] = accs
        avg = np.mean(accs)
        t0_avg = np.mean(accs[:N_CLASSES_PER_TASK])
        prior_avg = np.mean(accs[N_CLASSES_PER_TASK:])

        print(f"\n  {name}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for i, c in enumerate(range(N_CLASSES_TOTAL)):
            marker = " [prior]" if i >= N_CLASSES_PER_TASK else " [task-0]"
            print(f"    class {c:2d}: {accs[i]:.3f}{marker}", flush=True)
        print(f"    avg: {avg:.3f}  task-0: {t0_avg:.3f}  prior: {prior_avg:.3f}  [{elapsed:.0f}s]", flush=True)

    # Summary comparison
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Method':<35s} {'Avg':>8s} {'Task-0':>8s} {'Prior':>8s}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for name, accs in all_accs.items():
        avg = np.mean(accs)
        t0_avg = np.mean(accs[:N_CLASSES_PER_TASK])
        prior_avg = np.mean(accs[N_CLASSES_PER_TASK:])
        print(f"  {name:<35s} {avg:>8.3f} {t0_avg:>8.3f} {prior_avg:>8.3f}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
