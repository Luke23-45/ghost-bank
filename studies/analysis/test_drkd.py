"""
Verification: DRKD (Decoupled Replay with Knowledge Distillation)

Implements the architecture from docs/suggestion1.md:
  - Feature training: CE on new classes + KD on old classes (NO replay during SGD)
  - Post-hoc classifier calibration: train new linear classifier on all exemplars

Compares 3 methods on a 2-task (10-class) CIFAR-100 subset:
  1. Baseline         : no replay
  2. Uniform + CE     : class-balanced retrieval + CE (flawed baseline)
  3. DRKD             : KD during feature training + classifier calibration

Run from repo root:
  python studies/analysis/test_drkd.py [--lam 1.0]
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

# KD hyperparameters
LAMBDA = 1.0
TEMPERATURE = 2.0


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


def get_features(model: ResNet, x: torch.Tensor) -> torch.Tensor:
    """Extract 512-dim features before the FC classifier layer."""
    x = F.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
    x = model.dropout(x)
    return x


def _raw_to_tensor(raw):
    """NHWC uint8 raw image → NCHW float32 [0,1] tensor."""
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    if raw.dim() == 3 and raw.shape[-1] == 3:
        return raw.permute(2, 0, 1).contiguous().float() / 255.0
    return raw.float() / 255.0 if raw.dtype == torch.uint8 else raw


def _make_optim(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


# ── replay bank (for Uniform+CE method) ──────────────────────────────────
class ReplayBank:
    def __init__(self, capacity: int, seed: int):
        self._bank: dict[int, list] = {}
        self._cap = capacity
        self._rng = random.Random(seed)

    def store(self, examples: list):
        for x, y in examples:
            c = int(y)
            if c not in self._bank:
                self._bank[c] = []
            p = self._bank[c]
            if len(p) < self._cap:
                p.append((x, y))
            else:
                p[self._rng.randint(0, len(p) - 1)] = (x, y)

    def query_by_class(self, budget: int, num_classes: int) -> list:
        if num_classes == 0:
            return []
        base = budget // num_classes
        extra = budget - base * num_classes
        alloc = [base + (1 if i < extra else 0) for i in range(num_classes)]
        return sample_by_allocation(self._bank, alloc, self._rng)


# ── exemplar bank (for DRKD post-hoc calibration) ────────────────────────
class ExemplarBank:
    def __init__(self, capacity: int):
        self._bank: dict[int, list] = {}
        self._cap = capacity

    def store_raw(self, raw_img, label: int):
        c = int(label)
        if c not in self._bank:
            self._bank[c] = []
        p = self._bank[c]
        if len(p) < self._cap:
            p.append(raw_img)

    def items(self, class_id: int) -> list:
        return self._bank.get(class_id, [])


# ── evaluation ────────────────────────────────────────────────────────────
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

def run_baseline(device, train_data, val_data):
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
# 2. Uniform + CE (class-balanced retrieval + CE on all logits)
# ──────────────────────────────────────────────────────────────────────────

def run_uniform_ce(device, train_data, val_data, mean, std):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    opt = _make_optim(model)
    bank = ReplayBank(CAPACITY_PER_CLASS, SEED)
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
                    replay = bank.query_by_class(RETRIEVAL_BUDGET, N_CLASSES_PER_TASK)
                    rxs, rys = [], []
                    for item in replay:
                        r, l = item[0], item[1]
                        if r.dim() == 3 and r.shape[-1] == 3:
                            t = r.permute(2, 0, 1).contiguous().float() / 255.0
                        else:
                            t = r.float() / 255.0
                        t = transform(t)
                        rxs.append(t); rys.append(l)
                    rx_s = torch.stack(rxs).to(device)
                    ry_s = torch.tensor(rys, device=device, dtype=torch.long)
                    cx = torch.cat([x, rx_s]); cy = torch.cat([y, ry_s])
                else:
                    cx, cy = x, y
                loss = F.cross_entropy(model(cx), cy)
                opt.zero_grad(); loss.backward(); opt.step()

    return _per_class_acc(model, val_data, device)


# ──────────────────────────────────────────────────────────────────────────
# 3. DRKD — Decoupled Replay with Knowledge Distillation
# ──────────────────────────────────────────────────────────────────────────

def run_drkd(device, train_data, val_data, class_images, mean, std, *,
             lam: float = LAMBDA):
    """DRKD: feature training with KD + post-hoc classifier calibration."""
    model = create_model(N_CLASSES_PER_TASK).to(device)
    transform = make_train_transform(mean, std)
    exemplar_bank = ExemplarBank(CAPACITY_PER_CLASS)
    teacher_state = None

    for task_id in range(N_TASKS):
        if task_id == 0:
            # First task: CE only (no old classes, no KD)
            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(model(x), y)
                    opt.zero_grad(); loss.backward(); opt.step()
        else:
            num_old = task_id * N_CLASSES_PER_TASK

            # Expand head
            model.expand_head(N_CLASSES_PER_TASK)

            # Load frozen teacher
            teacher = create_model(num_old).to(device)
            teacher.load_state_dict(teacher_state)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

            # Feature training: CE on new classes + KD on old classes
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    logits_all = model(x)

                    # CE on new-class logits only
                    new_logits = logits_all[:, num_old:]
                    ce_loss = F.cross_entropy(new_logits, y - num_old, reduction='mean')

                    # KD on old-class logits (compare with frozen teacher)
                    with torch.no_grad():
                        teacher_logits = teacher(x)
                    student_old = logits_all[:, :num_old]
                    kd_loss = F.kl_div(
                        F.log_softmax(student_old / TEMPERATURE, dim=1),
                        F.softmax(teacher_logits / TEMPERATURE, dim=1),
                        reduction='batchmean',
                    ) * (TEMPERATURE ** 2)

                    loss = ce_loss + lam * kd_loss
                    opt.zero_grad(); loss.backward()

                    # Zero gradients to old-class head (don't update old rows)
                    if model.fc.weight.grad is not None:
                        model.fc.weight.grad[:num_old] = 0
                    if model.fc.bias is not None and model.fc.bias.grad is not None:
                        model.fc.bias.grad[:num_old] = 0

                    opt.step()

        # Store exemplars for this task's classes (raw NHWC uint8)
        start_c = task_id * N_CLASSES_PER_TASK
        end_c = start_c + N_CLASSES_PER_TASK
        for c in range(start_c, end_c):
            imgs = class_images[c]
            selected = random.Random(SEED).sample(imgs, min(CAPACITY_PER_CLASS, len(imgs)))
            for img in selected:
                exemplar_bank.store_raw(img, c)

        # Save teacher snapshot
        teacher_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Post-hoc classifier calibration (§2.3) ──
    print("    Calibrating classifier...", flush=True)
    model.eval()
    feat_dim = 512
    num_total = N_CLASSES_TOTAL

    cal_features, cal_labels = [], []
    for c in range(num_total):
        imgs = exemplar_bank.items(c)
        if not imgs:
            continue
        for raw in imgs:
            t = _raw_to_tensor(raw).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = get_features(model, t).cpu()
            cal_features.append(feat)
            cal_labels.append(c)

    cal_features = torch.cat(cal_features)
    cal_labels = torch.tensor(cal_labels)

    # Train new linear classifier from scratch
    new_fc = nn.Linear(feat_dim, num_total).to(device)
    cal_opt = torch.optim.SGD(new_fc.parameters(), lr=0.1)
    cal_ds = TensorDataset(cal_features, cal_labels)
    cal_loader = DataLoader(cal_ds, 64, shuffle=True)

    for _ in range(10):
        for cf, cl in cal_loader:
            cf, cl = cf.to(device), cl.to(device)
            cal_loss = F.cross_entropy(new_fc(cf), cl)
            cal_opt.zero_grad(); cal_loss.backward(); cal_opt.step()

    model.fc = new_fc
    model.train()

    return _per_class_acc(model, val_data, device)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lam", type=float, default=LAMBDA,
                   help="KD weight lambda (default: 1.0)")
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
        ("1. Baseline (no replay)", lambda: run_baseline(device, train_data, val_data)),
        ("2. Uniform + CE (class-balanced)", lambda: run_uniform_ce(device, train_data, val_data, mean, std)),
        ("3. DRKD (lam=%.1f)" % args.lam, lambda: run_drkd(device, train_data, val_data, class_images, mean, std, lam=args.lam)),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"  DRKD verification (2 tasks, {N_CLASSES_TOTAL} classes)", flush=True)
    print(f"  lam={args.lam}, tau={TEMPERATURE}", flush=True)
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

    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Method':<30s} {'Avg':>8s} {'Task-0':>8s} {'Prior':>8s}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for name, accs in all_accs.items():
        print(f"  {name:<30s} {np.mean(accs):>8.3f} {np.mean(accs[:N_CLASSES_PER_TASK]):>8.3f} {np.mean(accs[N_CLASSES_PER_TASK:]):>8.3f}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
