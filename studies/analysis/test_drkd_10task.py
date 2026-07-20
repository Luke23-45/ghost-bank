"""
DRKD: 10-task CIFAR-100 benchmark (matches original benchmark config).

Compares against published StaticBank (13.1%) and PID-GB (10.9%) results.

Run from repo root:
  python studies/analysis/test_drkd_10task.py [--lam 1.0] [--epochs 70]
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

SEED = 13
N_TASKS = 10
N_CLASSES_PER_TASK = 10
N_CLASSES_TOTAL = N_TASKS * N_CLASSES_PER_TASK          # 100
EPOCHS_PER_TASK = 70
BATCH_SIZE = 128
RETRIEVAL_BUDGET = 64
CAPACITY_PER_CLASS = 200
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

LAMBDA = 1.0
TEMPERATURE = 2.0


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


def create_model(num_classes: int) -> ResNet:
    return ResNet(num_classes=num_classes, base_filters=64)


def get_features(model: ResNet, x: torch.Tensor) -> torch.Tensor:
    x = F.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
    x = model.dropout(x)
    return x


def _raw_to_tensor(raw):
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    if raw.dim() == 3 and raw.shape[-1] == 3:
        return raw.permute(2, 0, 1).contiguous().float() / 255.0
    return raw.float() / 255.0 if raw.dtype == torch.uint8 else raw


def _make_optim(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


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
        return self._sample_by_allocation(self._bank, alloc, self._rng)

    @staticmethod
    def _sample_by_allocation(bank, alloc, rng):
        result = []
        classes = sorted(bank.keys())
        for c, n in zip(classes, alloc):
            pool = bank[c]
            if len(pool) <= n:
                result.extend(pool)
            else:
                result.extend(rng.sample(pool, n))
        return result


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


# ── Baseline (no replay) ──

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


# ── Uniform + CE (class-balanced retrieval + CE on all logits) ──

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
                    num_old = task_id * N_CLASSES_PER_TASK
                    replay = bank.query_by_class(RETRIEVAL_BUDGET, num_old)
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


# ── DRKD ──

def run_drkd(device, train_data, val_data, class_images, mean, std, *,
             lam: float = LAMBDA):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    transform = make_train_transform(mean, std)
    exemplar_bank = ExemplarBank(CAPACITY_PER_CLASS)
    teacher_state = None

    for task_id in range(N_TASKS):
        t1 = time.time()
        if task_id > 0:
            num_old = task_id * N_CLASSES_PER_TASK
            model.expand_head(N_CLASSES_PER_TASK)

            teacher = create_model(num_old).to(device)
            teacher.load_state_dict(teacher_state)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            opt = torch.optim.SGD(
                model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
            )

            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for epoch in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits_all = model(x)
                    new_logits = logits_all[:, num_old:]
                    ce_loss = F.cross_entropy(new_logits, y - num_old, reduction='mean')
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
                    if model.fc.weight.grad is not None:
                        model.fc.weight.grad[:num_old] = 0
                    if model.fc.bias is not None and model.fc.bias.grad is not None:
                        model.fc.bias.grad[:num_old] = 0
                    opt.step()

        start_c = task_id * N_CLASSES_PER_TASK
        end_c = start_c + N_CLASSES_PER_TASK
        for c in range(start_c, end_c):
            imgs = class_images[c]
            selected = random.Random(SEED).sample(imgs, min(CAPACITY_PER_CLASS, len(imgs)))
            for img in selected:
                exemplar_bank.store_raw(img, c)
        teacher_state = {k: v.clone() for k, v in model.state_dict().items()}
        elapsed = time.time() - t1
        print(f"    Task {task_id+1:2d}/{N_TASKS} done in {elapsed:.0f}s")

    # Post-hoc classifier calibration
    print("\n    Calibrating classifier across all 100 classes...", flush=True)
    model.eval()
    feat_dim = 512
    cal_features, cal_labels = [], []
    for c in range(N_CLASSES_TOTAL):
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

    new_fc = nn.Linear(feat_dim, N_CLASSES_TOTAL).to(device)
    cal_opt = torch.optim.SGD(new_fc.parameters(), lr=0.1)
    cal_ds = TensorDataset(cal_features, cal_labels)
    cal_loader = DataLoader(cal_ds, 64, shuffle=True)

    for _ in range(20):
        for cf, cl in cal_loader:
            cf, cl = cf.to(device), cl.to(device)
            cal_loss = F.cross_entropy(new_fc(cf), cl)
            cal_opt.zero_grad(); cal_loss.backward(); cal_opt.step()

    model.fc = new_fc
    model.train()
    return _per_class_acc(model, val_data, device)


# ── main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--epochs", type=int, default=EPOCHS_PER_TASK,
                   help="epochs per task (default: 70)")
    args = p.parse_args()

    global EPOCHS_PER_TASK
    EPOCHS_PER_TASK = args.epochs

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
    print(f"  DRKD 10-task benchmark", flush=True)
    print(f"  lam={args.lam}, tau={TEMPERATURE}, epochs={EPOCHS_PER_TASK}", flush=True)
    print(f"{'='*70}", flush=True)

    all_accs = {}
    for name, fn in methods:
        t1 = time.time()
        accs = fn()
        elapsed = time.time() - t1
        all_accs[name] = accs
        avg = np.mean(accs)
        print(f"\n  {name}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for i in range(N_TASKS):
            task_accs = accs[i*N_CLASSES_PER_TASK:(i+1)*N_CLASSES_PER_TASK]
            task_avg = np.mean(task_accs)
            print(f"    task {i}: {task_avg:.3f}  ({', '.join(f'{a:.3f}' for a in task_accs)})", flush=True)
        print(f"    avg overall: {avg:.3f}  [{elapsed:.0f}s]", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Method':<30s} {'Avg':>8s} {'T0':>8s} {'T1':>8s} {'T2':>8s} {'T3':>8s} {'T4':>8s} {'T5':>8s} {'T6':>8s} {'T7':>8s} {'T8':>8s} {'T9':>8s}", flush=True)
    print(f"  {'-'*108}", flush=True)
    for name, accs in all_accs.items():
        task_avgs = [np.mean(accs[i*N_CLASSES_PER_TASK:(i+1)*N_CLASSES_PER_TASK]) for i in range(N_TASKS)]
        row = f"  {name:<30s} {np.mean(accs):>8.3f}"
        for ta in task_avgs:
            row += f" {ta:>8.3f}"
        print(row, flush=True)

    print()
    print("  Published baselines (from original config):")
    print("    StaticBank: 13.1%")
    print("    PID-GB:     10.9%")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
