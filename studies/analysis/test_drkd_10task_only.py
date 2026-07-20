"""
DRKD-only: 10-task CIFAR-100 benchmark.
Skips baselines — runs only DRKD (fastest path to result).

Run from repo root:
  python studies/analysis/test_drkd_10task_only.py [--lam 1.0] [--epochs 70]
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.cifar100 import CIFAR100DataModule, CIFAR100Config
from src.models import ResNet

SEED = 13
N_TASKS = 10
N_CLASSES_PER_TASK = 10
N_CLASSES_TOTAL = 100
EPOCHS_PER_TASK = 70
BATCH_SIZE = 128
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


def prepare_task_data(class_images, val_split=0.2):
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


def run_drkd(device, train_data, val_data, class_images, *, lam: float = LAMBDA):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    exemplar_bank = ExemplarBank(CAPACITY_PER_CLASS)
    teacher_state = None

    for task_id in tqdm(range(N_TASKS), desc="DRKD", leave=False):
        t1 = time.time()
        if task_id == 0:
            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(model(x), y)
                    opt.zero_grad(); loss.backward(); opt.step()
        else:
            num_old = task_id * N_CLASSES_PER_TASK
            model.expand_head(N_CLASSES_PER_TASK)

            teacher = create_model(num_old).to(device)
            teacher.load_state_dict(teacher_state)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for _ in range(EPOCHS_PER_TASK):
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
        tqdm.write(f"    Task {task_id+1:2d}/{N_TASKS} done in {time.time()-t1:.0f}s")

    tqdm.write(f"    Calibrating classifier across all {N_CLASSES_TOTAL} classes...")
    model.eval()
    feat_dim = 512
    cal_features, cal_labels = [], []
    for c in range(N_CLASSES_TOTAL):
        imgs = exemplar_bank.items(c)
        if not imgs:
            continue
        batch = torch.stack([_raw_to_tensor(raw) for raw in imgs]).to(device)
        with torch.no_grad():
            feats = get_features(model, batch).cpu()
        cal_features.append(feats)
        cal_labels.extend([c] * len(imgs))

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

    tqdm.write("    Evaluating...")
    return _per_class_acc(model, val_data, device)


def main():
    global EPOCHS_PER_TASK
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--epochs", type=int, default=EPOCHS_PER_TASK,
                   help="epochs per task (default: 70)")
    args = p.parse_args()
    EPOCHS_PER_TASK = args.epochs

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    print("Loading CIFAR-100 data...", flush=True)
    class_images, _mean, _std = load_cifar100_data()
    print(f"Loaded {sum(len(c) for c in class_images)} images in {time.time()-t0:.1f}s", flush=True)

    train_data, val_data = prepare_task_data(class_images)
    print(f"Train per task: {[len(d) for d in train_data]}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  DRKD 10-task benchmark (DRKD only)", flush=True)
    print(f"  lam={args.lam}, tau={TEMPERATURE}, epochs={EPOCHS_PER_TASK}", flush=True)
    print(f"{'='*70}", flush=True)

    t1 = time.time()
    accs = run_drkd(device, train_data, val_data, class_images, lam=args.lam)
    elapsed = time.time() - t1

    avg = np.mean(accs)
    print(f"\n  3. DRKD (lam={args.lam})", flush=True)
    print(f"  {'-'*50}", flush=True)
    for i in range(N_TASKS):
        task_accs = accs[i*N_CLASSES_PER_TASK:(i+1)*N_CLASSES_PER_TASK]
        task_avg = np.mean(task_accs)
        print(f"    task {i}: {task_avg:.3f}  ({', '.join(f'{a:.3f}' for a in task_accs)})", flush=True)
    print(f"    avg overall: {avg:.3f}  [{elapsed:.0f}s]", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  RESULT vs published baselines", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Method':<30s} {'Avg':>8s}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'Baseline (no replay)':<30s} {7.8:>8.1f}% (from results.md)", flush=True)
    print(f"  {'StaticBank':<30s} {13.1:>8.1f}% (from results.md)", flush=True)
    print(f"  {'PID-GB':<30s} {10.9:>8.1f}% (from results.md)", flush=True)
    print(f"  {'DRKD (lam='+str(args.lam)+')':<30s} {avg*100:>8.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
