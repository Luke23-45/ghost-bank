"""
iCaRL: Class-Incremental Learning with Exemplar Memory (Rebuffi et al. 2017).

Faithful implementation in the same framework as DRKD/PID-DDC for honest
side-by-side comparison on the 10-task CIFAR-100 benchmark.

Key differences from DRKD:
  1. Exemplars ARE used in the gradient stream (replay during SGD)
  2. Herding for exemplar selection (not random)
  3. Sigmoid BCE + distillation loss (not softmax CE + KL)
  4. Nearest-mean-of-exemplars (NME) at test time (not linear classifier)

Run from repo root:
  python studies/analysis/test_icarl.py [--epochs 70]
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


# ── Herding exemplar selection (iCaRL §6) ─────────────────────────────

def herding_select(features: torch.Tensor, budget: int, rng) -> list[int]:
    """Greedy herding: select `budget` indices minimizing mean distance to class mean."""
    mu = features.mean(dim=0, keepdim=True)
    selected = []
    selected_set = set()
    for _ in range(budget):
        best_idx = -1
        best_dist = float("inf")
        # Compute mean of currently selected + candidate
        for i in range(len(features)):
            if i in selected_set:
                continue
            cand_indices = selected + [i]
            cand_mean = features[cand_indices].mean(dim=0, keepdim=True)
            dist = (mu - cand_mean).norm().item()
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            selected.append(best_idx)
            selected_set.add(best_idx)
    return selected


class ExemplarMemory:
    def __init__(self, capacity_per_class: int, seed: int):
        self._bank: dict[int, list[torch.Tensor]] = {}
        self._cap = capacity_per_class
        self._rng = random.Random(seed)

    def store_all(self, class_id: int, images: list):
        """Store all images for a class (will be herded later)."""
        self._bank[class_id] = list(images)

    def select_exemplars(self, class_id: int, model: ResNet, device, budget: int):
        """Run herding to select `budget` exemplars for class_id."""
        imgs = self._bank.get(class_id, [])
        if not imgs:
            return
        budget = min(budget, len(imgs))
        # Extract features for all images of this class
        batch = torch.stack([_raw_to_tensor(raw) for raw in imgs]).to(device)
        with torch.no_grad():
            feats = get_features(model, batch).cpu()
        # Herding selection
        indices = herding_select(feats, budget, self._rng)
        self._bank[class_id] = [imgs[i] for i in indices]

    def items(self, class_id: int) -> list:
        return self._bank.get(class_id, [])

    def sample_for_replay(self, class_ids: list[int], budget: int, rng) -> list:
        """Sample exemplars uniformly across classes for replay batch.

        Returns list of (raw_tensor, class_label) tuples.
        """
        if not class_ids or budget == 0:
            return []
        per_class = max(1, budget // len(class_ids))
        result: list[tuple] = []
        for c in class_ids:
            pool = self._bank.get(c, [])
            if not pool:
                continue
            k = min(per_class, len(pool))
            selected = rng.sample(pool, k)
            result.extend([(img, c) for img in selected])
        # Trim to exact budget
        if len(result) > budget:
            result = rng.sample(result, budget)
        return result


# ── iCaRL loss ────────────────────────────────────────────────────────

def sigmoid_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with sigmoid for multi-class."""
    prob = torch.sigmoid(logits)
    return F.binary_cross_entropy(prob, targets, reduction='mean')


def icarl_loss(logits_all: torch.Tensor, y: torch.Tensor,
               num_old: int, old_logits: torch.Tensor | None,
               lam: float) -> torch.Tensor:
    """iCaRL combined loss.

    For new classes: BCE(sigmoid(logit), one-hot target)
    For old classes (if old_logits provided): distillation
        BCE(sigmoid(current_logit), sigmoid(old_logit))
    """
    num_total = logits_all.shape[1]
    num_new = num_total - num_old
    device = logits_all.device

    # One-hot targets for BCE
    targets = torch.zeros_like(logits_all)
    targets.scatter_(1, y.unsqueeze(1), 1.0)

    # Classification loss on NEW classes: BCE(sigmoid(logit), target)
    new_logits = logits_all[:, num_old:]
    new_targets = targets[:, num_old:]
    loss_cls = sigmoid_bce(new_logits, new_targets)

    if old_logits is None or num_old == 0:
        return loss_cls * num_new

    # Distillation on OLD classes: BCE(sigmoid(new_logit), sigmoid(old_logit))
    old_logits_new = logits_all[:, :num_old]
    old_probs = torch.sigmoid(old_logits_new)
    old_probs_teacher = torch.sigmoid(old_logits)
    loss_dist = sigmoid_bce(old_logits_new, old_probs_teacher)

    return loss_cls * num_new + lam * loss_dist * num_old


# ── NME classification ────────────────────────────────────────────────

def nme_classify(model: ResNet, exemplar_memory: ExemplarMemory,
                 x: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    """Nearest-mean-of-exemplars classification.

    Computes class prototypes from stored exemplars, then assigns
    the test sample to the class with the nearest prototype (L2 distance).
    """
    model.eval()
    with torch.no_grad():
        feat = get_features(model, x.to(device))
    # Compute prototypes for all classes seen so far
    best_class = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
    best_dist = torch.full((x.size(0),), float("inf"), device=device)
    for c in range(num_classes):
        imgs = exemplar_memory.items(c)
        if not imgs:
            continue
        batch = torch.stack([_raw_to_tensor(raw) for raw in imgs]).to(device)
        with torch.no_grad():
            proto = get_features(model, batch).mean(dim=0, keepdim=True)
        dist = (feat - proto).norm(dim=1)
        mask = dist < best_dist
        best_dist[mask] = dist[mask]
        best_class[mask] = c
    model.train()
    return best_class


# ── Run ────────────────────────────────────────────────────────────────

def run_icarl(device, train_data, val_data, class_images, *, lam: float = LAMBDA):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    exemplar_memory = ExemplarMemory(CAPACITY_PER_CLASS, SEED)
    teacher_state = None

    for task_id in tqdm(range(N_TASKS), desc="iCaRL", leave=False):
        t1 = time.time()

        if task_id == 0:
            # First task: standard CE training
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

            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            rng = random.Random(SEED + task_id)

            # Training with exemplar replay
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    # Retrieve exemplars for replay
                    old_class_ids = list(range(num_old))
                    replay_raw = exemplar_memory.sample_for_replay(
                        old_class_ids, RETRIEVAL_BUDGET, rng,
                    )
                    if replay_raw:
                        rxs, rys = [], []
                        for raw, lbl in replay_raw:
                            t = _raw_to_tensor(raw).unsqueeze(0).to(device)
                            rxs.append(t); rys.append(lbl)
                        rx = torch.cat(rxs)
                        ry = torch.tensor(rys, device=device, dtype=torch.long)
                        cx = torch.cat([x, rx]); cy = torch.cat([y, ry])
                    else:
                        cx, cy = x, y

                    # Combined loss
                    logits_all = model(cx)
                    teacher_logits = None
                    if teacher_state is not None:
                        teacher = create_model(num_old).to(device)
                        teacher.load_state_dict(teacher_state)
                        teacher.eval()
                        with torch.no_grad():
                            teacher_logits = teacher(cx)

                    loss = icarl_loss(logits_all, cy, num_old, teacher_logits, lam)
                    opt.zero_grad(); loss.backward(); opt.step()

        # Store exemplars for current task via herding (ALL tasks, including task 0)
        start_c = task_id * N_CLASSES_PER_TASK
        end_c = start_c + N_CLASSES_PER_TASK
        for c in range(start_c, end_c):
            exemplar_memory.store_all(c, class_images[c])
            exemplar_memory.select_exemplars(c, model, device, CAPACITY_PER_CLASS)

        teacher_state = {k: v.clone() for k, v in model.state_dict().items()}
        tqdm.write(f"    Task {task_id+1:2d}/{N_TASKS} done in {time.time()-t1:.0f}s")

    # ── Evaluate with NME ──
    tqdm.write(f"    Evaluating with NME...")
    model.eval()
    correct, total = {}, {}
    for t in range(N_TASKS):
        val_loader = DataLoader(val_data[t], 128, shuffle=False)
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = nme_classify(model, exemplar_memory, x, N_CLASSES_TOTAL, device)
            for i in range(len(y)):
                c = int(y[i])
                correct[c] = correct.get(c, 0) + (preds[i] == y[i]).item()
                total[c] = total.get(c, 0) + 1
    accs = [correct[c] / total[c] if total.get(c, 0) > 0 else 0.0 for c in range(N_CLASSES_TOTAL)]
    model.train()
    return accs


def main():
    global EPOCHS_PER_TASK
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lam", type=float, default=LAMBDA,
                   help="distillation weight (default: 1.0)")
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
    print(f"  iCaRL 10-task benchmark", flush=True)
    print(f"  lam={args.lam}, epochs={EPOCHS_PER_TASK}", flush=True)
    print(f"{'='*70}", flush=True)

    t1 = time.time()
    accs = run_icarl(device, train_data, val_data, class_images, lam=args.lam)
    elapsed = time.time() - t1

    avg = np.mean(accs)
    print(f"\n  iCaRL (lam={args.lam})", flush=True)
    print(f"  {'-'*50}", flush=True)
    for i in range(N_TASKS):
        task_accs = accs[i*N_CLASSES_PER_TASK:(i+1)*N_CLASSES_PER_TASK]
        task_avg = np.mean(task_accs)
        print(f"    task {i}: {task_avg:.3f}  ({', '.join(f'{a:.3f}' for a in task_accs)})", flush=True)
    print(f"    avg overall: {avg:.3f}  [{elapsed:.0f}s]", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Method':<30s} {'Avg':>8s}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'Baseline (no replay)':<30s} {7.8:>8.1f}%", flush=True)
    print(f"  {'StaticBank':<30s} {13.1:>8.1f}%", flush=True)
    print(f"  {'PID-GB':<30s} {10.9:>8.1f}%", flush=True)
    print(f"  {'iCaRL (ours)':<30s} {avg*100:>8.1f}%", flush=True)
    print(f"  {'DRKD (lam=1.0)':<30s} {18.1:>8.1f}%", flush=True)
    print(f"  {'PID-DDC (lam0=1.0, α=1.0)':<30s} {19.1:>8.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
