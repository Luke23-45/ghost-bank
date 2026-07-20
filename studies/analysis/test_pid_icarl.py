"""
PID-iCaRL: iCaRL with per-class adaptive KD weights via PID controller.

Extends iCaRL (Rebuffi et al. 2017) with two PID-driven enhancements:
  1. Per-class λ_c for the distillation loss (forgotten classes get stronger KD)
  2. Replay budget proportional to PID debt (forgotten classes get more exemplars)

The probe (gradient-free) at each task boundary measures per-class sigmoid BCE
on stored exemplars.  The PID controller converts this into per-class debt,
which drives both λ_c and budget allocation.

Run from repo root:
  python studies/analysis/test_pid_icarl.py [--epochs 70] [--lam0 1.0] [--alpha 1.0]
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

from src.bank.core.pid_controller import PIDController
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

# PID hyperparameters (same as PID-DDC)
K_P = 1.0
K_I = 0.1
K_D = 0.5
PID_DECAY = 0.99
PID_SMOOTH = 0.9
ALPHA = 1.0


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
    mu = features.mean(dim=0, keepdim=True)
    selected = []
    selected_set = set()
    for _ in range(budget):
        best_idx = -1
        best_dist = float("inf")
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
        self._bank[class_id] = list(images)

    def select_exemplars(self, class_id: int, model: ResNet, device, budget: int):
        imgs = self._bank.get(class_id, [])
        if not imgs:
            return
        budget = min(budget, len(imgs))
        batch = torch.stack([_raw_to_tensor(raw) for raw in imgs]).to(device)
        with torch.no_grad():
            feats = get_features(model, batch).cpu()
        indices = herding_select(feats, budget, self._rng)
        self._bank[class_id] = [imgs[i] for i in indices]

    def items(self, class_id: int) -> list:
        return self._bank.get(class_id, [])

    def sample_for_replay(self, class_ids: list[int], budget: int, rng,
                          per_class_budget: list[int] | None = None) -> list:
        """Sample exemplars for replay.

        When per_class_budget is provided, each class gets that many slots
        (allowing debt-driven allocation).  Otherwise, samples uniformly.
        Returns list of (raw_tensor, class_label) tuples.
        """
        if not class_ids or budget == 0:
            return []
        result: list[tuple] = []
        if per_class_budget is not None:
            for c, alloc in zip(class_ids, per_class_budget):
                pool = self._bank.get(c, [])
                if not pool:
                    continue
                k = min(alloc, len(pool))
                if k > 0:
                    selected = rng.sample(pool, k)
                    result.extend([(img, c) for img in selected])
        else:
            per_class = max(1, budget // len(class_ids))
            for c in class_ids:
                pool = self._bank.get(c, [])
                if not pool:
                    continue
                k = min(per_class, len(pool))
                selected = rng.sample(pool, k)
                result.extend([(img, c) for img in selected])
        if len(result) > budget:
            rng = random.Random(42)
            result = rng.sample(result, budget)
        return result


# ── iCaRL loss (with per-class λ support) ─────────────────────────────

def sigmoid_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    return F.binary_cross_entropy(prob, targets, reduction='mean')


def icarl_loss_pid(logits_all: torch.Tensor, y: torch.Tensor,
                   num_old: int, old_logits: torch.Tensor | None,
                   lambda_c: torch.Tensor | None) -> torch.Tensor:
    """iCaRL loss with per-class λ_c on the distillation term.

    When lambda_c is None (task 0 or no old classes), falls back to
    plain CE on new classes.
    """
    num_total = logits_all.shape[1]
    num_new = num_total - num_old

    targets = torch.zeros_like(logits_all)
    targets.scatter_(1, y.unsqueeze(1), 1.0)

    # Classification loss on NEW classes
    new_logits = logits_all[:, num_old:]
    new_targets = targets[:, num_old:]
    loss_cls = sigmoid_bce(new_logits, new_targets)

    if old_logits is None or num_old == 0 or lambda_c is None:
        return loss_cls * num_new

    # Per-class weighted distillation on OLD classes
    old_logits_new = logits_all[:, :num_old]
    old_probs_teacher = torch.sigmoid(old_logits)
    prob = torch.sigmoid(old_logits_new)
    bce = F.binary_cross_entropy(prob, old_probs_teacher, reduction='none')

    # (B, num_old) → (num_old,) per-class mean, then weight by λ_c and sum
    loss_dist = (bce.mean(dim=0) * lambda_c).sum()

    return loss_cls * num_new + loss_dist


# ── NME classification ────────────────────────────────────────────────

def nme_classify(model: ResNet, exemplar_memory: ExemplarMemory,
                 x: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        feat = get_features(model, x.to(device))
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


# ── PID probe and helpers ─────────────────────────────────────────────

def compute_probe_losses_icarl(model: ResNet, exemplar_memory: ExemplarMemory,
                               num_old: int, device) -> list[float | None]:
    """Gradient-free probe: per-class sigmoid BCE on exemplars.

    For each old class c, measures the BCE(sigmoid(logit), one-hot(c))
    over only old-class logits.  Higher loss → more forgetting.
    Returns list aligned with classes 0..num_old-1; None if too few exemplars.
    """
    model.eval()
    losses: list[float | None] = []
    with torch.no_grad():
        for c in range(num_old):
            imgs = exemplar_memory.items(c)
            if not imgs or len(imgs) < 2:
                losses.append(None)
                continue
            batch = torch.stack([_raw_to_tensor(raw) for raw in imgs]).to(device)
            logits = model(batch)[:, :num_old]
            target = torch.zeros_like(logits)
            target[:, c] = 1.0
            loss = F.binary_cross_entropy_with_logits(logits, target).item()
            losses.append(loss)
    model.train()
    return losses


def debt_to_lambda(debt: list[float], lam0: float, alpha: float) -> torch.Tensor:
    """Convert per-class PID debt to per-class KD weight λ_c = lam0 · (1 + α · d_c)."""
    arr = [lam0 * (1.0 + alpha * d) for d in debt]
    return torch.tensor(arr, dtype=torch.float)


def allocate_budget_proportional(debt: list[float | None],
                                 total_budget: int) -> list[int]:
    """Allocate replay budget proportionally to PID debt (floored at 0.1)."""
    debt_arr = np.array([max(d, 0.1) if d is not None else 0.1 for d in debt])
    weights = debt_arr / debt_arr.sum()
    alloc = np.floor(weights * total_budget).astype(int)
    remainder = int(total_budget - alloc.sum())
    for idx in np.argsort(-weights):
        if remainder <= 0:
            break
        alloc[idx] += 1
        remainder -= 1
    return alloc.tolist()


def expand_pid_controller(pid: PIDController) -> PIDController:
    """Expand PID controller by N_CLASSES_PER_TASK while preserving old state."""
    new_num = pid.num_classes + N_CLASSES_PER_TASK
    new_pid = PIDController(
        num_classes=new_num,
        K_p=K_P, K_i=K_I, K_d=K_D,
        decay=PID_DECAY, smooth=PID_SMOOTH,
    )
    old_state = pid.state_dict()
    new_state = new_pid.state_dict()
    for key in old_state:
        new_state[key][:pid.num_classes] = old_state[key]
    new_pid.load_state_dict(new_state)
    return new_pid


# ── Run ───────────────────────────────────────────────────────────────

def run_pid_icarl(device, train_data, val_data, class_images, *,
                  lam0: float = LAMBDA, alpha: float = ALPHA):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    exemplar_memory = ExemplarMemory(CAPACITY_PER_CLASS, SEED)
    teacher_state = None
    pid: PIDController | None = None

    for task_id in tqdm(range(N_TASKS), desc="PID-iCaRL", leave=False):
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

            # ── PID probe before training this task ──
            probe_losses = compute_probe_losses_icarl(model, exemplar_memory, num_old, device)
            raw_debt = pid.update(probe_losses)
            lambda_c = debt_to_lambda(raw_debt, lam0, alpha).to(device)

            # Per-class replay budget allocation
            per_class_budget = allocate_budget_proportional(raw_debt, RETRIEVAL_BUDGET)

            tqdm.write(
                f"    Task {task_id+1:2d} — PID debts: "
                f"max={max(raw_debt):.2f} mean={np.mean(raw_debt):.2f} "
                f"λ range=[{lambda_c.min().item():.2f}, {lambda_c.max().item():.2f}]"
            )

            # ── Training with per-class weighted KD + debt-driven replay ──
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            old_class_ids = list(range(num_old))
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    replay_raw = exemplar_memory.sample_for_replay(
                        old_class_ids, RETRIEVAL_BUDGET, rng,
                        per_class_budget=per_class_budget,
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

                    logits_all = model(cx)
                    teacher_logits = None
                    if teacher_state is not None:
                        teacher = create_model(num_old).to(device)
                        teacher.load_state_dict(teacher_state)
                        teacher.eval()
                        with torch.no_grad():
                            teacher_logits = teacher(cx)

                    loss = icarl_loss_pid(logits_all, cy, num_old,
                                          teacher_logits, lambda_c)
                    opt.zero_grad(); loss.backward(); opt.step()

        # Store exemplars for current task via herding (ALL tasks, including task 0)
        start_c = task_id * N_CLASSES_PER_TASK
        end_c = start_c + N_CLASSES_PER_TASK
        for c in range(start_c, end_c):
            exemplar_memory.store_all(c, class_images[c])
            exemplar_memory.select_exemplars(c, model, device, CAPACITY_PER_CLASS)

        teacher_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Expand PID controller after task 0 (creates initial PID)
        if task_id == 0:
            pid = PIDController(
                num_classes=N_CLASSES_PER_TASK,
                K_p=K_P, K_i=K_I, K_d=K_D,
                decay=PID_DECAY, smooth=PID_SMOOTH,
            )
        else:
            pid = expand_pid_controller(pid)

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
    p.add_argument("--lam0", type=float, default=LAMBDA,
                   help="base KD weight λ₀ (default: 1.0)")
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="debt-to-lambda scaling α (default: 1.0)")
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
    print(f"  PID-iCaRL 10-task benchmark", flush=True)
    print(f"  lam0={args.lam0}, alpha={args.alpha}, epochs={EPOCHS_PER_TASK}", flush=True)
    print(f"{'='*70}", flush=True)

    t1 = time.time()
    accs = run_pid_icarl(device, train_data, val_data, class_images,
                         lam0=args.lam0, alpha=args.alpha)
    elapsed = time.time() - t1

    avg = np.mean(accs)
    print(f"\n  PID-iCaRL (lam0={args.lam0}, alpha={args.alpha})", flush=True)
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
    print(f"  {'iCaRL':<30s} {37.4:>8.1f}%", flush=True)
    print(f"  {'PID-iCaRL (ours)':<30s} {avg*100:>8.1f}%", flush=True)
    print(f"  {'DRKD (lam=1.0)':<30s} {18.1:>8.1f}%", flush=True)
    print(f"  {'PID-DDC (lam0=1.0, α=1.0)':<30s} {19.1:>8.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
