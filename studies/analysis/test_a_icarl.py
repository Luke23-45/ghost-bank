"""
Adaptive iCaRL (A-iCaRL): iCaRL with task-level PI-controlled distillation.

Adds one mechanism to corrected iCaRL: a PI controller automatically adjusts
the global distillation weight λ based on held-out probe measurements of
per-class forgetting.

Key differences from corrected iCaRL:
  1. Held-out probe set (30/class) measures generalisation without
     contamination from replay buffer memorisation.
  2. After each task, the forgetting signal F_t drives a PI controller
     that adapts λ for the next task.
  3. Diagnostic logging reports the λ(t) and F(t) trajectories.

Modes:
  Default (--lam L0):      PI-controlled λ starting from L0.
  --fixed-lambda FLOAT:    Fixed-λ mode for grid sweep (no adaptation).
  --force-lambda FLOAT:    Bypass PI entirely, force λ = value
                           (regression check — must match corrected iCaRL).

Run from repo root:
  python studies/analysis/test_a_icarl.py [--lam 1.0] [--epochs 70]
  python studies/analysis/test_a_icarl.py --fixed-lambda 2.0
  python studies/analysis/test_a_icarl.py --force-lambda 1.0
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
HELD_OUT_SIZE = 30
MEMORY_TOTAL = 2000
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

LAMBDA_0 = 1.0
LAMBDA_MIN = 0.1
LAMBDA_MAX = 5.0
TEMPERATURE = 2.0

F_TARGET = 0.05
K_P = 1.0
K_I = 0.1
EMA_SMOOTH = 0.7


# ── Data ──────────────────────────────────────────────────────────────

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


# ── Budget ────────────────────────────────────────────────────────────

def compute_per_class_budget(task_id: int) -> int:
    num_seen = (task_id + 1) * N_CLASSES_PER_TASK
    return MEMORY_TOTAL // num_seen


# ── Model ─────────────────────────────────────────────────────────────

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


# ── Held-out probe ────────────────────────────────────────────────────

def extract_held_out(class_images: list[list], size: int, seed: int
                     ) -> tuple[dict[int, list], dict[int, list]]:
    held_out: dict[int, list] = {}
    herding_pool: dict[int, list] = {}
    for c in range(len(class_images)):
        imgs = list(class_images[c])
        rng = random.Random(seed + c)
        rng.shuffle(imgs)
        held_out[c] = imgs[:size]
        herding_pool[c] = imgs[size:]
    return held_out, herding_pool


def compute_probe_loss(model: ResNet, held_out_images: list,
                       target_class: int, num_old: int,
                       device) -> float:
    model.eval()
    with torch.no_grad():
        batch = torch.stack([_raw_to_tensor(raw) for raw in held_out_images]).to(device)
        logits = model(batch)[:, :num_old]
        target = torch.full((len(held_out_images),), target_class,
                            dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target).item()
    model.train()
    return loss


# ── Herding ───────────────────────────────────────────────────────────

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
    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._bank: dict[int, list] = {}
        self._pool: dict[int, list] = {}

    def store_pool(self, class_id: int, images: list):
        self._pool[class_id] = list(images)

    def select_exemplars(self, class_id: int, model: ResNet, device, budget: int):
        imgs = self._pool.get(class_id, [])
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

    def sample_for_replay(self, class_ids: list[int], budget: int, rng) -> list:
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
        if len(result) > budget:
            result = rng.sample(result, budget)
        return result


# ── iCaRL loss ────────────────────────────────────────────────────────

def sigmoid_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    return F.binary_cross_entropy(prob, targets, reduction='mean')


def icarl_loss(logits_all: torch.Tensor, y: torch.Tensor,
               num_old: int, old_logits: torch.Tensor | None,
               lam: float) -> torch.Tensor:
    num_total = logits_all.shape[1]
    num_new = num_total - num_old
    device = logits_all.device

    targets = torch.zeros_like(logits_all)
    targets.scatter_(1, y.unsqueeze(1), 1.0)

    new_logits = logits_all[:, num_old:]
    new_targets = targets[:, num_old:]
    loss_cls = sigmoid_bce(new_logits, new_targets)

    if old_logits is None or num_old == 0:
        return loss_cls * num_new

    old_logits_new = logits_all[:, :num_old]
    old_probs = torch.sigmoid(old_logits_new)
    old_probs_teacher = torch.sigmoid(old_logits)
    loss_dist = sigmoid_bce(old_logits_new, old_probs_teacher)

    return loss_cls * num_new + lam * loss_dist * num_old


# ── NME ───────────────────────────────────────────────────────────────

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


# ── PI Controller ─────────────────────────────────────────────────────

class PIController:
    """Proportional-integral controller for global λ, no D term.

    Error: e_t = F_t - F_target  (forgetting relative to target).
    Output: λ_t+1 = clamp(λ₀ + u_t, λ_min, λ_max).
    """

    def __init__(self, lam0: float, lam_min: float, lam_max: float,
                 K_p: float, K_i: float, ema_smooth: float):
        self.lam0 = lam0
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.K_p = K_p
        self.K_i = K_i
        self.ema_smooth = ema_smooth

        self.integral = 0.0
        self.e_smooth = 0.0
        self._initialised = False

        # Diagnostics
        self.lambda_history: list[float] = []
        self.F_history: list[float] = []

    def update(self, F_t: float, F_target: float) -> float:
        """Compute new λ for the NEXT task. Call after each task t ≥ 1."""
        e_t = F_t - F_target

        # Initialise EMA on first call
        if not self._initialised:
            self.e_smooth = e_t
            self._initialised = True
        else:
            self.e_smooth = self.ema_smooth * self.e_smooth + (1.0 - self.ema_smooth) * e_t

        self.integral += self.e_smooth

        # Anti-windup: clamp integral so output stays within bounds
        I_max = (self.lam_max - self.lam_min) / (2.0 * max(self.K_i, 1e-8))
        self.integral = max(-I_max, min(I_max, self.integral))

        u_t = self.K_p * self.e_smooth + self.K_i * self.integral
        lam_next = max(self.lam_min, min(self.lam_max, self.lam0 + u_t))

        self.lambda_history.append(lam_next)
        self.F_history.append(F_t)

        return lam_next

    def reset(self):
        self.integral = 0.0
        self.e_smooth = 0.0
        self._initialised = False


# ── Run ───────────────────────────────────────────────────────────────

def run_a_icarl(device, train_data, val_data, class_images, *,
                lam0: float = LAMBDA_0,
                fixed_lambda: float | None = None,
                force_lambda: float | None = None):
    # ── Held-out probe setup ──
    held_out, herding_pool = extract_held_out(class_images, HELD_OUT_SIZE, SEED)

    model = create_model(N_CLASSES_PER_TASK).to(device)
    exemplar_memory = ExemplarMemory(SEED)
    teacher_state = None

    # Reference probe losses stored after each class is first learned.
    # Each class c is always probed with ref_num_old[c] (the number of old
    # classes at the time the reference was stored), so the softmax denominator
    # is constant across comparisons for that class.
    ref_loss: dict[int, float] = {}
    ref_num_old: dict[int, int] = {}

    # PI controller (bypassed in fixed/force modes)
    if force_lambda is not None:
        pi = None
        lam_current = force_lambda
    elif fixed_lambda is not None:
        pi = None
        lam_current = fixed_lambda
    else:
        pi = PIController(
            lam0=lam0, lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX,
            K_p=K_P, K_i=K_I, ema_smooth=EMA_SMOOTH,
        )
        lam_current = lam0

    # lambda_used[t] = λ used to TRAIN task t
    # lambda_set[t]  = λ set AFTER task t (for use in task t+1)
    lambda_used: list[float] = []
    lambda_set: list[float] = []
    F_traj: list[float] = []

    for task_id in tqdm(range(N_TASKS), desc="A-iCaRL", leave=False):
        t1 = time.time()
        num_old = task_id * N_CLASSES_PER_TASK
        num_total_classes = (task_id + 1) * N_CLASSES_PER_TASK

        # ── Training with current λ ──
        if task_id == 0:
            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(model(x), y)
                    opt.zero_grad(); loss.backward(); opt.step()
        else:
            model.expand_head(N_CLASSES_PER_TASK)

            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            rng = random.Random(SEED + task_id)

            loader = DataLoader(train_data[task_id], BATCH_SIZE, shuffle=True)
            old_ids = list(range(num_old))
            for _ in range(EPOCHS_PER_TASK):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    replay_raw = exemplar_memory.sample_for_replay(
                        old_ids, RETRIEVAL_BUDGET, rng,
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

                    loss = icarl_loss(logits_all, cy, num_old, teacher_logits, lam_current)
                    opt.zero_grad(); loss.backward(); opt.step()

        lambda_used.append(lam_current)

        # ── Forgetting measurement (AFTER training) ──
        # Compare current model's probe loss to each class's stored reference.
        # The adjusted λ is used for the NEXT task.
        if task_id > 0 and pi is not None:
            rel_incs = []
            for c in range(num_old):
                n_old = ref_num_old.get(c, num_old)
                cur = compute_probe_loss(model, held_out[c], c, n_old, device)
                ref = ref_loss.get(c)
                if ref is not None and ref > 1e-8:
                    inc = (cur - ref) / ref
                    if inc > 0.0:
                        rel_incs.append(inc)
            F_t = float(np.mean(rel_incs)) if rel_incs else 0.0

            lam_current = pi.update(F_t, F_TARGET)
            F_traj.append(F_t)
            lambda_set.append(lam_current)

        # ── Herding ──
        budget = compute_per_class_budget(task_id)
        start_c = task_id * N_CLASSES_PER_TASK
        end_c = start_c + N_CLASSES_PER_TASK

        for c in range(start_c, end_c):
            exemplar_memory.store_pool(c, herding_pool[c])
            exemplar_memory.select_exemplars(c, model, device, budget)

        if task_id > 0:
            old_budget = compute_per_class_budget(task_id - 1)
            if budget < old_budget:
                for c in range(start_c):
                    exemplar_memory.select_exemplars(c, model, device, budget)

        # ── Update reference probe losses for the just-learned classes ──
        for c in range(start_c, end_c):
            pl = compute_probe_loss(model, held_out[c], c, num_total_classes, device)
            ref_loss[c] = pl
            ref_num_old[c] = num_total_classes

        teacher_state = {k: v.clone() for k, v in model.state_dict().items()}
        tqdm.write(f"    Task {task_id+1:2d}/{N_TASKS} — "
                   f"λ_used={lambda_used[-1]:.3f}  [{time.time()-t1:.0f}s]")

    # ── Report trajectory ──
    tqdm.write(f"    λ used per task: {[f'{v:.3f}' for v in lambda_used]}")
    if lambda_set:
        tqdm.write(f"    λ set after task: {[f'{v:.3f}' for v in lambda_set]}")
    if F_traj:
        tqdm.write(f"    F (forgetting) measured: {[f'{v:.4f}' for v in F_traj]}")

    # ── NME evaluation ──
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
    return accs, lambda_used, lambda_set, F_traj


def main():
    global EPOCHS_PER_TASK
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lam", type=float, default=LAMBDA_0,
                   help="base / initial λ (default: 1.0)")
    p.add_argument("--fixed-lambda", type=float, default=None,
                   help="fixed λ mode (grid sweep, no adaptation)")
    p.add_argument("--force-lambda", type=float, default=None,
                   help="force λ mode (regression check for corrected baseline)")
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

    mode = "PI-controlled"
    if args.force_lambda is not None:
        mode = f"force λ={args.force_lambda}"
    elif args.fixed_lambda is not None:
        mode = f"fixed λ={args.fixed_lambda}"

    print(f"\n{'='*70}", flush=True)
    print(f"  A-iCaRL 10-task benchmark", flush=True)
    print(f"  mode={mode}, lam0={args.lam}", flush=True)
    print(f"  F_target={F_TARGET}, K_p={K_P}, K_i={K_I}, ema={EMA_SMOOTH}", flush=True)
    print(f"  epochs={EPOCHS_PER_TASK}", flush=True)
    print(f"  Memory: total={MEMORY_TOTAL}, held_out={HELD_OUT_SIZE}/class", flush=True)
    print(f"{'='*70}", flush=True)

    t1 = time.time()
    accs, lambda_used, lambda_set, F_traj = run_a_icarl(
        device, train_data, val_data, class_images,
        lam0=args.lam,
        fixed_lambda=args.fixed_lambda,
        force_lambda=args.force_lambda,
    )
    elapsed = time.time() - t1

    avg = np.mean(accs)
    print(f"\n  A-iCaRL ({mode})", flush=True)
    print(f"  {'-'*50}", flush=True)
    for i in range(N_TASKS):
        task_accs = accs[i*N_CLASSES_PER_TASK:(i+1)*N_CLASSES_PER_TASK]
        task_avg = np.mean(task_accs)
        print(f"    task {i}: {task_avg:.3f}  ({', '.join(f'{a:.3f}' for a in task_accs)})", flush=True)
    print(f"    avg overall: {avg:.3f}  [{elapsed:.0f}s]", flush=True)

    print(f"\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()
