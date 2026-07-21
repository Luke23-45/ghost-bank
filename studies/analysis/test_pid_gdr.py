"""
PID-Guided Distillation Replay (PID-GDR): iCaRL + per-class PID-weighted KL.

Bridges the gap between decoupled distillation (DRKD/PID-DDC) and replay-based
methods (iCaRL).  Key idea: use iCaRL's uniform exemplar replay for stability,
but replace its sigmoid BCE distillation with per-class PID-weighted KL
divergence on the full softmax over old classes — so classes experiencing more
feature drift receive proportionally stronger distillation pressure.

Key differences from iCaRL:
  1. Softmax + KL divergence (with temperature)  instead of sigmoid BCE
  2. Per-class PID weighting on the KL term      instead of uniform λ
  3. Old FC rows frozen (gradient masked)        "only θ and new rows updated"

Key differences from PID-DDC:
  1. Exemplars IN the gradient stream             replay during SGD
  2. Herding for exemplar selection               not random
  3. NME classification at test time              not calibration
  4. Per-task PID probe at task boundaries        (same as PID-DDC)

Per the GDR proposal (§3.2 Step 3): old classifier rows are frozen to prevent
decision-boundary drift; only θ (feature extractor) and new-class rows receive
gradients.  Replay samples provide ground-truth CE signal through the frozen old
weights via `∂L_CE/∂z_i · W_old[:,i]`, giving the backbone a direct learning
signal absent in pure-decoupling methods.

Run from repo root:
  python studies/analysis/test_pid_gdr.py [--lam 1.0] [--alpha 1.0] [--freeze-old-fc] [--epochs 70]
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
from src.bank.core.pid_controller import PIDController

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
ALPHA = 1.0
TEMPERATURE = 2.0
FREEZE_OLD_FC = True

K_P = 1.0
K_I = 0.1
K_D = 0.5
PID_DECAY = 0.99
PID_SMOOTH = 0.9


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


# ── PID-GDR loss ──────────────────────────────────────────────────────

def pid_gdr_loss(logits_all: torch.Tensor, y: torch.Tensor,
                 num_old: int, teacher_logits: torch.Tensor | None,
                 lambda_c: torch.Tensor | None,
                 temperature: float, lam: float) -> torch.Tensor:
    """CE on new classes + PID-weighted KL on old classes (all samples).

    CE is applied only to samples whose label >= num_old (new data).
    KL is applied to all (B+R) samples over old-class logits.

    Args:
        logits_all:  (B+R, num_total) student logits
        y:           (B+R,) labels
        num_old:     number of old classes
        teacher_logits: (B+R, num_old) or None
        lambda_c:    (num_old,) per-class KD weights = (1 + α·debt_c) or None
        temperature: distillation temperature τ
        lam:         global KD multiplier
    """
    num_total = logits_all.shape[1]
    num_new = num_total - num_old

    new_logits = logits_all[:, num_old:]
    new_mask = y >= num_old
    if new_mask.any():
        ce_loss = F.cross_entropy(new_logits[new_mask], y[new_mask] - num_old, reduction='mean')
    else:
        ce_loss = torch.tensor(0.0, device=logits_all.device)

    if teacher_logits is None or num_old == 0 or lambda_c is None:
        return ce_loss

    student_old = logits_all[:, :num_old] / temperature
    teacher_old = teacher_logits / temperature

    student_logp = F.log_softmax(student_old, dim=1)
    teacher_p = F.softmax(teacher_old, dim=1)

    lc = lambda_c.to(logits_all.device).unsqueeze(0)
    kl = teacher_p * (torch.log(teacher_p + 1e-10) - student_logp)
    kd_loss = (kl * lc).sum(dim=1).mean() * (temperature ** 2)

    return ce_loss + lam * kd_loss


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


# ── PID probe ──────────────────────────────────────────────────────────

def compute_probe_losses(model: ResNet, exemplar_memory: ExemplarMemory,
                         class_ids: list[int], num_old: int,
                         device) -> list[float | None]:
    """Gradient-free CE probe on old-class logits only.

    Returns per-class CE loss aligned with class_ids; None if insufficient
    exemplars for that class.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for c in class_ids:
            imgs = exemplar_memory.items(c)
            if not imgs or len(imgs) < 2:
                losses.append(None)
                continue
            batch = torch.stack([_raw_to_tensor(raw) for raw in imgs]).to(device)
            logits = model(batch)[:, :num_old]
            target = torch.full((len(imgs),), c, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target).item()
            losses.append(loss)
    model.train()
    return losses


def debt_to_lambda(debt: list[float], alpha: float) -> torch.Tensor:
    """Convert per-class PID debt to per-class KD weight λ_c = 1 + α·debt_c."""
    arr = [1.0 + alpha * d for d in debt]
    return torch.tensor(arr, dtype=torch.float)


# ── Run ───────────────────────────────────────────────────────────────

def run_pid_gdr(device, train_data, val_data, class_images, *,
                lam: float = LAMBDA, alpha: float = ALPHA,
                freeze_old_fc: bool = FREEZE_OLD_FC):
    model = create_model(N_CLASSES_PER_TASK).to(device)
    exemplar_memory = ExemplarMemory(CAPACITY_PER_CLASS, SEED)
    teacher_state = None

    total_classes_seen = 0
    pid = None
    old_class_ids: list[int] = []

    for task_id in tqdm(range(N_TASKS), desc="PID-GDR", leave=False):
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

            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            rng = random.Random(SEED + task_id)

            # ── PID probe at task boundary ──
            probe_losses = compute_probe_losses(model, exemplar_memory, old_class_ids, num_old, device)
            raw_debt = pid.update(probe_losses)
            lambda_c = debt_to_lambda(raw_debt, alpha)

            tqdm.write(f"    Task {task_id+1:2d} — PID debts: "
                       f"max={max(raw_debt):.4f} mean={np.mean(raw_debt):.4f} "
                       f"λc range=[{lambda_c.min().item():.4f}, {lambda_c.max().item():.4f}]")

            # ── Training with exemplar replay + PID-weighted KD ──
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

                    loss = pid_gdr_loss(logits_all, cy, num_old, teacher_logits,
                                        lambda_c, TEMPERATURE, lam)
                    opt.zero_grad(); loss.backward()

                    if freeze_old_fc:
                        if model.fc.weight.grad is not None:
                            model.fc.weight.grad[:num_old] = 0
                        if model.fc.bias is not None and model.fc.bias.grad is not None:
                            model.fc.bias.grad[:num_old] = 0

                    opt.step()

        # ── Herding: store exemplars for current task ──
        start_c = task_id * N_CLASSES_PER_TASK
        end_c = start_c + N_CLASSES_PER_TASK
        for c in range(start_c, end_c):
            exemplar_memory.store_all(c, class_images[c])
            exemplar_memory.select_exemplars(c, model, device, CAPACITY_PER_CLASS)

        # ── Save teacher state ──
        teacher_state = {k: v.clone() for k, v in model.state_dict().items()}
        total_classes_seen += N_CLASSES_PER_TASK

        # ── Expand PID controller ──
        if task_id == 0:
            pid = PIDController(
                num_classes=N_CLASSES_PER_TASK,
                K_p=K_P, K_i=K_I, K_d=K_D,
                decay=PID_DECAY, smooth=PID_SMOOTH,
            )
        else:
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
            pid = new_pid

        old_class_ids = list(range(total_classes_seen))

        tqdm.write(f"    Task {task_id+1:2d}/{N_TASKS} done in {time.time()-t1:.0f}s")

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
    return accs


def main():
    global EPOCHS_PER_TASK
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lam", type=float, default=LAMBDA,
                   help="global KD weight λ (default: 1.0)")
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="debt-to-lambda scaling α (default: 1.0)")
    p.add_argument("--no-freeze-old-fc", action="store_false", dest="freeze_old_fc",
                   default=FREEZE_OLD_FC,
                   help="disable old-class FC freezing (default: frozen)")
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
    print(f"  PID-GDR 10-task benchmark", flush=True)
    print(f"  lam={args.lam}, alpha={args.alpha}, tau={TEMPERATURE}", flush=True)
    print(f"  freeze_old_fc={args.freeze_old_fc}, epochs={EPOCHS_PER_TASK}", flush=True)
    print(f"{'='*70}", flush=True)

    t1 = time.time()
    accs = run_pid_gdr(device, train_data, val_data, class_images,
                       lam=args.lam, alpha=args.alpha,
                       freeze_old_fc=args.freeze_old_fc)
    elapsed = time.time() - t1

    avg = np.mean(accs)
    print(f"\n  PID-GDR (lam={args.lam}, alpha={args.alpha})", flush=True)
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
    print(f"  {'iCaRL+FC-GM':<30s} {1.8:>8.1f}%", flush=True)
    print(f"  {'DRKD (lam=1.0)':<30s} {18.1:>8.1f}%", flush=True)
    print(f"  {'PID-DDC (lam0=1.0, α=1.0)':<30s} {19.1:>8.1f}%", flush=True)
    fc_str = "frozen" if args.freeze_old_fc else "trainable"
    print(f"  {'PID-GDR (lam='+str(args.lam)+', α='+str(args.alpha)+', FC='+fc_str+')':<30s} {avg*100:>8.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
