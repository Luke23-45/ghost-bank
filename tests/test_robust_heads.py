import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Adjust python path if run standalone
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.heads.cosine_margin import MarginCosineHead

NUM_SEEDS = 5
DIM = 64
NUM_CLASSES = 4
EPOCHS = 25
N_TRAIN = 1000  # samples per class
N_TEST = 200    # samples per class
MEM_PER_CLASS = 50
NOISE = 0.5


def make_class_means(num_classes: int, dim: int, seed: int) -> torch.Tensor:
    """Draw the class centers ONCE per seed; train and test sample from the SAME centers."""
    rng = torch.Generator()
    rng.manual_seed(1000 + seed)
    means = torch.randn(num_classes, dim, generator=rng)
    return F.normalize(means, p=2, dim=1) * 5.0


def sample_task(means: torch.Tensor, classes: list[int], n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample train or test points from the fixed class centers (isotropic Gaussian)."""
    X, Y = [], []
    for c in classes:
        features = means[c] + torch.randn(n_samples, DIM) * NOISE
        X.append(features)
        Y.append(torch.full((n_samples,), c, dtype=torch.long))
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)


def forward_logits(model: nn.Module, x: torch.Tensor, y: torch.Tensor, use_targets: bool) -> torch.Tensor:
    """Forward with optional per-sample targets (needed by the margin head during training)."""
    if use_targets:
        return model(x, y)
    return model(x)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    use_targets: bool,
    epochs: int = EPOCHS,
    lr: float = 0.1,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            logits = forward_logits(model, x, y, use_targets)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def make_replay_loader(means: torch.Tensor) -> DataLoader:
    """Task-1 training set: new classes + memory exemplars from task 0."""
    X_new, Y_new = sample_task(means, [2, 3], N_TRAIN)
    X_old, Y_old = sample_task(means, [0, 1], MEM_PER_CLASS)
    X, Y = torch.cat([X_new, X_old]), torch.cat([Y_new, Y_old])
    return DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)


class HeadBackbone(nn.Module):
    """Feature extractor + head, forwarding per-sample targets to heads that accept them."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x)
        if targets is not None and getattr(self.head, "accepts_targets", False):
            return self.head(feat, targets=targets)
        return self.head(feat)


def build_joint_model(head: nn.Module) -> HeadBackbone:
    """Small feature extractor + head, trained jointly (this is where recency bias lives)."""
    backbone = nn.Sequential(
        nn.Linear(DIM, DIM),
        nn.Tanh(),
        nn.Linear(DIM, DIM),
        nn.Tanh(),
    )
    return HeadBackbone(backbone, head)


def run_head_only(seed: int) -> dict:
    """Frozen-features experiment: heads trained on task 0, then task 1 with replay."""
    torch.manual_seed(seed)
    means = make_class_means(NUM_CLASSES, DIM, seed)

    loader0 = DataLoader(TensorDataset(*sample_task(means, [0, 1], N_TRAIN)), batch_size=128, shuffle=True)
    loader1 = make_replay_loader(means)
    test0 = DataLoader(TensorDataset(*sample_task(means, [0, 1], N_TEST)), batch_size=128)
    test1 = DataLoader(TensorDataset(*sample_task(means, [2, 3], N_TEST)), batch_size=128)

    results: dict[str, dict] = {}
    for name, head in [
        ("linear", nn.Linear(DIM, NUM_CLASSES)),
        ("cosine_margin", MarginCosineHead(DIM, NUM_CLASSES, scale=30.0, margin=0.35)),
    ]:
        use_targets = getattr(head, "accepts_targets", False)
        train_model(head, loader0, use_targets, lr=0.1)
        acc0_pre = evaluate(head, test0)
        train_model(head, loader1, use_targets, lr=0.05)
        acc0_post = evaluate(head, test0)
        acc1_post = evaluate(head, test1)
        results[name] = {
            "acc0_pre": acc0_pre,
            "acc0_post": acc0_post,
            "acc1_post": acc1_post,
            "forgetting": acc0_pre - acc0_post,
            "final_avg": (acc0_post + acc1_post) / 2,
        }
    return results


def run_joint(seed: int) -> dict:
    """Joint backbone+head training with replay, mirroring the real CIFAR-100 pipeline."""
    torch.manual_seed(seed)
    means = make_class_means(NUM_CLASSES, DIM, seed)

    loader0 = DataLoader(TensorDataset(*sample_task(means, [0, 1], N_TRAIN)), batch_size=128, shuffle=True)
    loader1 = make_replay_loader(means)
    test0 = DataLoader(TensorDataset(*sample_task(means, [0, 1], N_TEST)), batch_size=128)
    test1 = DataLoader(TensorDataset(*sample_task(means, [2, 3], N_TEST)), batch_size=128)

    results: dict[str, dict] = {}
    for name, head in [
        ("linear", nn.Linear(DIM, NUM_CLASSES)),
        ("cosine_margin", MarginCosineHead(DIM, NUM_CLASSES, scale=30.0, margin=0.35)),
    ]:
        use_targets = getattr(head, "accepts_targets", False)
        model = build_joint_model(head)
        train_model(model, loader0, use_targets, lr=0.05)
        acc0_pre = evaluate(model, test0)
        train_model(model, loader1, use_targets, lr=0.02)
        acc0_post = evaluate(model, test0)
        acc1_post = evaluate(model, test1)

        lin = model.head if isinstance(model.head, nn.Linear) else None
        old_norm = torch.norm(lin.weight[:2], dim=1).mean().item() if lin else float("nan")
        new_norm = torch.norm(lin.weight[2:], dim=1).mean().item() if lin else float("nan")

        results[name] = {
            "acc0_pre": acc0_pre,
            "acc0_post": acc0_post,
            "acc1_post": acc1_post,
            "forgetting": acc0_pre - acc0_post,
            "final_avg": (acc0_post + acc1_post) / 2,
            "w_norm_old": old_norm,
            "w_norm_new": new_norm,
        }
    return results


def summarize(name: str, results: list[dict]) -> None:
    keys = ["acc0_pre", "acc0_post", "acc1_post", "forgetting", "final_avg"]
    print(f"\n  {name}")
    for k in keys:
        vals = [r[k] for r in results]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"    {k:12s} = {mean * 100:6.1f}% +/- {std * 100:5.1f}%")
    for extra in ("w_norm_old", "w_norm_new"):
        vals = [r[extra] for r in results if extra in r and not isinstance(r[extra], float) or extra in r]
        vals = [r[extra] for r in results if extra in r]
        if vals and all(v == vals[0] or not (v != v) for v in vals):
            mean = sum(vals) / len(vals)
            print(f"    {extra:12s} = {mean:.2f}")


def run_experiment() -> None:
    print(f"=== Head-only (frozen features, shared train/test centers, {NUM_SEEDS} seeds) ===")
    ho = {"linear": [], "cosine_margin": []}
    for seed in range(NUM_SEEDS):
        res = run_head_only(seed)
        ho["linear"].append(res["linear"])
        ho["cosine_margin"].append(res["cosine_margin"])
    summarize("linear", ho["linear"])
    summarize("cosine_margin", ho["cosine_margin"])

    print(f"\n=== Joint backbone+head training with replay ({NUM_SEEDS} seeds) ===")
    jo = {"linear": [], "cosine_margin": []}
    for seed in range(NUM_SEEDS):
        res = run_joint(seed)
        jo["linear"].append(res["linear"])
        jo["cosine_margin"].append(res["cosine_margin"])
    summarize("linear", jo["linear"])
    summarize("cosine_margin", jo["cosine_margin"])


if __name__ == "__main__":
    run_experiment()
