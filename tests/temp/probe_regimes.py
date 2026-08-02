"""Probe regimes where linear-head recency bias actually appears, comparing heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys = None
from pathlib import Path
import sys as _sys
_sys.path.append(str(Path("tests").resolve()))
import test_robust_heads as T
from src.models.heads.cosine_margin import MarginCosineHead

DIM = T.DIM


def make_means(num_classes, dim, seed, norm):
    rng = torch.Generator()
    rng.manual_seed(2000 + seed)
    means = torch.randn(num_classes, dim, generator=rng)
    return F.normalize(means, p=2, dim=1) * norm


def sample(means, classes, n, noise):
    X, Y = [], []
    for c in classes:
        X.append(means[c] + torch.randn(n, DIM) * noise)
        Y.append(torch.full((n,), c, dtype=torch.long))
    return torch.cat(X), torch.cat(Y)


class HB(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x, targets=None):
        feat = self.backbone(x)
        if targets is not None and getattr(self.head, "accepts_targets", False):
            return self.head(feat, targets=targets)
        return self.head(feat)


def make_backbone(nonneg):
    act = nn.ReLU() if nonneg else nn.Tanh()
    return nn.Sequential(nn.Linear(DIM, DIM), act, nn.Linear(DIM, DIM), act)


def train(model, loader, use_targets, epochs, lr, wd):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            logits = model(x, y) if use_targets else model(x)
            F.cross_entropy(logits, y).backward()
            opt.step()


def ev(model, loader):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, y in loader:
            ok += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
    return ok / tot


def run_regime(regime, seed):
    torch.manual_seed(seed)
    nc = regime["classes_total"]
    means = make_means(nc, DIM, seed, regime["norm"])
    t0 = [0, 1] if nc == 4 else [0, 1, 2, 3]
    t1 = [2, 3] if nc == 4 else [4, 5, 6, 7]
    mem_per = regime["mem_per_class"]
    X_old, Y_old = sample(means, t0, mem_per, regime["noise"])

    out = {}
    for name, head in [
        ("linear", nn.Linear(DIM, nc)),
        ("cosine", MarginCosineHead(DIM, nc, scale=regime["scale"], margin=0.0)),
        ("cosine_margin", MarginCosineHead(DIM, nc, scale=regime["scale"], margin=regime["margin"])),
    ]:
        model = HB(make_backbone(regime["nonneg"]), head)
        ut = getattr(head, "accepts_targets", False)
        loader0 = DataLoader(TensorDataset(*sample(means, t0, regime["n_train"], regime["noise"])), 128, shuffle=True)
        loader1 = DataLoader(TensorDataset(torch.cat([sample(means, t1, regime["n_train"], regime["noise"])[0], X_old]),
                                           torch.cat([sample(means, t1, regime["n_train"], regime["noise"])[1], Y_old])), 128, shuffle=True)
        test0 = DataLoader(TensorDataset(*sample(means, t0, 200, regime["noise"])), 128)
        test1 = DataLoader(TensorDataset(*sample(means, t1, 200, regime["noise"])), 128)
        train(model, loader0, ut, regime["epochs_t0"], regime["lr"], regime["wd"])
        a0_pre = ev(model, test0)
        train(model, loader1, ut, regime["epochs_t1"], regime["lr"] * 0.4, regime["wd"])
        a0_post = ev(model, test0)
        a1_post = ev(model, test1)
        lin = head if isinstance(head, nn.Linear) else None
        old_n = torch.norm(lin.weight[: len(t0)], dim=1).mean().item() if lin else float("nan")
        new_n = torch.norm(lin.weight[len(t0):], dim=1).mean().item() if lin else float("nan")
        out[name] = dict(forget=a0_pre - a0_post, acc1=a1_post, old_n=old_n, new_n=new_n)
    return out


REGIMES = {
    "default(25ep,wd1e-4,tanh)": dict(classes_total=4, n_train=1000, mem_per_class=50, noise=0.5, norm=5.0,
                                       epochs_t0=25, epochs_t1=25, lr=0.05, wd=1e-4, nonneg=False, scale=30.0, margin=0.35),
    "long,no-wd,relu": dict(classes_total=4, n_train=1000, mem_per_class=20, noise=0.6, norm=4.0,
                            epochs_t0=150, epochs_t1=150, lr=0.05, wd=0.0, nonneg=True, scale=30.0, margin=0.35),
    "8classes,long,no-wd,relu": dict(classes_total=8, n_train=1000, mem_per_class=20, noise=0.6, norm=4.0,
                                     epochs_t0=150, epochs_t1=150, lr=0.05, wd=0.0, nonneg=True, scale=30.0, margin=0.35),
    "8classes,150ep,wd,tanh": dict(classes_total=8, n_train=1000, mem_per_class=20, noise=0.6, norm=4.0,
                                   epochs_t0=150, epochs_t1=150, lr=0.05, wd=1e-4, nonneg=False, scale=30.0, margin=0.35),
}

if __name__ == "__main__":
    for rname, reg in REGIMES.items():
        acc = {k: [] for k in ("linear", "cosine", "cosine_margin")}
        norm_info = None
        for seed in range(3):
            r = run_regime(reg, seed)
            for k in acc:
                acc[k].append(r[k]["forget"])
            if norm_info is None:
                lin = r["linear"]
                norm_info = f"lin w-norm old={lin['old_n']:.2f} new={lin['new_n']:.2f}"
        line = "  ".join(f"{k}: {sum(v)/len(v)*100:5.1f}%" for k, v in acc.items())
        print(f"[{rname}] forgetting -> {line}   ({norm_info})")