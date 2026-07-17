"""H7: Does the experiment design confound method rankings?

Hypothesis: The test set uses pre-swap labels while training uses swapped
labels after the shift.  This inflates methods that preserve the OLD mapping
(static_bank, PID-CR alike) and hides the adaptation benefit of PID-CR.

Test: Run all methods under label shift, evaluate on BOTH original and
shifted test labels.  If method ranking changes between the two evaluations,
the confound is confirmed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import math
import random
from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.synthetic import SyntheticDataModule, SyntheticConfig
from src.models import MLPClassifier
from src.bank.strategies import StaticReplayBank, ExposureDebtGhostBank
from src.bank.core.pid_controller import PIDController
from src.methods import StaticBankMethod, PIDGBMethod, EDGBMethod, BaselineMethod
from src.training import GhostBankLightningModule
from studies.output import OutputManager


def create_shifted_test_dataset(test_dataset, swap_classes=(0, 2)):
    """Return a copy of test_dataset with swapped labels."""
    import copy
    ds = copy.deepcopy(test_dataset)
    c1, c2 = swap_classes
    mask1 = ds._ys == c1
    mask2 = ds._ys == c2
    ds._ys[mask1] = c2
    ds._ys[mask2] = c1
    return ds


def evaluate(model, test_loader, device):
    """Evaluate model on a dataloader, return per-class and balanced acc."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    num_classes = int(labels.max().item()) + 1
    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[f"acc_class_{c}"] = (preds[mask] == c).float().mean().item()

    # Balanced accuracy
    class_accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_accs.append((preds[mask] == c).float().mean().item())
    balanced = sum(class_accs) / len(class_accs) if class_accs else 0.0

    # Minority recall (class 2)
    mask2 = labels == 2
    minority = (preds[mask2] == 2).float().mean().item() if mask2.sum() > 0 else 0.0

    return {"balanced_acc": balanced, "minority_recall": minority, **per_class}


def run_method(method_name, seed, swap_classes=(0, 2), shift_epoch=5, max_epochs=10):
    """Run a single method and return metrics on original and shifted test sets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(seed)

    # Data
    data_cfg = SyntheticConfig(seed=seed, imbalance_ratio=100, majority_train=2000,
                                test_per_class=500, batch_size=32)
    dm = SyntheticDataModule(data_cfg)
    dm.setup("fit")

    num_classes = dm.train_dataset.num_classes
    class_counts = dm.train_dataset.class_counts

    # Model
    model = MLPClassifier(input_dim=2, hidden_dim=64, num_classes=num_classes)
    model = model.to(device)

    # Bank
    if method_name == "static_bank":
        bank = StaticReplayBank(num_classes, capacity_per_class=200, seed=seed, exclude_classes=[])
    elif method_name in ("pid_gb", "ed_gb"):
        bank = ExposureDebtGhostBank(num_classes, capacity_per_class=200, seed=seed, exclude_classes=[])
    else:
        bank = None

    # Method
    if method_name == "baseline":
        method = BaselineMethod()
    elif method_name == "static_bank":
        method = StaticBankMethod(retrieval_budget=8, warmup_steps=0)
    elif method_name == "ed_gb":
        method = EDGBMethod(retrieval_budget=8, warmup_steps=0)
    elif method_name == "pid_gb":
        method = PIDGBMethod(retrieval_budget=8, warmup_steps=0, use_class_weights=False)
        method.class_weights = [1.0] * num_classes
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Module
    module = GhostBankLightningModule(
        model=model, method=method, bank=bank,
        learning_rate=0.05, num_classes=num_classes,
    )

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    global_step = 0

    # Training loop
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    orig_test_dataset = dm.test_dataset

    # Get shifted test dataset
    shifted_test_dataset = create_shifted_test_dataset(orig_test_dataset, swap_classes)
    from torch.utils.data import DataLoader
    shifted_test_loader = DataLoader(shifted_test_dataset, batch_size=32, shuffle=False)

    # Epoch loop
    shifted = False
    for epoch in range(max_epochs):
        # Check shift
        if epoch == shift_epoch and not shifted:
            ds = dm.train_dataset
            c1, c2 = swap_classes
            mask1 = ds._ys == c1
            mask2 = ds._ys == c2
            ds._ys[mask1] = c2
            ds._ys[mask2] = c1
            shifted = True

        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch = (x, y)

            # Forward through method
            loss = method.compute_loss(batch, module, bank=bank)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

    # Evaluate on original test set
    orig_metrics = evaluate(model, test_loader, device)
    # Evaluate on shifted test set
    shift_metrics = evaluate(model, shifted_test_loader, device)

    return {
        "method": method_name,
        "seed": seed,
        "orig": orig_metrics,
        "shifted": shift_metrics,
    }


if __name__ == "__main__":
    seeds = [13, 42, 73]
    methods = ["baseline", "static_bank", "ed_gb", "pid_gb"]

    all_results = []
    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method_name}")
        print(f"{'='*60}")
        for seed in tqdm(seeds, desc=f"{method_name}", unit="seed"):
            result = run_method(method_name, seed)
            all_results.append(result)

    # Aggregate and display results
    print("\n\n" + "="*70)
    print("  RESULTS: ORIGINAL test labels (pre-swap)")
    print("="*70)
    print(f"  {'Method':<15} {'Balanced Acc':>15} {'Minority Recall':>17} {'Class 0 Acc':>13} {'Class 2 Acc':>13}")
    print("  " + "-"*73)
    for method_name in methods:
        orig_ball = [r["orig"]["balanced_acc"] for r in all_results if r["method"] == method_name]
        orig_min = [r["orig"]["minority_recall"] for r in all_results if r["method"] == method_name]
        orig_c0 = [r["orig"].get("acc_class_0", 0) for r in all_results if r["method"] == method_name]
        orig_c2 = [r["orig"].get("acc_class_2", 0) for r in all_results if r["method"] == method_name]
        if orig_ball:
            mu = sum(orig_ball) / len(orig_ball)
            sd = (sum((v - mu)**2 for v in orig_ball) / len(orig_ball))**0.5
            mu_min = sum(orig_min) / len(orig_min)
            sd_min = (sum((v - mu_min)**2 for v in orig_min) / len(orig_min))**0.5
            mu_c0 = sum(orig_c0) / len(orig_c0)
            mu_c2 = sum(orig_c2) / len(orig_c2)
            print(f"  {method_name:<15} {mu:>8.4f} +/- {sd:<.4f}  {mu_min:>8.4f} +/- {sd_min:<.4f}  {mu_c0:>8.4f}        {mu_c2:>8.4f}")

    print("\n\n" + "="*70)
    print("  RESULTS: SHIFTED test labels (post-swap)")
    print("="*70)
    print(f"  {'Method':<15} {'Balanced Acc':>15} {'Minority Recall':>17} {'Class 0 Acc':>13} {'Class 2 Acc':>13}")
    print("  " + "-"*73)
    for method_name in methods:
        shift_ball = [r["shifted"]["balanced_acc"] for r in all_results if r["method"] == method_name]
        shift_min = [r["shifted"]["minority_recall"] for r in all_results if r["method"] == method_name]
        shift_c0 = [r["shifted"].get("acc_class_0", 0) for r in all_results if r["method"] == method_name]
        shift_c2 = [r["shifted"].get("acc_class_2", 0) for r in all_results if r["method"] == method_name]
        if shift_ball:
            mu = sum(shift_ball) / len(shift_ball)
            sd = (sum((v - mu)**2 for v in shift_ball) / len(shift_ball))**0.5
            mu_min = sum(shift_min) / len(shift_min)
            sd_min = (sum((v - mu_min)**2 for v in shift_min) / len(shift_min))**0.5
            mu_c0 = sum(shift_c0) / len(shift_c0)
            mu_c2 = sum(shift_c2) / len(shift_c2)
            print(f"  {method_name:<15} {mu:>8.4f} +/- {sd:<.4f}  {mu_min:>8.4f} +/- {sd_min:<.4f}  {mu_c0:>8.4f}        {mu_c2:>8.4f}")

    # Verdict
    print("\n\n" + "="*70)
    print("  VERDICT")
    print("="*70)
    print("  If method ranking changes between original and shifted test labels,")
    print("  the experiment confound is CONFIRMED: the benchmark measures memory")
    print("  retention, not adaptation speed.\n")
    print("  Under shifted labels, we expect the baseline (no replay) to improve")
    print("  because the model trained on swapped labels will correctly predict")
    print("  swapped test labels.  Replay methods should degrade because their")
    print("  stale data reinforces the old mapping.")
