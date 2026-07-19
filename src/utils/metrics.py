from __future__ import annotations

from collections.abc import Sequence

import torch


def balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Build confusion matrix in one pass: cm[true, pred]
    cm = torch.zeros(num_classes, num_classes, device=y_true.device)
    cm.index_put_(
        (y_true.long(), y_pred.long()),
        torch.ones(y_true.shape[0], device=y_true.device),
        accumulate=True,
    )
    support = cm.sum(dim=1)  # per-class true count
    per_class_acc = torch.zeros(num_classes, device=y_true.device)
    valid = support > 0
    per_class_acc[valid] = cm.diag()[valid] / support[valid]
    return per_class_acc.mean()


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Build confusion matrix in one pass
    cm = torch.zeros(num_classes, num_classes, device=y_true.device)
    cm.index_put_(
        (y_true.long(), y_pred.long()),
        torch.ones(y_true.shape[0], device=y_true.device),
        accumulate=True,
    )
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp  # predicted as c but not c
    fn = cm.sum(dim=1) - tp  # is c but not predicted as c
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    return f1s.mean()


def minority_recall(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    minority_classes: Sequence[int] | None = None,
) -> torch.Tensor:
    if minority_classes is not None:
        recalls = []
        for c in minority_classes:
            mask = y_true == c
            if mask.sum() > 0:
                recalls.append((y_pred[mask] == c).float().mean())
        if not recalls:
            return torch.tensor(0.0, device=y_true.device)
        return torch.stack(recalls).mean()
    return balanced_accuracy(y_true, y_pred, max(y_true.max().item(), y_pred.max().item()) + 1)
