from __future__ import annotations

from collections.abc import Sequence

import torch


def balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    per_class = torch.zeros(num_classes, device=y_true.device)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class[c] = (y_pred[mask] == c).float().mean()
    return per_class.mean()


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    f1s = torch.zeros(num_classes, device=y_true.device)
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1s[c] = 2 * precision * recall / (precision + recall + 1e-12)
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
