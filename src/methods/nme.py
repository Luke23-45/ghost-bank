"""Nearest Mean Exemplar (NME) classification, shared by replay methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank


def nme_predict(
    x: torch.Tensor,
    pl_module,
    bank: AbstractGhostBank | None = None,
) -> torch.Tensor:
    """Classify by nearest L2-normalized class mean in feature space.

    iCaRL protocol: the backbone acts as a frozen feature extractor, both the
    query features and the stored class means are L2-normalized, and the
    prediction is the class of the closest mean.  Class means come from
    ``bank.class_means`` (the mean feature of the *selected* exemplars,
    computed at rebuild time).  Falls back to the head's argmax when no
    class means are available.
    """
    if bank is None or not hasattr(bank, "class_means") or not bank.class_means:
        return pl_module.model(x).argmax(dim=-1)

    class_ids, means = zip(*sorted(bank.class_means.items()))
    means_tensor = torch.stack(means).to(x.device)  # [num_classes, feature_dim]
    class_ids_tensor = torch.as_tensor(class_ids, device=x.device)

    features = pl_module.model.extract_features(x)  # [batch_size, feature_dim]
    features = F.normalize(features, p=2, dim=1)
    means_tensor = F.normalize(means_tensor, p=2, dim=1)

    nearest = torch.cdist(features, means_tensor).argmin(dim=1)  # [batch_size]
    return class_ids_tensor[nearest]
