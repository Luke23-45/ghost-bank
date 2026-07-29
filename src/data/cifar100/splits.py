from __future__ import annotations

import torch


def create_class_wise_splits(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    probe_size: int = 30,
    val_size: int = 20,
    seed: int = 13,
) -> dict[str, torch.Tensor]:
    """Partition CIFAR-100 training images into probe/val/train per class.

    For each class ``c`` in ``range(num_classes)``:

        - ``probe``: ``probe_size`` images, never replayed or used for SGD,
        - ``validation``: ``val_size`` images, used only for tuning,
        - ``train``: the remaining images, used for SGD.

    The partition is deterministic given ``seed``.  The returned dict
    has keys ``"probe_images"``, ``"probe_targets"``, ``"val_images"``,
    ``"val_targets"``, ``"train_images"``, ``"train_targets"``, each
    a ``torch.Tensor``.

    Raises
    ------
    ValueError
        If any class has fewer than ``probe_size + val_size + 1`` images.
    """
    all_probe_imgs: list[torch.Tensor] = []
    all_probe_tgts: list[torch.Tensor] = []
    all_val_imgs: list[torch.Tensor] = []
    all_val_tgts: list[torch.Tensor] = []
    all_train_imgs: list[torch.Tensor] = []
    all_train_tgts: list[torch.Tensor] = []

    rng = torch.Generator().manual_seed(seed)

    for c in range(num_classes):
        mask = targets == c
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        n_total = idx.shape[0]
        needed = probe_size + val_size + 1
        if n_total < needed:
            raise ValueError(
                f"Class {c} has {n_total} images, but at least {needed} are "
                f"required (probe={probe_size}, val={val_size}, train>=1)."
            )

        perm = idx[torch.randperm(n_total, generator=rng)]

        probe_idx = perm[:probe_size]
        val_idx = perm[probe_size: probe_size + val_size]
        train_idx = perm[probe_size + val_size:]

        all_probe_imgs.append(images[probe_idx])
        all_probe_tgts.append(targets[probe_idx])
        all_val_imgs.append(images[val_idx])
        all_val_tgts.append(targets[val_idx])
        all_train_imgs.append(images[train_idx])
        all_train_tgts.append(targets[train_idx])

    return {
        "probe_images": torch.cat(all_probe_imgs, dim=0),
        "probe_targets": torch.cat(all_probe_tgts, dim=0),
        "val_images": torch.cat(all_val_imgs, dim=0),
        "val_targets": torch.cat(all_val_tgts, dim=0),
        "train_images": torch.cat(all_train_imgs, dim=0),
        "train_targets": torch.cat(all_train_tgts, dim=0),
    }


def create_probe_loader_per_class(
    probe_images: torch.Tensor,
    probe_targets: torch.Tensor,
    num_classes: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return a dict ``{class_id: (images, targets)}`` for probe evaluation.

    Each entry contains all probe images for that class as
    ``(N, H, W, C)`` uint8 NHWC tensors ready for feature extraction.
    """
    loaders: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for c in range(num_classes):
        mask = probe_targets == c
        loaders[c] = (probe_images[mask], probe_targets[mask])
    return loaders
