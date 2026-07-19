from __future__ import annotations

from typing import Sequence

import torch
import torchvision.transforms as T


def make_train_transform(
    mean: Sequence[float],
    std: Sequence[float],
) -> T.Compose:
    """Standard CIFAR-100 train transform.

    Pipeline:
        1. RandomCrop(32, padding=4, pad_if_needed=False)
        2. RandomHorizontalFlip(p=0.5)
        3. ConvertImageDtype(torch.float32)
        4. Normalize(mean, std)

    Assumes input is uint8 with shape ``[3, 32, 32]`` (NCHW).
    """
    return T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def make_eval_transform(
    mean: Sequence[float],
    std: Sequence[float],
) -> T.Compose:
    """CIFAR-100 test-time transform (no augmentation, just normalize).

    Assumes input is uint8 with shape ``[3, 32, 32]`` (NCHW).
    """
    return T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def make_train_transform_from_rng(
    mean: Sequence[float],
    std: Sequence[float],
    rng: torch.Generator | None = None,
) -> "T.Compose":
    """Augmentation transform that uses a *caller-supplied* RNG.

    Equivalent to :func:`make_train_transform` for identical inputs,
    but the random crop offset and horizontal-flip decision are drawn
    from ``rng`` instead of PyTorch's global default generator.  This
    lets the bank replay path augment without disturbing the main
    training RNG state.

    The implementation is a thin wrapper that calls into the same
    operations but seeds them manually.  We patch torchvision by using
    a directly constructed Generator.
    """

    def _random_crop(x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        pad_h = h + 4
        pad_w = w + 4
        padded = torch.zeros(
            (x.shape[-3], pad_h, pad_w), dtype=x.dtype, device=x.device,
        )
        padded[..., 4:4 + h, 4:4 + w] = x

        if rng is not None:
            top = int(torch.randint(0, 9, (1,), generator=rng).item())
            left = int(torch.randint(0, 9, (1,), generator=rng).item())
        else:
            top = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
        return padded[..., top:top + h, left:left + w]

    def _random_hflip(x: torch.Tensor) -> torch.Tensor:
        if rng is not None:
            flip = bool(torch.rand(1, generator=rng).item() < 0.5)
        else:
            flip = bool(torch.rand(1).item() < 0.5)
        return torch.flip(x, dims=[-1]) if flip else x

    return T.Compose(
        [
            T.Lambda(_random_crop),
            T.Lambda(_random_hflip),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )
