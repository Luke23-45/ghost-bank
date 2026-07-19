from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable

import torch

from src.bank.core.base import AbstractGhostBank


class MethodContext:
    """Optional context passed alongside the (x, y) batch to ``Method.compute_loss``.

    Carries the side-channel information needed for standard replay-style
    methods: the raw pre-augmentation image tensor of the current batch
    (looked up by the PL module from the live task dataset), the view
    indices used to pull them, the train-time augmentation pipeline,
    and a ``torch.Generator`` so that bank retrieval can apply fresh
    augmentation without disturbing the trainer's RNG state.

    Methods store ``(raw_images, labels)`` populated from :attr:`raw_x`
    and :attr:`raw_y` and re-augment each replay item on retrieval via
    :attr:`train_transform` seeded with :attr:`augment_rng`.  When the
    context is ``None`` (e.g. unit tests), methods fall back to using
    the transformed batch directly.
    """

    def __init__(
        self,
        raw_x: torch.Tensor | None = None,
        raw_y: torch.Tensor | None = None,
        raw_indices: torch.Tensor | None = None,
        train_transform: Callable | None = None,
        augment_rng: torch.Generator | None = None,
    ) -> None:
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.raw_indices = raw_indices
        self.train_transform = train_transform
        self.augment_rng = augment_rng


class Method(ABC):
    @abstractmethod
    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        ...
