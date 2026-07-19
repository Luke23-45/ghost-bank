from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext


class StaticBankMethod(Method):
    """Uniformly-random replay baseline (Experience Replay / BiR).

    Storage stores the **raw, pre-augmentation** image plus its label —
    i.e. ``(raw_image_uint8, label)`` — instead of the augmented view
    produced by the dataset.  On retrieval, each replay item is freshly
    augmented via :attr:`MethodContext.train_transform` seeded by
    :attr:`MethodContext.augment_rng` so the augmentation does not
    interfere with the main trainer RNG and is re-rolled each call.

    Retrieval is uniformly random across non-empty per-class pools,
    with sample size capped by ``retrieval_budget``.  The budget is
    converted to an exact count via Python's ``random.choices`` (with
    replacement) — same construction as BiR/ER.
    """

    def __init__(
        self,
        retrieval_budget: int = 64,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch
        if bank is None:
            return F.cross_entropy(pl_module(x), y)

        # Store RAW images (uint8 NHWC) for replay.
        if context is not None and context.raw_x is not None and context.raw_y is not None:
            examples = list(zip(context.raw_x, context.raw_y.tolist()))
            bank.store(examples)
        else:
            bank.store([(x[i], y[i]) for i in range(len(y))])

        # Replay (only after warmup).
        if (
            pl_module.global_step >= self.warmup_steps
        ):
            replay_items = bank.query(budget=self.retrieval_budget)
            replay_x = _augment_replay(
                replay_items,
                transform=context.train_transform if context is not None else None,
                rng=context.augment_rng if context is not None else None,
                device=y.device,
            )
            replay_y = torch.tensor(
                [int(item[1]) for item in replay_items],
                device=y.device,
                dtype=torch.long,
            ) if replay_items else None

            if replay_x is not None and replay_y is not None and replay_y.numel() > 0:
                x = torch.cat([x, replay_x], dim=0)
                y = torch.cat([y, replay_y], dim=0)

        return F.cross_entropy(pl_module(x), y)


def _augment_replay(
    replay_items: list,
    transform: object | None,
    rng: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Apply ``transform`` per-item and stack into ``[B, C, H, W]``.

    Each item is expected to be ``(raw_image, label)`` where the image is
    NHWC uint8 ``[32, 32, 3]``.  The transform is applied CPU-side per
    item to keep random-crop / random-flip RNG isolated from the trainer
    global RNG; the resulting tensor is then moved to ``device``.

    Falls back to the input tensor when ``transform`` is ``None``
    (e.g. unit tests, or when RAW is unavailable and the input is
    already post-transform).
    """
    if not replay_items:
        return None

    if transform is None:
        from_stack = []
        for item in replay_items:
            raw = item[0]
            from_stack.append(raw if torch.is_tensor(raw) else torch.as_tensor(raw))
        return torch.stack(from_stack, dim=0).to(device)

    augmented = []
    for item in replay_items:
        raw = item[0]
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        # Permute to NCHW uint8 if needed and apply transform.
        if raw.dim() == 3 and raw.shape[-1] == 3 and raw.shape[0] != 3:
            tensor_for_transform = raw.permute(2, 0, 1).contiguous()
        else:
            tensor_for_transform = raw
        augmented.append(transform(tensor_for_transform))
    return torch.stack(augmented, dim=0).to(device)
