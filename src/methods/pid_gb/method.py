from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext
from src.utils.logging import get_logger
from src.methods.static_bank.method import _augment_replay

LOGGER = get_logger(__name__)


class PIDGBMethod(Method):
    """PID-Controlled Replay method.

    Uses per-class training loss as the error signal for a PID
    controller.  When a class is absent from the current batch, a
    small uniform sample from the replay buffer is evaluated with the
    model (no grad) — this is the *bank probe*.  Each probe item is
    converted to model-ready NCHW float32 via the eval transform so
    the probe sees the same input distribution as the live batch (modulo
    augmentation).

    Replay budget is allocated proportionally to PID debt.  Storage uses
    raw uint8 NHWC tensors (same convention as
    :class:`StaticBankMethod`); retrieval re-augments each replay item
    with the train transform seeded by the bank's RNG.
    """

    needs_pid_controller = True

    needs_class_counts = True

    def __init__(
        self,
        retrieval_budget: int = 64,
        warmup_steps: int = 0,
        K_p: float = 1.0,
        K_i: float = 0.1,
        K_d: float = 0.5,
        pid_decay: float = 0.99,
        pid_smooth: float = 0.9,
        temperature: float = 1.0,
        bank_probe_size: int = 16,
        eval_absent_classes: bool = True,
        use_class_weights: bool = False,
        debt_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.pid_decay = pid_decay
        self.pid_smooth = pid_smooth
        self.temperature = temperature
        self.bank_probe_size = bank_probe_size
        self.eval_absent_classes = eval_absent_classes
        self.use_class_weights = use_class_weights
        self.debt_clip = debt_clip

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch
        pid = getattr(pl_module, "pid_controller", None)

        # --- 1. Per-class loss on current batch (no grad) for the PID
        # signal.  We do this forward pass with no_grad for the PID
        # signal alone; the *training* forward pass below carries the
        # grad enabled (so it can update the model).
        debt = None
        if pid is not None:
            with torch.no_grad():
                logits = pl_module(x)
            per_class_loss = self._compute_per_class_loss(
                logits, y, pid.num_classes, pl_module, bank,
                context=context,
            )
            debt = pid.update(per_class_loss)
            if self.debt_clip is not None and self.debt_clip > 0:
                debt = [min(d, self.debt_clip) for d in debt]

        # --- 2. Store and (optionally) replay from bank ---
        if bank is None:
            return F.cross_entropy(pl_module(x), y)

        if context is not None and context.raw_x is not None and context.raw_y is not None:
            examples = list(zip(context.raw_x, context.raw_y.tolist()))
            bank.store(examples)
        else:
            bank.store([(x[i], y[i]) for i in range(len(y))])

        if pl_module.global_step < self.warmup_steps or debt is None:
            return F.cross_entropy(pl_module(x), y)

        replay_items = bank.query(
            budget=self.retrieval_budget,
            debt=debt,
            temperature=self.temperature,
        )
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

    def _compute_per_class_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        pl_module=None,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> list[float | None]:
        """Compute per-class training loss over the live batch.

        For classes absent from the batch, optionally evaluate a small
        sample from the replay buffer through the model.  The probe
        uses the **standardized NCHW float32 input distribution** that
        the model's forward expects.
        """
        losses: list[float | None] = []
        device = logits.device

        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], targets[mask], reduction="mean")
                losses.append(loss.item())
                continue

            if (
                self.eval_absent_classes
                and pl_module is not None
                and bank is not None
                and hasattr(bank, "_bank")
                and c in bank._bank
                and len(bank._bank[c]) > 0
            ):
                items = bank._bank[c]
                n = min(self.bank_probe_size, len(items))
                sampled = random.sample(items, n)
                bx, by = self._assemble_probe_batch(sampled, context=context, device=device)
                if bx is None:
                    losses.append(None)
                    continue
                with torch.no_grad():
                    bank_logits = pl_module(bx)
                loss = F.cross_entropy(bank_logits, by, reduction="mean")
                losses.append(loss.item())
            else:
                losses.append(None)

        return losses

    @staticmethod
    def _assemble_probe_batch(
        sampled: list,
        context: MethodContext | None,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Convert a list of ``(raw_image, label)`` samples into a
        stacked ``[B, 3, 32, 32]`` float32 batch moved to ``device``.

        The raw_image may be either:
        * uint8 NHWC ``[32, 32, 3]`` (the standard raw store path), or
        * already-normalized NCHW float32 (legacy storage path).

        When a ``MethodContext.train_transform`` is provided, we apply it
        after NHWC→NCHW permutation.  Otherwise we just normalize via a
        ship-as-is pass with the same dtype conversion.  This keeps the
        probe on the same input scale as the live batch.
        """
        if not sampled:
            return None, None

        by = torch.tensor(
            [int(item[1]) for item in sampled],
            device=device,
            dtype=torch.long,
        )

        try:
            tensors = []
            for item in sampled:
                raw = item[0]
                if not torch.is_tensor(raw):
                    raw = torch.as_tensor(raw)
                if raw.dim() == 3 and raw.shape[-1] == 3 and raw.shape[0] != 3:
                    tensor = raw.permute(2, 0, 1).contiguous()
                else:
                    tensor = raw

                if context is not None and context.train_transform is not None:
                    tensor = context.train_transform(tensor)
                else:
                    tensor = tensor.float() / 255.0
                tensors.append(tensor)

            bx = torch.stack(tensors, dim=0).to(device)
            return bx, by
        except Exception:
            return None, None
