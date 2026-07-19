from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext
from src.utils.logging import get_logger
from src.methods.static_bank.method import _augment_replay

LOGGER = get_logger(__name__)


class EDGBMethod(Method):
    """Exposure-Debt replay method.

    Uses the :class:`ExposureTracker` (number of mini-batches each
    class has been seen in so far over the entire experiment) as the
    ground-truth exposure signal.  Debt per class is computed as
    ``max(0, target − accumulated)`` where ``target`` is the **expected**
    per-class sample count given the cumulative global step count:

        target_c = total_steps_seen * batch_size / num_total_classes

    Replay budget is split proportionally to per-class debt.  Storage
    uses the **raw** pre-augmentation image (uint8 NHWC) per batch
    sample and re-augments each retrieved item at query time, matching
    the standard replay convention of :class:`StaticBankMethod`.
    """

    needs_exposure_tracker = True

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
        tracker = getattr(pl_module, "exposure_tracker", None)

        if tracker is not None:
            for label in y:
                tracker.record(label)
        elif getattr(self, "_warned_missing_tracker", False) is False:
            LOGGER.warning(
                "EDGBMethod: no exposure_tracker found on pl_module. "
                "Exposure tracking and debt-based retrieval disabled."
            )
            self._warned_missing_tracker = True

        if bank is None:
            return F.cross_entropy(pl_module(x), y)

        # Store RAW images.
        if context is not None and context.raw_x is not None and context.raw_y is not None:
            examples = list(zip(context.raw_x, context.raw_y.tolist()))
            bank.store(examples)
        else:
            bank.store([(x[i], y[i]) for i in range(len(y))])

        if pl_module.global_step < self.warmup_steps or tracker is None:
            return F.cross_entropy(pl_module(x), y)

        acc = tracker.accumulated()
        target_per_class = _uniform_target_per_class(pl_module, acc, x.size(0))

        replay_items = bank.query(
            budget=self.retrieval_budget,
            exposure=acc,
            target_per_class=target_per_class,
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
            # Record that the replay classes were exposed so exposure
            # accounting remains consistent with the augmented samples.
            for label in replay_y:
                tracker.record(label.item())
            x = torch.cat([x, replay_x], dim=0)
            y = torch.cat([y, replay_y], dim=0)

        return F.cross_entropy(pl_module(x), y)


def _uniform_target_per_class(pl_module, acc: list[int], batch_size: int) -> list[float]:
    """Compute the per-class exposure target across the full experiment run.

    The exposure target represents the **expected** per-class sample count
    if sampling were perfectly uniform across the *entire* run (i.e.
    classical ER under an idealized uniform-data assumption).  For
    class-IL with ``num_classes`` expanding across tasks, we use the
    current ``num_classes`` known to the model (``pl_module.num_classes``)
    because the ExposureTracker is bound to that cardinality.

    Falls back to a per-step estimate using only ``global_step`` if
    ``pl_module.num_classes`` is unavailable, which still gives a
    reasonable ratio for the within-task view.
    """
    num_classes = getattr(pl_module, "num_classes", None) or len(acc)
    if num_classes <= 0:
        return list(acc)
    if pl_module.global_step > 0 and batch_size > 0:
        target = float(pl_module.global_step * batch_size) / float(num_classes)
    else:
        target = float(sum(acc)) / float(num_classes) if num_classes > 0 else 0.0
    return [target] * num_classes
