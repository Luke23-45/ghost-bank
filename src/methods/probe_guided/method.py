from __future__ import annotations

import copy
import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.bank.core.allocator import allocate_fixed_total, allocate_uniform_fixed_total
from src.bank.core.base import AbstractGhostBank
from src.bank.core.retrieval import sample_by_quota
from src.methods.base import Method, MethodContext
from src.methods.static_bank.method import _augment_replay
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ProbeGuidedMethod(Method):
    """Probe-Guided Replay with Pre-Trained Backbone.

    The method maintains a fixed-total memory bank ``M``.  At the end
    of each task, the runner computes held-out probe scores for all
    seen classes.  The method converts these scores to a per-class
    memory allocation and prunes the bank to match.

    Training loss is a mixture of:
    - current-task cross-entropy,
    - replayed old-class cross-entropy (sampled uniformly from stored
      exemplars, which already reflect the probe-driven allocation),
    - optional distillation against the previous model snapshot.

    The research claim tested by this method is:
        A clean forgetting probe, measured on held-out exemplars and
        coupled to a scarcity-aware replay policy, improves CIL under
        fixed memory.
    """

    def __init__(
        self,
        seed: int = 13,
        retrieval_budget: int = 64,
        warmup_steps: int = 0,
        memory_total: int = 2000,
        memory_floor: int = 1,
        gamma: float = 0.5,
        beta: float = 1.0,
        probe_smoothing: float = 0.0,
        distillation_weight: float = 0.0,
        calibrate: bool = False,
        calibration_lr: float = 0.01,
        calibration_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps
        self.memory_total = memory_total
        self.memory_floor = memory_floor
        self.gamma = gamma
        self.beta = beta
        self.probe_smoothing = probe_smoothing
        self.distillation_weight = distillation_weight
        self.calibrate = calibrate
        self.calibration_lr = calibration_lr
        self.calibration_epochs = calibration_epochs

        self._prev_model_snapshot: torch.nn.Module | None = None
        self._current_allocation: list[int] | None = None
        self._exemplar_bank: dict[int, list] = {}
        self._replay_class_count: int = 0
        self._rng = random.Random(seed)
        self.allocation_history: list[list[int]] = []

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch

        if context is not None and context.raw_x is not None and context.raw_y is not None:
            examples = list(zip(context.raw_x, context.raw_y.tolist()))
            self._store_exemplars(examples)
        else:
            self._store_exemplars([(x[i], y[i]) for i in range(len(y))])

        loss = F.cross_entropy(pl_module(x), y)

        if pl_module.global_step >= self.warmup_steps:
            replay_loss = self._compute_replay_loss(pl_module, y.device)
            loss = loss + replay_loss

        if self.distillation_weight > 0.0 and self._prev_model_snapshot is not None:
            distill_loss = self._compute_distillation_loss(pl_module, x)
            loss = loss + self.distillation_weight * distill_loss

        return loss

    def _store_exemplars(self, examples: list) -> None:
        """Store raw exemplars in the per-class bank."""
        for img, label in examples:
            cid = int(label) if torch.is_tensor(label) else int(label)
            if cid not in self._exemplar_bank:
                self._exemplar_bank[cid] = []
            self._exemplar_bank[cid].append((img, label))

    def _compute_replay_loss(
        self,
        pl_module,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample replay items from the exemplar bank and compute CE loss."""
        if (
            not self._exemplar_bank
            or self.retrieval_budget <= 0
            or self._replay_class_count <= 0
        ):
            return torch.tensor(0.0, device=device)

        num_classes = self._replay_class_count
        if num_classes <= 0:
            return torch.tensor(0.0, device=device)

        allocation = self._current_allocation
        if allocation is not None and len(allocation) == num_classes and sum(allocation) > 0:
            alloc_sum = sum(allocation)
            frac = [a * self.retrieval_budget / alloc_sum for a in allocation]
            raw = [int(f) for f in frac]
            remainders = sorted(
                [(f - int(f), i) for i, f in enumerate(frac)],
                key=lambda x: x[0], reverse=True,
            )
            for j in range(self.retrieval_budget - sum(raw)):
                raw[remainders[j][1]] += 1
            allocation = raw
        else:
            allocation = allocate_uniform_fixed_total(
                num_classes=num_classes,
                total_budget=self.retrieval_budget,
                floor=self.memory_floor,
            )

        replay_bank = {
            cid: self._exemplar_bank.get(cid, [])
            for cid in range(num_classes)
        }

        replay_items = sample_by_quota(
            replay_bank,
            allocation,
            rng=self._rng,
        )

        replay_x = _augment_replay(
            replay_items,
            transform=None,
            rng=None,
            device=device,
        )
        replay_y = torch.tensor(
            [int(item[1]) for item in replay_items],
            device=device,
            dtype=torch.long,
        ) if replay_items else None

        if replay_x is not None and replay_y is not None and replay_y.numel() > 0:
            logits = pl_module(replay_x)
            return F.cross_entropy(logits, replay_y)

        return torch.tensor(0.0, device=device)

    def _compute_distillation_loss(
        self,
        pl_module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Logit distillation from the previous model snapshot."""
        if self._prev_model_snapshot is None:
            return torch.tensor(0.0, device=x.device)

        with torch.no_grad():
            prev_logits = self._prev_model_snapshot(x)

        curr_logits = pl_module(x)
        prev_probs = F.softmax(prev_logits / 2.0, dim=-1)
        curr_log_probs = F.log_softmax(curr_logits / 2.0, dim=-1)
        loss = F.kl_div(curr_log_probs, prev_probs, reduction="batchmean")
        return loss

    def update_allocation(self, probe_scores: list[float] | None) -> None:
        """Update per-class memory allocation from probe scores.

        Called by the runner at task boundaries.

        Parameters
        ----------
        probe_scores : list[float] or None
            Probe-derived importance scores for all seen classes.
            When None, allocation is uniform.
        """
        num_classes = self._replay_class_count
        if num_classes <= 0:
            num_classes = max(self._exemplar_bank.keys()) + 1 if self._exemplar_bank else 0

        if probe_scores is not None:
            padded_scores = [0.0] * num_classes
            for c in range(min(len(probe_scores), num_classes)):
                padded_scores[c] = probe_scores[c]
        else:
            padded_scores = None

        self._current_allocation = allocate_fixed_total(
            num_classes=num_classes,
            total_budget=self.memory_total,
            probe_scores=padded_scores,
            gamma=self.gamma,
            beta=self.beta,
            floor=self.memory_floor,
        )
        self.allocation_history.append(list(self._current_allocation))

    def prune_memory(self) -> None:
        """Prune the exemplar bank to match the current allocation.

        Keeps only the most recently stored exemplars for each class
        up to its allocated quota.
        """
        if self._current_allocation is None:
            return

        for cid in list(self._exemplar_bank.keys()):
            pool = self._exemplar_bank[cid]
            quota = self._current_allocation[cid] if cid < len(self._current_allocation) else 0
            if quota <= 0:
                del self._exemplar_bank[cid]
            elif len(pool) > quota:
                self._exemplar_bank[cid] = pool[-quota:]

    def snapshot_model(self, pl_module) -> None:
        """Save a copy of the current model for distillation."""
        self._prev_model_snapshot = copy.deepcopy(pl_module.model).eval()
        for param in self._prev_model_snapshot.parameters():
            param.requires_grad = False

    def set_replay_class_count(self, num_classes: int) -> None:
        """Set the number of old classes eligible for replay."""
        self._replay_class_count = max(0, int(num_classes))

    def state_dict(self) -> dict:
        return {
            "allocation": list(self._current_allocation) if self._current_allocation else None,
            "replay_class_count": self._replay_class_count,
            "exemplar_bank": {
                c: list(pool) for c, pool in self._exemplar_bank.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("allocation") is not None:
            self._current_allocation = list(state["allocation"])
        self._replay_class_count = int(state.get("replay_class_count", 0))
        self._exemplar_bank = {
            int(c): list(pool) for c, pool in state.get("exemplar_bank", {}).items()
        }
