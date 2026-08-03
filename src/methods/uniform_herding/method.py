from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext
from src.methods.nme import nme_predict
from src.methods.static_bank.method import _augment_replay


class UniformHerdingMethod(Method):
    """Replay method that uses a fixed-total herding bank.

    Predictions use the Nearest Mean Exemplar rule (the mean feature of the
    selected exemplars per class), which pairs naturally with the cosine
    margin head: both operate on L2-normalized prototypes in feature space.

    Optional knowledge distillation (``kd_weight > 0``): at the start of each
    task a frozen snapshot of the model is cached before the head is expanded,
    and a cross-entropy + KL(in Hinton's T-softened form) loss on the old
    classes is used so that old-class scores are anchored to the teacher.
    This mirrors the iCaRL KD teacher handling (device, dtype, autocast), but
    keeps cross-entropy classification instead of iCaRL's per-class BCE.
    """

    def __init__(
        self,
        retrieval_budget: int = 64,
        warmup_steps: int = 0,
        kd_weight: float = 0.0,
        kd_temperature: float = 2.0,
        predict_mode: str = "nme",
    ) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.predict_mode = predict_mode
        if predict_mode not in ("nme", "head"):
            raise ValueError(
                f"predict_mode must be 'nme' or 'head', got {predict_mode!r}"
            )
        self.old_model = None

    def on_task_start(self, model: nn.Module, task_id: int) -> None:
        """Cache the model before the head is expanded for KD distillation."""
        if task_id > 0 and self.kd_weight > 0.0:
            self.old_model = copy.deepcopy(model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        x, y = batch
        if bank is None:
            return F.cross_entropy(pl_module(x, targets=y), y)

        if context is not None and context.raw_x is not None and context.raw_y is not None:
            examples = list(zip(context.raw_x, context.raw_y.tolist()))
            bank.store(examples, raw_indices=context.raw_indices)
        else:
            x_cpu = x.detach().cpu()
            y_labels = y.detach().cpu().tolist()
            bank.store([(x_i.clone(), y_i) for x_i, y_i in zip(x_cpu, y_labels)])

        if pl_module.global_step >= self.warmup_steps:
            replay_items = bank.query(budget=self.retrieval_budget)
            replay_x = _augment_replay(
                replay_items,
                transform=context.train_transform if context is not None else None,
                rng=context.augment_rng if context is not None else None,
                device=y.device,
            )
            replay_y = (
                torch.tensor(
                    [int(item[1]) for item in replay_items],
                    device=y.device,
                    dtype=torch.long,
                )
                if replay_items
                else None
            )

            if replay_x is not None and replay_y is not None and replay_y.numel() > 0:
                x = torch.cat([x, replay_x], dim=0)
                y = torch.cat([y, replay_y], dim=0)

        logits = pl_module(x, targets=y)
        loss = F.cross_entropy(logits, y)

        if self.old_model is not None:
            old_fc = self.old_model.fc
            num_old_classes = getattr(
                old_fc, "out_features", getattr(old_fc, "num_classes", 0)
            )
            if num_old_classes > 0:
                self.old_model.to(x.device)
                with torch.no_grad():
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        old_dtype = next(self.old_model.parameters()).dtype
                        old_logits = self.old_model(x.to(dtype=old_dtype)).to(dtype=x.dtype)

                t = self.kd_temperature
                kd_loss = F.kl_div(
                    F.log_softmax(logits[:, :num_old_classes].float() / t, dim=1),
                    F.softmax(old_logits.float() / t, dim=1),
                    reduction="batchmean",
                ) * (t * t)
                loss = loss.float() + self.kd_weight * kd_loss

        return loss

    def predict(self, x: torch.Tensor, pl_module, bank: AbstractGhostBank | None = None) -> torch.Tensor:
        """Classify using the configured policy.

        ``nme``: Nearest Mean Exemplar rule over the bank's selected exemplars.
        ``head``: argmax over the head logits (the cosine margin classifier).
        """
        if self.predict_mode == "head":
            return pl_module.model(x).argmax(dim=-1)
        return nme_predict(x, pl_module, bank=bank)
