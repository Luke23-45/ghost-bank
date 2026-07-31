from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext
from src.methods.static_bank.method import _augment_replay

class iCaRLMethod(Method):
    """Incremental Classifier and Representation Learning (iCaRL) baseline.
    
    Uses BCE distillation for old classes and BCE classification for new classes,
    paired with a Nearest Mean Exemplar (NME) rule during testing.
    """
    
    def __init__(self, retrieval_budget: int = 64, warmup_steps: int = 0) -> None:
        super().__init__()
        self.retrieval_budget = retrieval_budget
        self.warmup_steps = warmup_steps
        self.old_model = None

    def on_task_start(self, model: nn.Module, task_id: int) -> None:
        """Cache the model before the head is expanded for KD distillation."""
        if task_id > 0:
            self.old_model = copy.deepcopy(model)
            self.old_model.eval()
            # Freeze old model
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
        batch_size = x.size(0)

        if bank is not None:
            # Store RAW images (uint8 NHWC) for replay
            if context is not None and context.raw_x is not None and context.raw_y is not None:
                examples = list(zip(context.raw_x, context.raw_y.tolist()))
                bank.store(examples, raw_indices=context.raw_indices)
            else:
                x_cpu = x.detach().cpu()
                y_labels = y.detach().cpu().tolist()
                bank.store([(x_i.clone(), y_i) for x_i, y_i in zip(x_cpu, y_labels)])

            # Retrieve and augment exemplars
            if pl_module.global_step >= self.warmup_steps:
                replay_items = bank.query(self.retrieval_budget)
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

        logits = pl_module(x)
        num_classes = logits.size(1)
        
        # One-hot targets for BCE
        targets = F.one_hot(y, num_classes=num_classes).float()
        
        # Apply KD for old classes using the frozen snapshot
        if self.old_model is not None:
            num_old_classes = self.old_model.fc.out_features
            
            # Ensure the teacher model is on the correct device (GPU) 
            # since copy.deepcopy might have captured it on the CPU.
            self.old_model.to(x.device)
            
            with torch.no_grad():
                # Disable autocast for the teacher to prevent float16/float32 mismatch bugs
                # and ensure we get the exact FP32 probabilities it originally produced.
                with torch.autocast(device_type=x.device.type, enabled=False):
                    old_dtype = next(self.old_model.parameters()).dtype
                    old_logits = self.old_model(x.to(dtype=old_dtype))
                    old_targets = torch.sigmoid(old_logits).to(dtype=x.dtype)
            
            # Replace the targets for old classes with the old model's probabilities
            targets[:, :num_old_classes] = old_targets
            
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = loss.sum(dim=1).mean(dim=0)
        return loss

    def predict(self, x: torch.Tensor, pl_module, bank: AbstractGhostBank | None = None) -> torch.Tensor:
        """Nearest Mean Exemplar (NME) Classification."""
        if bank is None or not hasattr(bank, "class_means") or not bank.class_means:
            # Fallback to linear fc if no class means are available
            return pl_module.model(x).argmax(dim=-1)

        # Ensure class means are a single tensor on the correct device
        means_list = []
        class_indices = []
        for cid, mean_t in sorted(bank.class_means.items()):
            means_list.append(mean_t)
            class_indices.append(cid)
            
        means_tensor = torch.stack(means_list).to(x.device) # [num_classes, feature_dim]
        
        # Extract features
        features = pl_module.model.extract_features(x) # [batch_size, feature_dim]
        
        # Normalize features like original iCaRL
        features = F.normalize(features, p=2, dim=1)
        means_tensor = F.normalize(means_tensor, p=2, dim=1)
        
        # Compute distances (L2)
        dists = torch.cdist(features, means_tensor) # [batch_size, num_classes]
        
        # Find nearest
        nearest_idx = dists.argmin(dim=1)
        
        # Map back to class indices just in case they aren't perfectly sequential
        preds = torch.tensor([class_indices[i] for i in nearest_idx], device=x.device)
        return preds
