from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bank.core.base import AbstractGhostBank
from src.methods.base import Method, MethodContext


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

        if bank is not None and pl_module.global_step >= self.warmup_steps:
            replays = bank.query(self.retrieval_budget)
            if replays:
                rx, ry = zip(*replays)
                if context and context.train_transform and context.augment_rng:
                    rx_t = torch.stack(rx, dim=0)
                    rx_t = rx_t.float().div(255.0)
                    # Use standard uniform random state for train_transform during training
                    rx_t = context.train_transform(rx_t)
                else:
                    rx_t = torch.stack(rx, dim=0)
                
                ry_t = torch.stack(ry, dim=0).to(x.device)
                rx_t = rx_t.to(x.device)
                
                x = torch.cat([x, rx_t], dim=0)
                y = torch.cat([y, ry_t], dim=0)

        logits = pl_module(x)
        num_classes = logits.size(1)
        
        # One-hot targets for BCE
        targets = F.one_hot(y, num_classes=num_classes).float()
        
        # Apply KD for old classes if we have an old model
        if self.old_model is not None:
            num_old_classes = self.old_model.fc.out_features
            with torch.no_grad():
                # Cast x to match old_model's parameter dtype to avoid AMP/mixed-precision crashes
                old_dtype = next(self.old_model.parameters()).dtype
                old_logits = self.old_model(x.to(dtype=old_dtype))
                old_targets = torch.sigmoid(old_logits).to(dtype=x.dtype)
            
            # Replace the targets for old classes with the old model's probabilities
            targets[:, :num_old_classes] = old_targets
            
        loss = F.binary_cross_entropy_with_logits(logits, targets)
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
