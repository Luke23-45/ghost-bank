import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginCosineHead(nn.Module):
    """
    Robust Cosine Similarity classification head with learnable margins.
    Prevents weight magnitude explosion and enforces geometric separation between classes.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.accepts_targets = True
        
        # Learnable class prototypes (weights)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            features: [batch_size, in_features]
            targets: [batch_size] ground truth labels (required during training if margin > 0)
        Returns:
            logits: [batch_size, num_classes]
        """
        # 1. L2 Normalize features and weights
        features_norm = F.normalize(features, p=2, dim=-1)
        weight_norm = F.normalize(self.weight, p=2, dim=-1)
        
        # 2. Compute cosine similarity
        # logits shape: [batch_size, num_classes]
        logits = F.linear(features_norm, weight_norm)
        
        # 3. Apply margin (only during training when targets are provided)
        if self.training and self.margin > 0.0 and targets is not None:
            # Subtract the margin from the target class's cosine similarity in place,
            # forcing the network to push the feature closer to the correct prototype
            # to achieve the same loss, thereby creating a geometric margin.
            index = torch.arange(logits.size(0), device=logits.device)
            logits[index, targets] -= self.margin

        # 4. Scale logits
        logits = logits * self.scale

        return logits

    def unmargined_logits(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Recover the cosine logits the margin was subtracted from.

        During training the target class's cosine similarity is reduced by
        ``margin`` before scaling (see ``forward``), so training-time logits
        are shifted by ``margin * scale`` at the true-class position.  When
        the same logits feed knowledge distillation they must be undone:
        the KD teacher snapshot ran in eval mode and never had a margin
        applied, so comparing margined student scores against the teacher's
        injects a spurious ``margin * scale`` gap on the true class and the
        KD gradient fights the margin head on old classes.
        """
        if self.training and self.margin > 0.0 and targets is not None:
            logits = logits.clone()
            index = torch.arange(logits.size(0), device=logits.device)
            logits[index, targets] += self.margin * self.scale
        return logits

    def expand(self, num_new_classes: int) -> None:
        old_out = self.num_classes
        new_out = old_out + num_new_classes
        device = self.weight.device

        new_weight = nn.Parameter(torch.empty(new_out, self.in_features, device=device))
        nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

        with torch.no_grad():
            new_weight.data[:old_out] = self.weight.data

        self.weight = new_weight
        self.num_classes = new_out

    def imprint(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_ids: list[int] | None = None,
    ) -> None:
        """Weight imprinting: initialize head rows with L2-normalized class-mean features.

        Mirrors LUCIR/PODNet: new classes start from their data rather than from
        random directions, so they do not initially overlap old prototypes.
        Only the rows given in ``class_ids`` are touched; by default, every
        class present in ``labels`` is imprinted.  Classes with no samples are
        left untouched, as are out-of-range ids.
        """
        if class_ids is None:
            ids = torch.unique(labels)
        else:
            ids = torch.unique(
                torch.as_tensor(class_ids, dtype=labels.dtype, device=labels.device)
            )
        ids = ids[(ids >= 0) & (ids < self.weight.size(0))]
        if ids.numel() == 0:
            return

        keep = torch.isin(labels, ids)
        feats = features[keep]
        lbls = labels[keep]
        if feats.numel() == 0:
            return

        # Dense per-class means via a vectorized segment reduction over `ids`.
        dense = torch.searchsorted(ids, lbls)  # ids is sorted (torch.unique)
        sums = torch.zeros(ids.numel(), feats.size(1), dtype=feats.dtype, device=feats.device)
        sums.index_add_(0, dense, feats)
        counts = torch.bincount(dense, minlength=ids.numel())
        nonempty = counts > 0
        means = torch.zeros_like(sums)
        means[nonempty] = F.normalize(sums[nonempty] / counts[nonempty].unsqueeze(1), dim=1)

        with torch.no_grad():
            self.weight.data[ids[nonempty]] = means[nonempty].to(device=self.weight.device)
