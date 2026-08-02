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
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))

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
            # We subtract margin from the target class's cosine similarity.
            # This makes the target class's logit artificially smaller during training,
            # forcing the network to push the feature even closer to the correct prototype
            # to achieve the same loss, thereby creating a geometric margin.
            
            # create a mask for the target class
            batch_size = features.size(0)
            target_mask = torch.zeros_like(logits, dtype=torch.bool)
            target_mask[torch.arange(batch_size, device=logits.device), targets] = True
            
            # Apply margin
            logits = torch.where(target_mask, logits - self.margin, logits)
            
        # 4. Scale logits
        logits = logits * self.scale
        
        return logits

    def expand(self, num_new_classes: int) -> None:
        old_out = self.num_classes
        new_out = old_out + num_new_classes
        device = self.weight.device
        
        new_weight = nn.Parameter(torch.Tensor(new_out, self.in_features).to(device))
        nn.init.kaiming_uniform_(new_weight, a=torch.math.sqrt(5))
        
        with torch.no_grad():
            new_weight.data[:old_out] = self.weight.data
            
        self.weight = new_weight
        self.num_classes = new_out
