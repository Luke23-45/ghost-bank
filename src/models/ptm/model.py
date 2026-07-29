from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.nn.init import kaiming_normal_

from src.models.base import BaseModel


class PTModel(BaseModel):
    """Pre-Trained Model wrapper for class-incremental learning.

    Wraps a torchvision pre-trained backbone (ResNet by default) with
    CIFAR-adapted input layers and supports multiple classifier modes:

    - ``linear``: standard FC layer (expandable for incremental tasks).
    - ``prototype``: classification via nearest-prototype cosine similarity.
      In this mode, ``classify()`` computes logits as cosine similarity
      between features and stored prototypes.

    The backbone can be frozen or partially fine-tuned.  For CIFAR-100
    (32×32 input), the first convolution is replaced with a 3×3 stride-1
    conv and the max-pool layer is removed.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 10,
        freeze_backbone: bool = True,
        embedding_dim: int | None = None,
        classifier_mode: str = "linear",
    ) -> None:
        super().__init__()
        self._classifier_mode = classifier_mode
        self._num_classes = num_classes
        self._backbone_name = backbone

        self._backbone, feat_dim = self._build_backbone(backbone, pretrained)
        self._embedding_dim = embedding_dim or feat_dim

        if freeze_backbone:
            self._freeze_backbone(self._backbone)

        self._temperature = 10.0

        if classifier_mode == "linear":
            self.head = nn.Linear(self._embedding_dim, num_classes)
        elif classifier_mode == "prototype":
            self.head = nn.Identity()
            self.register_buffer("prototypes", torch.zeros(num_classes, self._embedding_dim))
            self.register_buffer("prototype_counts", torch.zeros(num_classes, dtype=torch.long))
        else:
            raise ValueError(f"Unknown classifier_mode: {classifier_mode}")

    @staticmethod
    def _build_backbone(
        backbone: str, pretrained: bool,
    ) -> tuple[nn.Module, int]:
        """Build a CIFAR-adapted pre-trained backbone.

        Returns ``(backbone, feature_dim)``.
        """
        if backbone == "resnet18":
            model_fn = tv_models.resnet18
            weights_enum = tv_models.ResNet18_Weights
            feat_dim = 512
        elif backbone == "resnet34":
            model_fn = tv_models.resnet34
            weights_enum = tv_models.ResNet34_Weights
            feat_dim = 512
        elif backbone == "resnet50":
            model_fn = tv_models.resnet50
            weights_enum = tv_models.ResNet50_Weights
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        weights = weights_enum.DEFAULT if pretrained else None
        net = model_fn(weights=weights)

        in_planes = net.conv1.in_channels
        old_weight = net.conv1.weight.data

        new_conv1 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            if old_weight.shape[-1] == 7:
                pooled = F.adaptive_avg_pool2d(old_weight, (3, 3))
                new_conv1.weight.data.copy_(pooled)
            else:
                kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")

        net.conv1 = new_conv1
        net.maxpool = nn.Identity()

        net.fc = nn.Identity()

        return net, feat_dim

    @staticmethod
    def _freeze_backbone(backbone: nn.Module) -> None:
        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_stage(self) -> None:
        """Enable gradients for the last residual stage (layer4)."""
        for param in self._backbone.layer4.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding vectors ``[B, D]`` from the backbone."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._classifier_mode == "prototype":
            features = self.extract_features(x)
            return self._prototype_logits(features)
        features = self.extract_features(x)
        return self.classifier(features)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits from pre-extracted features."""
        if self._classifier_mode == "prototype":
            return self._prototype_logits(features)
        return self.classifier(features)

    def _prototype_logits(self, features: torch.Tensor) -> torch.Tensor:
        features_norm = F.normalize(features, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        return torch.mm(features_norm, proto_norm.t()) * self._temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    def update_prototypes(self, class_id: int, features: torch.Tensor) -> None:
        if self._classifier_mode != "prototype":
            return
        n = features.shape[0]
        old_count = self.prototype_counts[class_id].item()
        new_count = old_count + n
        old_proto = self.prototypes[class_id].clone()
        updated = (old_proto * old_count + features.sum(dim=0)) / new_count
        self.prototypes[class_id] = updated
        self.prototype_counts[class_id] = new_count

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        if prototypes.shape != self.prototypes.shape:
            self.prototypes = torch.zeros_like(prototypes)
        self.prototypes.copy_(prototypes)

    def expand_head(self, num_new_classes: int) -> None:
        if self._classifier_mode == "prototype":
            old_num = self.prototypes.shape[0]
            new_num = old_num + num_new_classes
            new_prototypes = torch.zeros(new_num, self._embedding_dim, device=self.prototypes.device)
            new_prototypes[:old_num] = self.prototypes
            self.prototypes = new_prototypes
            new_counts = torch.zeros(new_num, dtype=torch.long, device=self.prototype_counts.device)
            new_counts[:old_num] = self.prototype_counts
            self.prototype_counts = new_counts
            self._num_classes = new_num
            return

        old_num = self.head.out_features
        new_out = old_num + num_new_classes

        device = self.head.weight.device
        dtype = self.head.weight.dtype

        old_weight = self.head.weight.data
        old_bias = self.head.bias.data if self.head.bias is not None else None

        new_head = nn.Linear(self.head.in_features, new_out).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_head.weight.data[:old_num] = old_weight
            if old_bias is not None:
                new_head.bias.data[:old_num] = old_bias
            nn.init.normal_(new_head.weight.data[old_num:], mean=0.0, std=0.001)
            if new_head.bias is not None:
                new_head.bias.data[old_num:].zero_()

        self.head = new_head
        self._num_classes = new_out

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def classifier(self) -> nn.Module:
        return self.head

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
