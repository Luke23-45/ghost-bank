"""Unit tests for the MarginCosineHead and its integration into ResNet / PL module.

Covers: the cosine+margin+scale math, train/eval gating, head expansion,
weight imprinting, ResNet wiring, and that per-sample targets actually reach
the head through the PL module and the replay methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.methods.baseline import BaselineMethod
from src.methods.static_bank import StaticBankMethod
from src.methods.uniform_herding import UniformHerdingMethod
from src.models.heads.cosine_margin import MarginCosineHead
from src.models.resnet.model import ResNet
from src.training import GhostBankLightningModule


class RecordingModel(nn.Module):
    """Records the last ``(x, targets)`` it was called with."""

    def __init__(self) -> None:
        super().__init__()
        self.last_targets: torch.Tensor | None = None
        self.last_x: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        self.last_x = x
        self.last_targets = targets
        return torch.randn(x.size(0), 3, requires_grad=True)


# -- Head math -----------------------------------------------------------------


def _head(in_features: int = 4, num_classes: int = 3, scale: float = 10.0, margin: float = 0.5) -> MarginCosineHead:
    head = MarginCosineHead(in_features, num_classes, scale=scale, margin=margin)
    with torch.no_grad():
        head.weight.copy_(F.normalize(torch.randn(num_classes, in_features), dim=1))
    return head


def _manual_cosine_logits(head: MarginCosineHead, features: torch.Tensor) -> torch.Tensor:
    w = F.normalize(head.weight, dim=1)
    return F.linear(F.normalize(features, dim=-1), w)


def test_margin_applied_only_in_training_with_targets():
    head = _head(margin=0.5, scale=10.0)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))

    head.train()
    logits_train = head(x, y)
    manual = _manual_cosine_logits(head, x)
    for i, c in enumerate(y.tolist()):
        expected = manual[i].clone()
        expected[c] -= head.margin
        assert torch.allclose(logits_train[i], expected * head.scale, atol=1e-6)

    head.eval()
    logits_eval = head(x, y)
    assert torch.allclose(logits_eval, manual * head.scale, atol=1e-6)

    logits_eval_mode = head(x, y)
    assert torch.allclose(logits_eval_mode, manual * head.scale, atol=1e-6)


def test_margin_requires_targets_even_in_training():
    head = _head(margin=0.5, scale=10.0)
    x = torch.randn(4, 4)
    head.train()
    logits = head(x, None)
    manual = _manual_cosine_logits(head, x)
    assert torch.allclose(logits, manual * head.scale, atol=1e-6)


def test_scale_applied_to_all_logits():
    head = _head(scale=7.0, margin=0.0)
    x = torch.randn(4, 4)
    logits = head(x)
    manual = _manual_cosine_logits(head, x)
    assert torch.allclose(logits, manual * 7.0, atol=1e-6)


def test_expand_preserves_old_rows_and_grows():
    head = _head(in_features=4, num_classes=3)
    old_weight = head.weight.data.clone()
    head.expand(2)
    assert head.num_classes == 5
    assert head.weight.shape == (5, 4)
    assert torch.equal(head.weight.data[:3], old_weight)


def test_expand_keeps_margin_and_scale():
    head = _head(scale=12.0, margin=0.4)
    head.expand(1)
    assert head.scale == 12.0
    assert head.margin == 0.4


def test_imprint_sets_normalized_class_means():
    head = _head(in_features=4, num_classes=3)
    rng = torch.Generator().manual_seed(0)
    feats = torch.randn(30, 4, generator=rng)
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)

    before = head.weight.data.clone()
    head.imprint(feats, labels)

    for c in range(3):
        mean = feats[labels == c].mean(dim=0)
        expected = F.normalize(mean, dim=0)
        assert torch.allclose(head.weight.data[c], expected, atol=1e-6)
    assert torch.norm(head.weight.data, dim=1).max().item() <= 1.0 + 1e-5
    assert not torch.allclose(head.weight.data[1], before[1])  # rows did change


def test_imprint_only_touches_requested_class_ids():
    head = _head(in_features=4, num_classes=3)
    feats = torch.randn(20, 4)
    labels = torch.tensor([0] * 10 + [1] * 10)
    before = head.weight.data.clone()
    head.imprint(feats, labels, class_ids=[1])
    assert torch.equal(head.weight.data[0], before[0])
    assert torch.equal(head.weight.data[2], before[2])
    assert not torch.equal(head.weight.data[1], before[1])


def test_imprint_ignores_labels_outside_rows():
    head = _head(in_features=4, num_classes=2)
    feats = torch.randn(4, 4)
    labels = torch.tensor([0, 0, 7, 7])
    head.imprint(feats, labels)  # class 7 out of range -> silently skipped
    assert head.weight.shape == (2, 4)


def test_imprint_ignores_negative_labels():
    head = _head(in_features=4, num_classes=2)
    feats = torch.randn(4, 4)
    labels = torch.tensor([0, -1, 0, 1])
    head.imprint(feats, labels)
    expected0 = F.normalize(feats[labels == 0].mean(0), dim=0)
    expected1 = F.normalize(feats[labels == 1].mean(0), dim=0)
    assert torch.allclose(head.weight.data[0], expected0, atol=1e-6)
    assert torch.allclose(head.weight.data[1], expected1, atol=1e-6)


# -- ResNet wiring ------------------------------------------------------------


def test_resnet_default_head_is_linear():
    model = ResNet(num_classes=5, base_filters=8)
    assert isinstance(model.fc, nn.Linear)


def test_resnet_cosine_head_wiring():
    model = ResNet(num_classes=5, base_filters=8, head="cosine_margin", head_scale=22.0, head_margin=0.3)
    assert isinstance(model.fc, MarginCosineHead)
    assert model.fc.scale == 22.0
    assert model.fc.margin == 0.3

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 3])
    model.eval()
    logits = model(x)
    assert logits.shape == (2, 5)
    model.train()
    logits_train = model(x, targets=y)
    assert logits_train.shape == (2, 5)
    assert not torch.allclose(logits_train, logits)  # margin active in train

    model.expand_head(2)
    assert model.fc.num_classes == 7
    assert model.num_classes == 7


def test_resnet_linear_head_ignores_targets():
    model = ResNet(num_classes=5, base_filters=8)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    model.train()
    logits_with = model(x, targets=y)
    logits_without = model(x)
    assert torch.equal(logits_with, logits_without)


def test_unknown_head_name_falls_back_to_linear():
    model = ResNet(num_classes=5, base_filters=8, head="bogus")
    assert isinstance(model.fc, nn.Linear)


# -- Targets plumbing through the PL module and methods -----------------------


def test_pl_module_forwards_targets():
    model = RecordingModel()
    method = BaselineMethod()
    pl_module = GhostBankLightningModule(model=model, method=method, learning_rate=0.01)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 2])
    pl_module(x, targets=y)
    assert pl_module.model.last_targets is not None
    assert torch.equal(pl_module.model.last_targets, y)


@pytest.mark.parametrize("method_cls", [BaselineMethod, StaticBankMethod, UniformHerdingMethod])
def test_methods_pass_targets_to_pl_module(method_cls):
    model = RecordingModel()
    method = method_cls() if method_cls is BaselineMethod else method_cls(retrieval_budget=4)
    pl_module = GhostBankLightningModule(model=model, method=method, learning_rate=0.01)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 2, 0])
    loss = method.compute_loss((x, y), pl_module, bank=None)
    assert loss.ndim == 0
    assert torch.equal(pl_module.model.last_targets, y)


def test_pl_module_forward_without_targets():
    model = RecordingModel()
    method = BaselineMethod()
    pl_module = GhostBankLightningModule(model=model, method=method, learning_rate=0.01)
    x = torch.randn(2, 4)
    pl_module(x)
    assert pl_module.model.last_targets is None
