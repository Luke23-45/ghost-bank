from __future__ import annotations

import pytest
import torch

from src.models.ptm.model import PTModel


@pytest.fixture
def ptm_model():
    model = PTModel(
        backbone="resnet18",
        pretrained=False,
        num_classes=10,
        freeze_backbone=True,
        classifier_mode="linear",
    )
    return model


@pytest.fixture
def ptm_prototype():
    model = PTModel(
        backbone="resnet18",
        pretrained=False,
        num_classes=10,
        freeze_backbone=True,
        classifier_mode="prototype",
    )
    return model


def test_ptm_forward_shape(ptm_model):
    x = torch.randn(4, 3, 32, 32)
    out = ptm_model(x)
    assert out.shape == (4, 10)


def test_ptm_extract_features(ptm_model):
    x = torch.randn(4, 3, 32, 32)
    features = ptm_model.extract_features(x)
    assert features.shape == (4, 512)


def test_ptm_frozen_backbone(ptm_model):
    for param in ptm_model.backbone.parameters():
        assert not param.requires_grad


def test_ptm_expand_head_linear(ptm_model):
    ptm_model.expand_head(5)
    assert ptm_model.num_classes == 15
    assert ptm_model.classifier.out_features == 15

    x = torch.randn(2, 3, 32, 32)
    out = ptm_model(x)
    assert out.shape == (2, 15)


def test_ptm_prototype_forward(ptm_prototype):
    x = torch.randn(4, 3, 32, 32)
    out = ptm_prototype(x)
    assert out.shape == (4, 10)


def test_ptm_prototype_update(ptm_prototype):
    features = torch.randn(5, ptm_prototype.embedding_dim)
    ptm_prototype.update_prototypes(0, features)
    assert ptm_prototype.prototype_counts[0].item() == 5


def test_ptm_prototype_expand(ptm_prototype):
    ptm_prototype.expand_head(5)
    assert ptm_prototype.num_classes == 15
    assert ptm_prototype.prototypes.shape[0] == 15
    assert ptm_prototype.prototype_counts.shape[0] == 15


def test_ptm_unfreeze_last_stage(ptm_model):
    ptm_model.unfreeze_last_stage()
    for param in ptm_model.backbone.layer4.parameters():
        assert param.requires_grad
    for param in ptm_model.backbone.layer3.parameters():
        assert not param.requires_grad


def test_ptm_temperature(ptm_prototype):
    assert ptm_prototype.temperature == 10.0
    ptm_prototype.temperature = 5.0
    assert ptm_prototype.temperature == 5.0


def test_ptm_set_prototypes(ptm_prototype):
    new_protos = torch.randn(10, ptm_prototype.embedding_dim)
    ptm_prototype.set_prototypes(new_protos)
    assert torch.equal(ptm_prototype.prototypes, new_protos)
