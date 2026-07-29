from __future__ import annotations

import torch

from src.data.cifar100.splits import create_class_wise_splits


def test_create_class_wise_splits_basic():
    num_classes = 5
    images_per_class = 60
    total = num_classes * images_per_class
    images = torch.randint(0, 256, (total, 32, 32, 3), dtype=torch.uint8)
    targets = torch.cat([torch.full((images_per_class,), c) for c in range(num_classes)])

    result = create_class_wise_splits(
        images, targets, num_classes,
        probe_size=10, val_size=5, seed=42,
    )

    for key in ("probe_images", "probe_targets", "val_images", "val_targets", "train_images", "train_targets"):
        assert key in result, f"Missing key: {key}"

    assert result["probe_images"].shape[0] == num_classes * 10
    assert result["val_images"].shape[0] == num_classes * 5
    assert result["train_images"].shape[0] == num_classes * (images_per_class - 15)


def test_create_class_wise_splits_deterministic():
    num_classes = 3
    images_per_class = 50
    total = num_classes * images_per_class
    images = torch.randint(0, 256, (total, 32, 32, 3), dtype=torch.uint8)
    targets = torch.cat([torch.full((images_per_class,), c) for c in range(num_classes)])

    r1 = create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=42)
    r2 = create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=42)

    assert torch.equal(r1["probe_images"], r2["probe_images"])
    assert torch.equal(r1["train_images"], r2["train_images"])
    assert torch.equal(r1["val_images"], r2["val_images"])


def test_create_class_wise_splits_different_seed():
    num_classes = 3
    images_per_class = 50
    total = num_classes * images_per_class
    images = torch.randint(0, 256, (total, 32, 32, 3), dtype=torch.uint8)
    targets = torch.cat([torch.full((images_per_class,), c) for c in range(num_classes)])

    r1 = create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=42)
    r2 = create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=99)

    assert not torch.equal(r1["probe_images"], r2["probe_images"])


def test_create_class_wise_splits_no_overlap():
    num_classes = 3
    images_per_class = 50
    total = num_classes * images_per_class
    images = torch.randint(0, 256, (total, 32, 32, 3), dtype=torch.uint8)
    targets = torch.cat([torch.full((images_per_class,), c) for c in range(num_classes)])

    result = create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=42)

    probe_set = set(result["probe_images"].unbind(0))
    val_set = set(result["val_images"].unbind(0))
    train_set = set(result["train_images"].unbind(0))

    assert probe_set.isdisjoint(val_set), "Probe and val sets overlap"
    assert probe_set.isdisjoint(train_set), "Probe and train sets overlap"
    assert val_set.isdisjoint(train_set), "Val and train sets overlap"


def test_create_class_wise_splits_preserves_labels():
    num_classes = 5
    images_per_class = 60
    total = num_classes * images_per_class
    images = torch.randint(0, 256, (total, 32, 32, 3), dtype=torch.uint8)
    targets = torch.cat([torch.full((images_per_class,), c) for c in range(num_classes)])

    result = create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=42)

    for c in range(num_classes):
        c_probe_mask = result["probe_targets"] == c
        assert c_probe_mask.sum() == 10, f"Class {c} should have 10 probe samples"
        c_val_mask = result["val_targets"] == c
        assert c_val_mask.sum() == 5, f"Class {c} should have 5 val samples"
        c_train_mask = result["train_targets"] == c
        assert c_train_mask.sum() == images_per_class - 15, f"Class {c} should have {images_per_class - 15} train samples"


def test_create_class_wise_splits_insufficient_samples():
    num_classes = 2
    images = torch.randint(0, 256, (5, 32, 32, 3), dtype=torch.uint8)
    targets = torch.tensor([0, 0, 0, 0, 0])

    import pytest
    with pytest.raises(ValueError, match="Class 0 has 5 images"):
        create_class_wise_splits(images, targets, num_classes, probe_size=10, val_size=5, seed=42)
