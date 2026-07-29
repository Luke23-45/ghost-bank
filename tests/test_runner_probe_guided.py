from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.bank.core.probe import ProbeScorer
from src.methods.probe_guided import ProbeGuidedMethod
from src.models.ptm import PTModel


def test_spearman_correlation_happy_path():
    from studies.runner.cifar100.probe_guided.run import _compute_spearman_correlation

    probe_scorer = ProbeScorer(num_classes=6, smoothing=0.0)
    probe_scorer.update([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    acc_matrix = [
        [80.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [70.0, 85.0, 0.0, 0.0, 0.0, 0.0],
        [60.0, 75.0, 90.0, 0.0, 0.0, 0.0],
        [50.0, 65.0, 80.0, 95.0, 0.0, 0.0],
        [40.0, 55.0, 70.0, 85.0, 92.0, 0.0],
        [30.0, 45.0, 60.0, 75.0, 82.0, 88.0],
    ]

    corr = _compute_spearman_correlation(probe_scorer, acc_matrix, 6, 1)
    assert corr is not None
    assert -1.0 <= corr <= 1.0


def test_spearman_correlation_insufficient_history():
    from studies.runner.cifar100.probe_guided.run import _compute_spearman_correlation

    probe_scorer = ProbeScorer(num_classes=3, smoothing=0.0)
    probe_scorer.update([1.0, 2.0, 3.0])

    acc_matrix = [[80.0, 0.0, 0.0]]

    corr = _compute_spearman_correlation(probe_scorer, acc_matrix, 1, 1)
    assert corr is None


def test_spearman_correlation_constant_scores():
    from studies.runner.cifar100.probe_guided.run import _compute_spearman_correlation

    probe_scorer = ProbeScorer(num_classes=4, smoothing=0.0)
    probe_scorer.update([1.0, 1.0, 1.0, 1.0])

    acc_matrix = [
        [80.0, 0.0, 0.0, 0.0],
        [70.0, 85.0, 0.0, 0.0],
        [60.0, 75.0, 90.0, 0.0],
        [50.0, 65.0, 80.0, 95.0],
    ]

    corr = _compute_spearman_correlation(probe_scorer, acc_matrix, 4, 1)
    assert corr is None


def test_per_class_forgetting():
    from studies.runner.cifar100.probe_guided.run import _compute_per_class_forgetting

    acc_matrix = [
        [90.0, 0.0, 0.0],
        [80.0, 85.0, 0.0],
        [70.0, 75.0, 80.0],
    ]

    forgetting = _compute_per_class_forgetting(acc_matrix, 3, 2)
    assert len(forgetting) == 6
    assert forgetting[0] == pytest.approx(20.0)
    assert forgetting[2] == pytest.approx(10.0)
    assert forgetting[4] == 0.0
    assert forgetting[1] == pytest.approx(20.0)


def test_aggregate_metrics():
    from studies.runner.cifar100.probe_guided.run import _aggregate_metrics

    all_metrics = [
        {"method": "probe_guided", "seed": 42, "test/avg_acc": 60.0, "test/forgetting": 15.0},
        {"method": "probe_guided", "seed": 1337, "test/avg_acc": 62.0, "test/forgetting": 13.0},
    ]

    agg = _aggregate_metrics(all_metrics)
    assert agg["method"] == "probe_guided"
    assert agg["num_seeds"] == 2
    assert agg["test/avg_acc_mean"] == 61.0
    assert agg["test/forgetting_mean"] == 14.0


def test_runner_compose_configs():
    from studies.runner.cifar100.probe_guided.run import ProbeGuidedCIFAR100Runner

    runner = ProbeGuidedCIFAR100Runner()
    configs = runner.compose_configs()
    assert len(configs) == 3
    names = [name for _, name in configs]
    assert "probe_guided" in names
    assert "uniform_replay" in names
    assert "frozen_baseline" in names


def test_probe_guided_method_allocation_history():
    method = ProbeGuidedMethod(memory_total=100, memory_floor=1)
    method._exemplar_bank = {0: [(1, 0)], 1: [(2, 1)], 2: [(3, 2)]}

    assert method.allocation_history == []

    method.update_allocation([10.0, 1.0, 0.1])
    assert len(method.allocation_history) == 1
    assert sum(method.allocation_history[0]) == 100
    assert all(a >= 1 for a in method.allocation_history[0])

    method.update_allocation([0.1, 10.0, 1.0])
    assert len(method.allocation_history) == 2


def test_ptm_model_state_dict_roundtrip():
    model = PTModel(
        backbone="resnet18",
        pretrained=False,
        num_classes=10,
        freeze_backbone=True,
    )
    state = model.state_dict()
    model2 = PTModel(
        backbone="resnet18",
        pretrained=False,
        num_classes=10,
        freeze_backbone=True,
    )
    model2.load_state_dict(state)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2)


def test_probe_scorer_state_dict_roundtrip():
    scorer = ProbeScorer(num_classes=5, smoothing=0.3)
    scorer.update([0.5, 1.0, 0.0, 2.0, 1.5])
    scorer.update([0.6, 0.9, 0.1, 2.1, 1.4])

    state = scorer.state_dict()
    scorer2 = ProbeScorer(num_classes=5, smoothing=0.3)
    scorer2.load_state_dict(state)

    assert scorer2.history == scorer.history
    assert scorer2.smoothed_scores == scorer.smoothed_scores
    assert scorer2.raw_scores == scorer.raw_scores
