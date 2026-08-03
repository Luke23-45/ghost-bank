"""Unit tests for the P0 ablation knobs: predict_mode and bank selection."""

from __future__ import annotations

import random

import pytest
import torch
from omegaconf import OmegaConf

from src.bank.strategies import HerdingReplayBank
from src.bank.strategies.herding import _herding_select
from src.methods import UniformHerdingMethod
from studies.runner.common.base_runner import create_bank, create_method


class _ArgmaxModel(torch.nn.Module):

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits


class _FakePl:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model


class _FeatureModel(torch.nn.Module):

    def __init__(self, features: torch.Tensor) -> None:
        super().__init__()
        self._features = features

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._features

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._features


class TestPredictMode:

    def test_default_is_nme(self):
        method = UniformHerdingMethod()
        assert method.predict_mode == "nme"

    def test_head_mode_returns_argmax_of_head_logits(self):
        logits = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        method = UniformHerdingMethod(predict_mode="head")
        preds = method.predict(
            torch.zeros(2, 3), _FakePl(_ArgmaxModel(logits)), bank=None
        )
        assert torch.equal(preds, logits.argmax(dim=-1))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="predict_mode"):
            UniformHerdingMethod(predict_mode="ghost")

    def test_create_method_wires_in_mode(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "uniform_herding",
                "retrieval_budget": 8,
                "predict_mode": "head",
            }
        })
        method = create_method(cfg)
        assert method.predict_mode == "head"
        default = create_method(OmegaConf.create({
            "method": {"name": "uniform_herding", "retrieval_budget": 8}
        }))
        assert default.predict_mode == "nme"


def _constant_pool(n: int) -> list:
    return [
        (torch.full((3, 4, 4), float(i)), torch.tensor(0))
        for i in range(n)
    ]


def _picked_indices(bank: HerdingReplayBank) -> list:
    return [int(item[0][0, 0, 0].item()) for item in bank.selected[0]]


class TestSelectionKnob:

    def test_default_is_herding(self):
        bank = HerdingReplayBank(num_classes=1, total_budget=4, seed=0)
        assert bank._selection == "herding"

    def test_invalid_selection_raises(self):
        with pytest.raises(ValueError, match="selection"):
            HerdingReplayBank(
                num_classes=1, total_budget=4, seed=0, selection="fancy"
            )

    def test_random_branch_matches_seeded_random(self):
        seed = 7
        features = torch.randn(8, 5)
        bank = HerdingReplayBank(
            num_classes=1, total_budget=4, seed=seed, selection="random"
        )
        for item in _constant_pool(8):
            bank.store([item])
        bank.rebuild_selected(_FeatureModel(features))
        expected = random.Random(seed).sample(range(8), 4)
        assert _picked_indices(bank) == expected

    def test_herding_branch_matches_herding_select(self):
        features = torch.randn(8, 5)
        bank = HerdingReplayBank(
            num_classes=1, total_budget=4, seed=99, selection="herding"
        )
        for item in _constant_pool(8):
            bank.store([item])
        bank.rebuild_selected(_FeatureModel(features))
        expected = _herding_select(features, 4)
        assert _picked_indices(bank) == expected

    def test_create_bank_wires_in_selection(self):
        cfg = OmegaConf.create({
            "bank": {"name": "herding", "selection": "random"},
            "data": {"memory_total": 100},
        })
        bank = create_bank(cfg, num_classes=4, run_seed=3)
        assert bank is not None
        assert bank._selection == "random"
        default = create_bank(OmegaConf.create({
            "bank": {"name": "herding"},
            "data": {"memory_total": 100},
        }), num_classes=4, run_seed=3)
        assert default._selection == "herding"