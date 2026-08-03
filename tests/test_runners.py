"""Tests for runner infrastructure: factory functions and AbstractRunner."""

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.bank.strategies import StaticReplayBank, HerdingReplayBank
from src.methods.baseline import BaselineMethod
from src.methods.static_bank import StaticBankMethod
from src.methods.uniform_herding import UniformHerdingMethod
from src.models import ResNet
from studies.runner.cifar100.run import CIFAR100Runner
from studies.runner.common.base_runner import (
    AbstractRunner,
    create_model,
    create_bank,
    create_method,
)


# -- create_model -------------------------------------------------------------

class TestCreateModel:
    def test_resnet(self):
        cfg = OmegaConf.create({
            "model": {
                "type": "resnet",
                "base_filters": 64,
                "dropout": 0.0,
            }
        })
        model = create_model(cfg, num_classes=10)
        assert isinstance(model, ResNet)

    def test_num_classes_from_arg_not_config(self):
        cfg = OmegaConf.create({
            "model": {
                "type": "resnet",
                "base_filters": 64,
                "num_classes": 3,
            }
        })
        model = create_model(cfg, num_classes=10)
        assert model.fc.out_features == 10


# -- create_bank --------------------------------------------------------------

class TestCreateBank:
    def test_no_bank_config_returns_none(self):
        cfg = OmegaConf.create({"method": {"name": "baseline"}})
        bank = create_bank(cfg, num_classes=10)
        assert bank is None

    def test_static_bank(self):
        cfg = OmegaConf.create({
            "bank": {
                "name": "static",
                "capacity_per_class": 100,
                "seed": 7,
            }
        })
        bank = create_bank(cfg, num_classes=10)
        assert isinstance(bank, StaticReplayBank)

    def test_herding_bank(self):
        cfg = OmegaConf.create({
            "data": {"memory_total": 2000},
            "bank": {
                "name": "herding",
                "seed": 42,
                "floor": 1,
            }
        })
        bank = create_bank(cfg, num_classes=10)
        assert isinstance(bank, HerdingReplayBank)

    def test_unknown_bank_returns_none(self):
        cfg = OmegaConf.create({
            "bank": {"name": "unknown"}
        })
        bank = create_bank(cfg, num_classes=10)
        assert bank is None


# -- create_method ------------------------------------------------------------

class TestCreateMethod:
    def test_baseline(self):
        cfg = OmegaConf.create({"method": {"name": "baseline"}})
        method = create_method(cfg)
        assert isinstance(method, BaselineMethod)

    def test_static_bank(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "static_bank",
                "retrieval_budget": 8,
                "warmup_steps": 0,
            }
        })
        method = create_method(cfg)
        assert isinstance(method, StaticBankMethod)

    def test_static_bank_with_warmup(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "static_bank",
                "retrieval_budget": 4,
                "warmup_steps": 100,
            }
        })
        method = create_method(cfg)
        assert isinstance(method, StaticBankMethod)
        assert method.warmup_steps == 100

    def test_uniform_herding(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "uniform_herding",
                "retrieval_budget": 8,
                "warmup_steps": 0,
            }
        })
        method = create_method(cfg)
        assert isinstance(method, UniformHerdingMethod)

    def test_uniform_herding_kd_params(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "uniform_herding",
                "retrieval_budget": 8,
                "kd_weight": 0.5,
                "kd_temperature": 4.0,
            }
        })
        method = create_method(cfg)
        assert method.kd_weight == 0.5
        assert method.kd_temperature == 4.0

    def test_uniform_herding_kd_defaults(self):
        cfg = OmegaConf.create({
            "method": {"name": "uniform_herding", "retrieval_budget": 8}
        })
        method = create_method(cfg)
        assert method.kd_weight == 0.0
        assert method.kd_temperature == 2.0

    def test_unknown_method_raises(self):
        import pytest
        cfg = OmegaConf.create({"method": {"name": "unknown"}})
        with pytest.raises(ValueError, match="Unsupported method"):
            create_method(cfg)


# -- AbstractRunner -----------------------------------------------------------

class TestAbstractRunner:
    def test_cannot_instantiate_directly(self):
        import pytest
        with pytest.raises(TypeError):
            AbstractRunner()

    def test_subclass_must_implement_compose_configs(self):
        import pytest
        with pytest.raises(TypeError):
            type("Incomplete", (AbstractRunner,), {})()

    def test_subclass_must_implement_run_experiment(self):
        import pytest
        with pytest.raises(TypeError):
            type("Incomplete", (AbstractRunner,), {"compose_configs": lambda self: []})()

    def test_concrete_subclass(self):
        class MinimalRunner(AbstractRunner):
            def compose_configs(self):
                return []
            def run_experiment(self, cfg, output_manager):
                return {}

        runner = MinimalRunner()
        assert runner.overrides == []
        assert runner.run() == []


# -- imprint hook --------------------------------------------------------------

class TestImprintHead:
    def _model(self, head: str) -> ResNet:
        cfg = OmegaConf.create({
            "model": {
                "type": "resnet",
                "base_filters": 8,
                "head": head,
            }
        })
        return create_model(cfg, num_classes=4)

    def _loader(self, features: torch.Tensor, labels: torch.Tensor) -> DataLoader:
        idx = torch.arange(features.size(0))
        return DataLoader(
            torch.utils.data.TensorDataset(idx, features, labels),
            batch_size=8,
            shuffle=False,
        )

    def test_task0_short_circuit(self):
        model = self._model("cosine_margin")
        before = model.fc.weight.data.clone()
        features = torch.randn(16, 3, 32, 32)
        labels = torch.tensor([0, 1] * 8)
        CIFAR100Runner._imprint_head(model, self._loader(features, labels), task_id=0, classes_per_task=2)
        assert torch.equal(model.fc.weight.data, before)

    def test_imprints_only_current_task_rows(self):
        model = self._model("cosine_margin")
        old_rows = model.fc.weight.data[:2].clone()
        features = torch.randn(32, 3, 32, 32)
        labels = torch.tensor([2] * 16 + [3] * 16)
        CIFAR100Runner._imprint_head(model, self._loader(features, labels), task_id=1, classes_per_task=2)
        feat0 = model.extract_features(features[:16]).detach()
        feat1 = model.extract_features(features[16:]).detach()
        w = model.fc.weight.data
        assert torch.allclose(w[2], torch.nn.functional.normalize(feat0.mean(0), dim=0), atol=1e-6)
        assert torch.allclose(w[3], torch.nn.functional.normalize(feat1.mean(0), dim=0), atol=1e-6)
        assert torch.equal(w[:2], old_rows)

    def test_linear_head_is_noop(self):
        model = self._model("linear")
        before = model.fc.weight.data.clone()
        features = torch.randn(16, 3, 32, 32)
        labels = torch.tensor([2, 3] * 8)
        if hasattr(model.fc, "imprint"):  # mirrors the runner's guard at the call site
            CIFAR100Runner._imprint_head(model, self._loader(features, labels), task_id=1, classes_per_task=2)
        assert torch.equal(model.fc.weight.data, before)

    def test_imprint_guard_by_head_type(self):
        assert hasattr(self._model("cosine_margin").fc, "imprint")
        assert not hasattr(self._model("linear").fc, "imprint")
