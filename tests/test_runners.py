"""Tests for runner infrastructure: factory functions and AbstractRunner."""

from omegaconf import OmegaConf

from src.bank.strategies import StaticReplayBank, HerdingReplayBank
from src.methods.baseline import BaselineMethod
from src.methods.static_bank import StaticBankMethod
from src.methods.uniform_herding import UniformHerdingMethod
from src.models import ResNet
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
