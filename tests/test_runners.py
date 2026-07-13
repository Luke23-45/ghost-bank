"""Tests for runner infrastructure: factory functions and AbstractRunner."""

from omegaconf import DictConfig, OmegaConf

from src.bank.strategies import StaticReplayBank, ExposureDebtGhostBank
from src.data.synthetic import SyntheticDataModule
from src.methods import (
    BaselineMethod,
    ClassBalancedMethod,
    EDGBMethod,
    FocalLossMethod,
    StaticBankMethod,
)
from src.models import MLPClassifier
from studies.runner.common.base_runner import (
    AbstractRunner,
    create_datamodule,
    create_model,
    create_bank,
    create_method,
)


# -- create_datamodule --------------------------------------------------------

class TestCreateDatamodule:
    def test_synthetic(self):
        cfg = OmegaConf.create({
            "data": {
                "type": "synthetic",
                "seed": 13,
                "imbalance_ratio": 100,
                "majority_train": 2000,
                "test_per_class": 500,
                "batch_size": 32,
            }
        })
        dm = create_datamodule(cfg)
        assert isinstance(dm, SyntheticDataModule)

    def test_synthetic_config_values_passed(self):
        cfg = OmegaConf.create({
            "data": {
                "type": "synthetic",
                "seed": 42,
                "imbalance_ratio": 50,
                "majority_train": 1000,
                "test_per_class": 200,
                "batch_size": 64,
            }
        })
        dm = create_datamodule(cfg)
        assert dm.config.seed == 42
        assert dm.config.imbalance_ratio == 50
        assert dm.config.batch_size == 64


# -- create_model -------------------------------------------------------------

class TestCreateModel:
    def test_mlp_classifier(self):
        cfg = OmegaConf.create({
            "model": {
                "type": "mlp",
                "input_dim": 4,
                "hidden_dim": 32,
                "num_classes": 5,
            }
        })
        model = create_model(cfg, num_classes=5)
        assert isinstance(model, MLPClassifier)
        assert model.fc1.in_features == 4
        assert model.fc1.out_features == 32
        assert model.fc2.out_features == 5

    def test_num_classes_from_arg_not_config(self):
        cfg = OmegaConf.create({
            "model": {
                "type": "mlp",
                "input_dim": 2,
                "hidden_dim": 16,
                "num_classes": 3,
            }
        })
        model = create_model(cfg, num_classes=10)
        assert model.fc2.out_features == 10


# -- create_bank --------------------------------------------------------------

class TestCreateBank:
    def test_no_bank_config_returns_none(self):
        cfg = OmegaConf.create({"method": {"name": "baseline"}})
        bank = create_bank(cfg, num_classes=3)
        assert bank is None

    def test_static_bank(self):
        cfg = OmegaConf.create({
            "bank": {
                "name": "static",
                "capacity_per_class": 100,
                "seed": 7,
            }
        })
        bank = create_bank(cfg, num_classes=3)
        assert isinstance(bank, StaticReplayBank)

    def test_ed_gb_bank(self):
        cfg = OmegaConf.create({
            "bank": {
                "name": "ed_gb",
                "capacity_per_class": 200,
                "seed": 42,
            }
        })
        bank = create_bank(cfg, num_classes=3)
        assert isinstance(bank, ExposureDebtGhostBank)

    def test_unknown_bank_returns_none(self):
        cfg = OmegaConf.create({
            "bank": {"name": "unknown"}
        })
        bank = create_bank(cfg, num_classes=3)
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

    def test_ed_gb(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "ed_gb",
                "retrieval_budget": 8,
                "warmup_steps": 0,
            }
        })
        method = create_method(cfg)
        assert isinstance(method, EDGBMethod)

    def test_focal_loss(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "focal_loss",
                "alpha": 0.25,
                "gamma": 2.0,
            }
        })
        method = create_method(cfg)
        assert isinstance(method, FocalLossMethod)

    def test_class_balanced(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "class_balanced",
                "beta": 0.999,
            }
        })
        method = create_method(cfg, class_counts=[100, 10, 5])
        assert isinstance(method, ClassBalancedMethod)

    def test_class_balanced_missing_counts_raises(self):
        cfg = OmegaConf.create({
            "method": {
                "name": "class_balanced",
                "beta": 0.999,
            }
        })
        import pytest
        with pytest.raises(ValueError, match="class_counts"):
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

    def test_concrete_subclass(self):
        class MinimalRunner(AbstractRunner):
            def compose_configs(self):
                return []

        runner = MinimalRunner()
        assert runner.overrides == []
        assert runner.run() == []
