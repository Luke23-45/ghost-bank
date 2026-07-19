from __future__ import annotations

from abc import ABC, abstractmethod

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from src.bank.core.base import AbstractGhostBank
from src.bank.core.exposure import ExposureTracker
from src.bank.core.pid_controller import PIDController
from src.bank.strategies import StaticReplayBank, ExposureDebtGhostBank
from src.data.cifar100 import CIFAR100DataModule, CIFAR100Config
from src.methods import (
    BaselineMethod,
    EDGBMethod,
    Method,
    PIDGBMethod,
    StaticBankMethod,
)
from src.models import ResNet, ResNetConfig
from src.training import (
    ConsoleEpochCallback,
    DebtCurveLogger,
    ExposureTrackerCallback,
    GhostBankLightningModule,
    GhostBankProgressBar,
)
from src.utils.logging import setup_logging
from studies.output import OutputManager
from studies.runner.common.path_utils import get_config_dir


class AbstractRunner(ABC):
    """Template method for experiment runners.

    Subclasses implement ``compose_configs()`` and ``run_experiment()``.
    Each composed config is executed by ``run()`` with a fresh
    ``OutputManager``.
    """

    def __init__(self, overrides: list[str] | None = None) -> None:
        self.overrides = overrides or []

    @abstractmethod
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        """Return (config, optional run_name) pairs to execute."""
        ...

    @abstractmethod
    def run_experiment(
        self,
        cfg: DictConfig,
        output_manager: OutputManager,
    ) -> dict:
        """Run a single experiment config over all configured seeds.

        Subclasses implement the full experiment orchestration
        (multi-task Class-IL loop, multi-seed aggregation, etc.).
        """
        ...

    def run(self) -> list[dict]:
        """Execute all composed configs and return per-run metrics."""
        configs = list(self.compose_configs())
        if configs:
            log_cfg = configs[0][0].training.get("logging", {})
            log_level = log_cfg.get("level", "info")
            setup_logging(level=log_level)
        all_metrics: list[dict] = []
        for cfg, run_name in tqdm(configs, desc="Experiment sweep", unit="run"):
            mgr = OutputManager(
                experiment=cfg.runner.experiment_name,
                base_dir=cfg.output.base_dir,
            )
            mgr.initialize()
            mgr.save_config(OmegaConf.to_yaml(cfg))
            try:
                metrics = self.run_experiment(cfg, output_manager=mgr)
                all_metrics.append(metrics)
            except BaseException:
                mgr.fail()
                raise
        return all_metrics


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_datamodule(cfg: DictConfig) -> CIFAR100DataModule:
    """Create a data module from a Hydra config."""
    dc = cfg.data
    if dc.type == "cifar100":
        config = CIFAR100Config(
            root=dc.root,
            seed=dc.seed,
            batch_size=dc.batch_size,
            num_workers=dc.get("num_workers", 4),
            pin_memory=dc.get("pin_memory", True),
            persistent_workers=dc.get("persistent_workers", True),
            prefetch_factor=dc.get("prefetch_factor", 2),
            num_tasks=dc.get("num_tasks", 10),
            classes_per_task=dc.get("classes_per_task", 10),
            mean=tuple(dc.get("mean", [0.5071, 0.4867, 0.4408])),
            std=tuple(dc.get("std", [0.2675, 0.2565, 0.2761])),
        )
        return CIFAR100DataModule(config)
    raise ValueError(f"Unsupported data type: {dc.type}")


def create_model(cfg: DictConfig, num_classes: int) -> ResNet:
    """Create a model from a Hydra config."""
    mc = cfg.model
    if mc.type == "resnet":
        return ResNet(
            num_classes=num_classes,
            base_filters=mc.get("base_filters", 64),
            dropout=mc.get("dropout", 0.0),
        )
    raise ValueError(f"Unsupported model type: {mc.type}")


def create_bank(cfg: DictConfig, num_classes: int) -> AbstractGhostBank | None:
    """Create a ghost bank from a Hydra config, or return None."""
    if "bank" not in cfg:
        return None
    bc = cfg.bank
    exclude = list(bc.get("exclude_classes", []))
    if bc.name == "static":
        return StaticReplayBank(num_classes, bc.capacity_per_class, bc.seed, exclude_classes=exclude)
    if bc.name == "ed_gb":
        return ExposureDebtGhostBank(num_classes, bc.capacity_per_class, bc.seed, exclude_classes=exclude)
    if bc.name == "pid_gb":
        return ExposureDebtGhostBank(num_classes, bc.capacity_per_class, bc.seed, exclude_classes=exclude)
    return None


def create_method(
    cfg: DictConfig,
    class_counts: list[int] | None = None,
) -> Method:
    """Create a training method from a Hydra config."""
    mc = cfg.method
    name = mc.name

    if name == "baseline":
        return BaselineMethod()

    if name == "static_bank":
        return StaticBankMethod(
            retrieval_budget=mc.retrieval_budget,
            warmup_steps=mc.get("warmup_steps", 0),
        )

    if name == "ed_gb":
        return EDGBMethod(
            retrieval_budget=mc.retrieval_budget,
            warmup_steps=mc.get("warmup_steps", 0),
        )

    if name == "pid_gb":
        method = PIDGBMethod(
            retrieval_budget=mc.retrieval_budget,
            warmup_steps=mc.get("warmup_steps", 0),
            K_p=mc.get("K_p", 1.0),
            K_i=mc.get("K_i", 0.1),
            K_d=mc.get("K_d", 0.5),
            pid_decay=mc.get("pid_decay", 0.99),
            pid_smooth=mc.get("pid_smooth", 0.9),
            temperature=mc.get("temperature", 1.0),
            bank_probe_size=mc.get("bank_probe_size", 16),
            eval_absent_classes=mc.get("eval_absent_classes", True),
            use_class_weights=mc.get("use_class_weights", False),
        )
        # Compute per-class weights from class counts
        if class_counts is not None and method.use_class_weights:
            max_count = max(class_counts)
            method.class_weights = [
                (max_count / c) ** 0.5 if c > 0 else 1.0
                for c in class_counts
            ]
        else:
            method.class_weights = [1.0] * len(class_counts) if class_counts else []
        return method

    raise ValueError(f"Unsupported method: {name}")


def create_pl_module(
    model: ResNet,
    method: Method,
    cfg: DictConfig,
    bank: AbstractGhostBank | None = None,
    num_classes: int | None = None,
    minority_classes: list[int] | None = None,
    exposure_tracker: ExposureTracker | None = None,
    pid_controller: PIDController | None = None,
    train_transform: object | None = None,
    augment_generator: torch.Generator | None = None,
) -> GhostBankLightningModule:
    """Create a PL LightningModule from components."""
    return GhostBankLightningModule(
        model=model,
        method=method,
        bank=bank,
        learning_rate=cfg.training.learning_rate,
        num_classes=num_classes,
        optimizer_name=cfg.training.get("optimizer", "sgd"),
        lr_scheduler=cfg.training.get("lr_scheduler", None),
        momentum=cfg.training.get("momentum", 0.0),
        weight_decay=cfg.training.get("weight_decay", 0.0),
        minority_classes=minority_classes,
        exposure_tracker=exposure_tracker,
        pid_controller=pid_controller,
        train_transform=train_transform,
        augment_generator=augment_generator,
    )
