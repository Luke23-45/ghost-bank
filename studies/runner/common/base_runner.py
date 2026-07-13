from __future__ import annotations

from abc import ABC, abstractmethod

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from src.bank.core.base import AbstractGhostBank
from src.bank.strategies import StaticReplayBank, ExposureDebtGhostBank
from src.data.synthetic import SyntheticDataModule, SyntheticConfig
from src.loss import FocalLoss, ClassBalancedLoss
from src.methods import (
    BaselineMethod,
    ClassBalancedMethod,
    EDGBMethod,
    FocalLossMethod,
    Method,
    StaticBankMethod,
)
from src.models import MLPClassifier
from src.training import (
    DebtCurveLogger,
    ExposureTrackerCallback,
    GhostBankLightningModule,
    GhostBankProgressBar,
)
from studies.output import OutputManager
from studies.runner.common.path_utils import get_config_dir


class AbstractRunner(ABC):
    """Template method for experiment runners.

    Subclasses implement ``compose_configs()`` to produce a list of
    ``(config, run_name)`` pairs.  Each pair is executed by ``run()``
    with a fresh ``OutputManager``.
    """

    def __init__(self, overrides: list[str] | None = None) -> None:
        self.overrides = overrides or []

    @abstractmethod
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        """Return (config, optional run_name) pairs to execute."""
        ...

    def run(self) -> list[dict]:
        """Execute all composed configs and return per-run metrics."""
        configs = list(self.compose_configs())
        all_metrics: list[dict] = []
        for cfg, run_name in tqdm(configs, desc="Experiment sweep", unit="run"):
            mgr = OutputManager(
                experiment=cfg.runner.experiment_name,
                base_dir=cfg.output.base_dir,
            )
            mgr.initialize()
            mgr.save_config(OmegaConf.to_yaml(cfg))
            try:
                metrics = run_experiment(cfg, output_manager=mgr)
                all_metrics.append(metrics)
            except BaseException:
                mgr.fail()
                raise
        return all_metrics


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_datamodule(cfg: DictConfig) -> SyntheticDataModule:
    """Create a data module from a Hydra config."""
    dc = cfg.data
    if dc.type == "synthetic":
        config = SyntheticConfig(
            seed=dc.seed,
            imbalance_ratio=dc.imbalance_ratio,
            majority_train=dc.majority_train,
            test_per_class=dc.test_per_class,
            batch_size=dc.batch_size,
        )
        return SyntheticDataModule(config)
    raise ValueError(f"Unsupported data type: {dc.type}")


def create_model(cfg: DictConfig, num_classes: int) -> MLPClassifier:
    """Create a model from a Hydra config."""
    mc = cfg.model
    if mc.type == "mlp":
        return MLPClassifier(
            input_dim=mc.input_dim,
            hidden_dim=mc.hidden_dim,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported model type: {mc.type}")


def create_bank(cfg: DictConfig, num_classes: int) -> AbstractGhostBank | None:
    """Create a ghost bank from a Hydra config, or return None."""
    if "bank" not in cfg:
        return None
    bc = cfg.bank
    exclude = list(bc.get("exclude_classes", [0]))
    if bc.name == "static":
        return StaticReplayBank(num_classes, bc.capacity_per_class, bc.seed, exclude_classes=exclude)
    if bc.name == "ed_gb":
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

    if name == "focal_loss":
        return FocalLossMethod(FocalLoss(alpha=mc.alpha, gamma=mc.gamma))

    if name == "class_balanced":
        if class_counts is None:
            raise ValueError("class_counts required for class_balanced method")
        return ClassBalancedMethod(
            ClassBalancedLoss(beta=mc.beta),
            class_counts=class_counts,
        )

    raise ValueError(f"Unsupported method: {name}")


def create_pl_module(
    model: MLPClassifier,
    method: Method,
    cfg: DictConfig,
    bank: AbstractGhostBank | None = None,
    num_classes: int | None = None,
) -> GhostBankLightningModule:
    """Create a PL LightningModule from components."""
    return GhostBankLightningModule(
        model=model,
        method=method,
        bank=bank,
        learning_rate=cfg.training.learning_rate,
        num_classes=num_classes,
        optimizer_name=cfg.training.get("optimizer", "sgd"),
    )


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: DictConfig,
    output_manager: OutputManager,
) -> dict:
    """Run a single experiment end-to-end.

    Expects ``output_manager`` to already be initialized with config saved.
    """
    pl.seed_everything(cfg.data.get("seed", 13))

    datamodule = create_datamodule(cfg)
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    num_classes = train_dataset.num_classes
    class_counts = train_dataset.class_counts

    model = create_model(cfg, num_classes=num_classes)
    bank = create_bank(cfg, num_classes=num_classes)
    method = create_method(cfg, class_counts=class_counts)
    pl_module = create_pl_module(model, method, cfg, bank=bank, num_classes=num_classes)

    callbacks: list[pl.Callback] = [
        GhostBankProgressBar(
            refresh_rate=cfg.training.get("progress_refresh_rate", 1),
            leave=True,
        ),
    ]
    if bank is not None:
        callbacks.append(DebtCurveLogger())
    if pl_module.exposure_tracker is not None:
        callbacks.append(ExposureTrackerCallback())

    csv_logger = CSVLogger(
        save_dir=str(output_manager.root),
        name="training_logs",
        version="",
    )

    trainer = pl.Trainer(
        accelerator=getattr(cfg.training, "accelerator", "auto"),
        devices=getattr(cfg.training, "devices", 1),
        precision=getattr(cfg.training, "precision", 32),
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        callbacks=callbacks,
        logger=[csv_logger],
        enable_checkpointing=False,
    )

    try:
        trainer.fit(pl_module, datamodule=datamodule)
        test_results = trainer.test(pl_module, datamodule=datamodule)

        metrics: dict = {
            "method": cfg.method.name,
            "seed": cfg.data.get("seed", 13),
        }
        if test_results:
            metrics["test_acc"] = test_results[0].get("test/acc", 0.0)

        output_manager.write_metrics(metrics)
        output_manager.finalize(
            {
                "test_metrics": test_results,
                "method": cfg.method.name,
            }
        )
        output_manager.complete()
    except BaseException:
        output_manager.fail()
        raise

    return metrics
