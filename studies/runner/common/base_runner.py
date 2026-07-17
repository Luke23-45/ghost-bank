from __future__ import annotations

from abc import ABC, abstractmethod

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from src.bank.core.base import AbstractGhostBank
from src.bank.strategies import StaticReplayBank, ExposureDebtGhostBank
from src.bank.core.pid_controller import PIDController
from src.data.synthetic import SyntheticDataModule, SyntheticConfig
from src.loss import FocalLoss, ClassBalancedLoss
from src.methods import (
    BaselineMethod,
    ClassBalancedMethod,
    EDGBMethod,
    FocalLossMethod,
    Method,
    PIDGBMethod,
    StaticBankMethod,
)
from src.models import MLPClassifier
from src.training import (
    DebtCurveLogger,
    DistributionShiftCallback,
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
            num_workers=dc.get("num_workers", 0),
            pin_memory=dc.get("pin_memory", False),
            persistent_workers=dc.get("persistent_workers", False),
            prefetch_factor=dc.get("prefetch_factor", 2),
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
    minority_classes: list[int] | None = None,
) -> GhostBankLightningModule:
    """Create a PL LightningModule from components."""
    return GhostBankLightningModule(
        model=model,
        method=method,
        bank=bank,
        learning_rate=cfg.training.learning_rate,
        num_classes=num_classes,
        optimizer_name=cfg.training.get("optimizer", "sgd"),
        minority_classes=minority_classes,
    )


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------


def _run_single_seed(
    cfg: DictConfig,
    output_root: str,
    seed: int,
) -> dict:
    """Run one seed of an experiment and return metrics.

    Does **not** touch the ``OutputManager`` state machine — all
    per-seed bookkeeping is the caller's responsibility.
    """
    cfg.data.seed = seed
    pl.seed_everything(seed)

    datamodule = create_datamodule(cfg)
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    num_classes = train_dataset.num_classes
    class_counts = train_dataset.class_counts

    model = create_model(cfg, num_classes=num_classes)
    bank = create_bank(cfg, num_classes=num_classes)
    method = create_method(cfg, class_counts=class_counts)
    minority_classes = cfg.data.get("minority_classes", None)
    pl_module = create_pl_module(
        model, method, cfg,
        bank=bank,
        num_classes=num_classes,
        minority_classes=minority_classes,
    )

    callbacks: list[pl.Callback] = []
    if cfg.training.get("enable_progress_bar", True):
        callbacks.append(
            GhostBankProgressBar(
                refresh_rate=cfg.training.get("progress_refresh_rate", 1),
                leave=True,
            )
        )
    if bank is not None:
        callbacks.append(DebtCurveLogger())
    if pl_module.exposure_tracker is not None:
        callbacks.append(ExposureTrackerCallback())

    shift_epoch = cfg.runner.get("shift_epoch", None)
    if shift_epoch is not None:
        swap = cfg.runner.get("swap_classes", [0, 2])
        freeze = cfg.runner.get("freeze_bank", False)
        shift_test = cfg.runner.get("shift_test_dataset", False)
        callbacks.append(
            DistributionShiftCallback(
                shift_epoch=shift_epoch,
                swap_classes=tuple(swap),
                freeze_bank=freeze,
                shift_test_dataset=shift_test,
            )
        )

    csv_logger = CSVLogger(
        save_dir=output_root,
        name=f"seed_{seed}",
        version="",
    )

    trainer = pl.Trainer(
        accelerator=getattr(cfg.training, "accelerator", "auto"),
        devices=getattr(cfg.training, "devices", 1),
        precision=getattr(cfg.training, "precision", 32),
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        enable_progress_bar=cfg.training.get("enable_progress_bar", True),
        callbacks=callbacks,
        logger=[csv_logger],
        enable_checkpointing=False,
    )

    trainer.fit(pl_module, datamodule=datamodule)
    test_results = trainer.test(pl_module, datamodule=datamodule)

    metrics: dict = {"method": cfg.method.name, "seed": seed}
    if test_results:
        for key, val in test_results[0].items():
            metrics[str(key)] = val

    return metrics


def _aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean ± std for every numeric metric across seeds."""
    if not all_metrics:
        return {}

    aggregated: dict = {
        "method": all_metrics[0]["method"],
        "num_seeds": len(all_metrics),
    }

    numeric_keys: set[str] = set()
    for m in all_metrics:
        for k, v in m.items():
            if k not in ("method", "seed") and isinstance(v, (int, float)):
                numeric_keys.add(k)

    for key in sorted(numeric_keys):
        values = [m[key] for m in all_metrics if key in m]
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            aggregated[f"{key}_mean"] = mean
            aggregated[f"{key}_std"] = std

    return aggregated


def run_experiment(
    cfg: DictConfig,
    output_manager: OutputManager,
) -> dict:
    """Run an experiment over all configured seeds and return aggregated metrics.

    ``output_manager`` must already be initialised with config saved.
    """
    seeds: list[int] = cfg.runner.get("seeds", [13])
    if not seeds:
        raise ValueError(
            f"runner.seeds is empty for method '{cfg.method.name}'. "
            "At least one seed is required."
        )

    all_metrics: list[dict] = []
    for seed in seeds:
        metrics = _run_single_seed(cfg, str(output_manager.root), seed)
        all_metrics.append(metrics)

    aggregated = _aggregate_metrics(all_metrics)

    try:
        for m in all_metrics:
            output_manager.write_metrics(m, f"seed_{m['seed']}_metrics.json")

        output_manager.write_metrics(aggregated, "aggregated_metrics.csv")
        output_manager.finalize(
            {
                "aggregated": aggregated,
                "per_seed_metrics": all_metrics,
                "method": cfg.method.name,
                "num_seeds": len(seeds),
            }
        )
        output_manager.complete()
    except BaseException:
        output_manager.fail()
        raise

    return aggregated
