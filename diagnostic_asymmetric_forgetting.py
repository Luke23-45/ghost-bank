"""Diagnostic: asymmetric forgetting benchmark.

At epoch 5, REMOVE minority class 2 from training (not a swap).
Only the replay buffer can preserve class 2 accuracy.
PID-CR should detect the rising class 2 loss and allocate more replay,
beating static_bank's uniform allocation.

Also tests: gradual removal, noise injection, and adaptive PID.

Usage:
    python diagnostic_asymmetric_forgetting.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir
from studies.runner.common.path_utils import get_config_dir
from studies.runner.common.base_runner import run_experiment
from studies.output import OutputManager
from pytorch_lightning.callbacks import Callback
import tempfile

SEEDS = [13, 42]


class ClassRemovalCallback(Callback):
    """Remove class 2 from training at a given epoch."""

    def __init__(self, remove_epoch: int = 5, remove_class: int = 2) -> None:
        super().__init__()
        self.remove_epoch = remove_epoch
        self.remove_class = remove_class
        self._removed = False

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == self.remove_epoch and not self._removed:
            ds = trainer.datamodule.train_dataset
            mask = ds._ys != self.remove_class
            ds._xs = ds._xs[mask]
            ds._ys = ds._ys[mask]
            self._removed = True
            print(f"\n  [ClassRemoval] Removed class {self.remove_class} at epoch {self.remove_epoch}")


def run_removal(
    method: str,
    overrides: list[str],
    label: str,
    remove_class: int = 2,
    remove_epoch: int = 5,
) -> dict:
    """Run class removal experiment."""
    cfg_overrides = [
        "+runner=shift",
        f"method={method}",
        f"+bank={method}" if method != "static_bank" else "+bank=static",
        "bank.exclude_classes=[]",
        "runner.seeds=[13,42]",
        "training.enable_progress_bar=false",
        "training.log_every_n_steps=50",
    ] + overrides

    with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
        cfg = compose("config", overrides=cfg_overrides)

    # Inject custom removal callback
    from studies.runner.common.base_runner import _run_single_seed
    import pytorch_lightning as pl

    # Patch: we need the removal callback to be added before training
    # Instead of modifying base_runner, let's add it via the callbacks list
    # We'll modify cfg.training to include a reference to our callback
    # Actually, we need to monkey-patch base_runner or use a different approach

    out_dir = str(Path(tempfile.gettempdir()) / "forget_diag")
    mgr = OutputManager(label, out_dir)
    mgr.initialize()
    mgr.save_config("")
    try:
        metrics = run_experiment(cfg, output_manager=mgr)
        return metrics
    except Exception as e:
        print(f"  ERROR: {e}")
        mgr.fail()
        return {}


# New approach: override runners to inject callbacks
def run_custom_experiment(
    overrides: list[str],
    label: str,
    custom_callbacks: list,
) -> dict:
    """Run experiment with custom callbacks injected."""
    import pytorch_lightning as pl
    from studies.runner.common.base_runner import run_experiment as _orig_run

    # We need to monkey-patch _run_single_seed to add custom callbacks
    from studies.runner.common import base_runner

    original_single = base_runner._run_single_seed

    def patched_single(cfg, output_root, seed):
        # Modify cfg to pass custom callback info
        # We'll pass it via a temp context
        return original_single(cfg, output_root, seed)

    # Actually this is getting too complicated. Let me use a simpler approach.
    # I'll modify the shift runner to support removal experiments.
    pass


if __name__ == "__main__":
    print("This diagnostic needs a custom experiment runner.")
    print("Creating a standalone removal experiment script instead.")
