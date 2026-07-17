"""Distribution shift experiment — compare method recovery after label swap.

Usage:
    python studies/runner/shift_experiment/run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from hydra import compose, initialize_config_dir

from studies.runner.common.base_runner import AbstractRunner
from studies.runner.common.path_utils import get_config_dir
from omegaconf import DictConfig


BANK_MAP = {"static_bank": "static", "ed_gb": "ed_gb", "pid_gb": "pid_gb"}


class ShiftExperimentRunner(AbstractRunner):
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            base_cfg = compose(
                "config", overrides=self.overrides + ["+runner=shift"]
            )

        pairs: list[tuple[DictConfig, str | None]] = []
        for method_name in base_cfg.runner.methods:
            method_overrides = [f"method={method_name}"]
            if method_name in BANK_MAP:
                method_overrides.append(f"+bank={BANK_MAP[method_name]}")
            method_overrides.append("bank.exclude_classes=[]")
            with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
                cfg = compose(
                    "config",
                    overrides=self.overrides + ["+runner=shift"] + method_overrides,
                )
            pairs.append((cfg, method_name))
        return pairs


if __name__ == "__main__":
    runner = ShiftExperimentRunner()
    runner.run()
