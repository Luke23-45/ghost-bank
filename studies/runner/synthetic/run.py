"""Synthetic runner — single controlled experiment with Gaussian data.

Usage:
    python studies/runner/synthetic/run.py
    python studies/runner/synthetic/run.py method=ed_gb +bank=ed_gb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from hydra import compose, initialize_config_dir

from studies.runner.common.base_runner import AbstractRunner
from studies.runner.common.path_utils import get_config_dir
from omegaconf import DictConfig


class SyntheticRunner(AbstractRunner):
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            cfg = compose("config", overrides=self.overrides + ["+runner=synthetic"])
        return [(cfg, None)]


if __name__ == "__main__":
    runner = SyntheticRunner()
    runner.run()
