"""Stress test runner — edge-case conditions for robustness.

Usage:
    python studies/runner/stress_test/run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from hydra import compose, initialize_config_dir

from studies.runner.common.base_runner import AbstractRunner
from studies.runner.common.path_utils import get_config_dir
from omegaconf import DictConfig


class StressTestRunner(AbstractRunner):
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        BASE_OVERRIDES = ["+runner=stress_test", "+bank=ed_gb", "method=ed_gb"]

        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            base_cfg = compose("config", overrides=self.overrides + BASE_OVERRIDES)

        pairs: list[tuple[DictConfig, str | None]] = []
        for test in base_cfg.runner.tests:
            test_overrides = [
                f"{k}={v}" for k, v in test.items() if k != "name"
            ]
            with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
                cfg = compose(
                    "config",
                    overrides=self.overrides + BASE_OVERRIDES + test_overrides,
                )
            pairs.append((cfg, test.name))
        return pairs


if __name__ == "__main__":
    runner = StressTestRunner()
    runner.run()
