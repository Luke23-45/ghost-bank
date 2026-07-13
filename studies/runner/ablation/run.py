"""Ablation runner — parameter sweep over bank/method hyperparameters.

Usage:
    python studies/runner/ablation/run.py
    python studies/runner/ablation/run.py runner.methods='["ed_gb","static_bank"]'
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from hydra import compose, initialize_config_dir

from studies.runner.common.base_runner import AbstractRunner
from studies.runner.common.path_utils import get_config_dir
from omegaconf import DictConfig

METHOD_BANK_MAP: dict[str, str] = {
    "ed_gb": "ed_gb",
    "static_bank": "static",
}


class AblationRunner(AbstractRunner):
    def compose_configs(self) -> list[tuple[DictConfig, str | None]]:
        with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
            base_cfg = compose(
                "config", overrides=self.overrides + ["+runner=ablation"]
            )

        sweep = base_cfg.runner.sweep
        keys = list(sweep.keys())
        values_lists = [list(v) for v in sweep.values()]
        methods: list[str] = list(base_cfg.runner.get("methods", ["ed_gb"]))

        pairs: list[tuple[DictConfig, str | None]] = []
        for method_name in methods:
            bank_name = METHOD_BANK_MAP[method_name]
            base_overrides = [
                "+runner=ablation",
                f"+bank={bank_name}",
                f"method={method_name}",
            ]
            for combo in itertools.product(*values_lists):
                sweep_overrides = [f"{k}={v}" for k, v in zip(keys, combo)]
                run_name = f"{method_name}_" + "_".join(
                    f"{k.rsplit('.', 1)[-1]}_{v}" for k, v in zip(keys, combo)
                )
                with initialize_config_dir(
                    config_dir=get_config_dir(), version_base=None
                ):
                    cfg = compose(
                        "config",
                        overrides=self.overrides
                        + base_overrides
                        + sweep_overrides,
                    )
                pairs.append((cfg, run_name))
        return pairs


if __name__ == "__main__":
    runner = AblationRunner()
    runner.run()
