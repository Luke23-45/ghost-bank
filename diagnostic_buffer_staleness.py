"""Diagnostic: measure buffer staleness under distribution shift.

Compares pid_gb with and without flushing the buffer at the shift point.
If flush beats no-flush (and approaches static_bank), then buffer
staleness is confirmed as the bottleneck.

Usage: python diagnostic_buffer_staleness.py
"""

import csv
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from hydra import compose, initialize_config_dir
from studies.runner.common.path_utils import get_config_dir
from studies.runner.common.base_runner import run_experiment
from studies.output import OutputManager
import tempfile


def run_shift(method: str, flush: bool, label: str) -> dict:
    """Run shift experiment with optional bank flush."""
    with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
        cfg = compose("config", overrides=[
            "+runner=shift",
            f"method={method}",
            f"+bank={method}" if method != "static_bank" else "+bank=static",
            "bank.exclude_classes=[]",
            "runner.seeds=[13,42,73]",
            f"++runner.flush_bank={str(flush).lower()}",
            "training.enable_progress_bar=false",
            "training.log_every_n_steps=50",
        ])

    out_dir = str(Path(tempfile.gettempdir()) / "staleness_diag")
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


if __name__ == "__main__":
    SEEDS = [13, 42, 73]
    print("=" * 70)
    print("Buffer Staleness Diagnostic")
    print("=" * 70)

    configs = [
        ("static_bank", False, "static_bank_noflush"),
        ("static_bank", True,  "static_bank_flush"),
        ("pid_gb",      False, "pid_gb_noflush"),
        ("pid_gb",      True,  "pid_gb_flush"),
    ]

    for method, flush, label in configs:
        m = run_shift(method, flush, label)
        bal = m.get("test/balanced_acc_mean", "N/A")
        bal_s = m.get("test/balanced_acc_std", "N/A")
        rec = m.get("test/minority_recall_mean", "N/A")
        rec_s = m.get("test/minority_recall_std", "N/A")
        f1 = m.get("test/macro_f1_mean", "N/A")
        f1_s = m.get("test/macro_f1_std", "N/A")
        print(f"\n{label:>20}:")
        print(f"  balanced_acc    = {bal} +/- {bal_s}")
        print(f"  macro_f1        = {f1} +/- {f1_s}")
        print(f"  minority_recall = {rec} +/- {rec_s}")
