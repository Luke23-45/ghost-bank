"""Comprehensive PID-CR diagnostic sweep.

Tests multiple hypotheses in parallel:
  1. Bank-probe for absent classes (vs default PID-CR)
  2. Temperature sweep
  3. PID ablation (P-only, PI, PD, I-only, Full PID)
  4. Multiple shifts (swap back and forth)
  5. Different swap pairs

Usage:
    python diagnostic_comprehensive_sweep.py
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
import tempfile

SEEDS = [13, 42]


def run_shift(
    method: str,
    overrides: list[str],
    label: str,
) -> dict:
    """Run a shift experiment with given overrides."""
    base_overrides = [
        "+runner=shift",
        f"method={method}",
        f"+bank={method}" if method != "static_bank" else "+bank=static",
        "bank.exclude_classes=[]",
        "runner.seeds=[13,42]",
        "training.enable_progress_bar=false",
        "training.log_every_n_steps=50",
    ]
    cfg_overrides = base_overrides + overrides

    with initialize_config_dir(config_dir=get_config_dir(), version_base=None):
        cfg = compose("config", overrides=cfg_overrides)

    out_dir = str(Path(tempfile.gettempdir()) / "comp_diag")
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


def print_result(label: str, metrics: dict):
    bal = metrics.get("test/balanced_acc_mean", "N/A")
    bal_s = metrics.get("test/balanced_acc_std", "N/A")
    rec = metrics.get("test/minority_recall_mean", "N/A")
    rec_s = metrics.get("test/minority_recall_std", "N/A")
    f1 = metrics.get("test/macro_f1_mean", "N/A")
    f1_s = metrics.get("test/macro_f1_std", "N/A")
    print(f"  {label:>50}: balanced_acc = {bal:>6.2f} ± {bal_s:<5.2f}  "
          f"minority_recall = {rec:>6.2f} ± {rec_s:<5.2f}  "
          f"macro_f1 = {f1:>6.2f} ± {f1_s:<5.2f}")


if __name__ == "__main__":
    print("=" * 120)
    print("COMPREHENSIVE PID-CR DIAGNOSTIC SWEEP")
    print("=" * 120)

    results: list[tuple[str, dict]] = []

    # ===== PART 1: Baselines =====
    print("\n--- PART 1: BASELINES ---")
    for method, label in [("static_bank", "static_bank"),
                          ("pid_gb", "pid_gb_default"),
                          ("ed_gb", "ed_gb_default")]:
        m = run_shift(method, [], label)
        print_result(label, m)
        results.append((label, m))

    # ===== PART 2: Bank-probe for absent classes =====
    print("\n--- PART 2: BANK-PROBE FOR ABSENT CLASSES ---")
    for val in [False, True]:
        label = f"pid_gb_eval_absent={val}"
        m = run_shift("pid_gb", [f"method.eval_absent_classes={val}"], label)
        print_result(label, m)
        results.append((label, m))

    # ===== PART 3: Temperature sweep =====
    print("\n--- PART 3: TEMPERATURE SWEEP ---")
    for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
        label = f"pid_gb_temp={T}"
        m = run_shift("pid_gb", [f"method.temperature={T}"], label)
        print_result(label, m)
        results.append((label, m))

    # ===== PART 4: PID ablation =====
    print("\n--- PART 4: PID ABLATION ---")
    configs = [
        ("pid_gb_P_only",  ["method.K_p=1.0", "method.K_i=0.0", "method.K_d=0.0"]),
        ("pid_gb_I_only",  ["method.K_p=0.0", "method.K_i=0.1", "method.K_d=0.0"]),
        ("pid_gb_PI",      ["method.K_p=1.0", "method.K_i=0.1", "method.K_d=0.0"]),
        ("pid_gb_PD",      ["method.K_p=1.0", "method.K_i=0.0", "method.K_d=0.5"]),
        ("pid_gb_FullPID", ["method.K_p=1.0", "method.K_i=0.1", "method.K_d=0.5"]),
    ]
    for label, overrides in configs:
        m = run_shift("pid_gb", overrides, label)
        print_result(label, m)
        results.append((label, m))

    # ===== PART 5: PID gain sweep =====
    print("\n--- PART 5: PID GAIN SWEEP ---")
    for K_p in [0.5, 1.0, 2.0]:
        for K_i in [0.05, 0.1, 0.2]:
            label = f"pid_gb_Kp={K_p}_Ki={K_i}"
            m = run_shift("pid_gb", [
                f"method.K_p={K_p}",
                f"method.K_i={K_i}",
                "method.K_d=0.5",
            ], label)
            print_result(label, m)
            results.append((label, m))

    # ===== SUMMARY =====
    print("\n" + "=" * 120)
    print("SUMMARY (sorted by balanced_acc)")
    print("=" * 120)
    sorted_results = sorted(results, key=lambda x: x[1].get("test/balanced_acc_mean", 0), reverse=True)
    for i, (label, m) in enumerate(sorted_results, 1):
        bal = m.get("test/balanced_acc_mean", 0)
        bal_s = m.get("test/balanced_acc_std", 0)
        rec = m.get("test/minority_recall_mean", 0)
        rec_s = m.get("test/minority_recall_std", 0)
        print(f"  {i:>2}. {label:>50}: {bal:>6.2f} ± {bal_s:<5.2f}  (minority_recall: {rec:>6.2f} ± {rec_s:<5.2f})")
