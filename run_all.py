"""Run all major experiments sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**sys.environ, "PYTHONIOENCODING": "utf-8"}

RUNNERS: list[tuple[str, list[str]]] = [
    ("synthetic (ED-GB)",  [str(ROOT / "studies/runner/synthetic/run.py"),         "method=ed_gb", "+bank=ed_gb"]),
    ("baseline_matrix",    [str(ROOT / "studies/runner/baseline_matrix/run.py")]),
    ("ablation",           [str(ROOT / "studies/runner/ablation/run.py")]),
    ("stress_test",        [str(ROOT / "studies/runner/stress_test/run.py")]),
]


def main() -> None:
    results: dict[str, bool] = {}

    try:
        for name, args in RUNNERS:
            print(f"\n{'=' * 70}")
            print(f"  [{name}]")
            print(f"{'=' * 70}")
            proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV)
            results[name] = proc.returncode == 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:4s}  {name}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n  All experiments completed successfully.")
    else:
        print("\n  Some experiments failed (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
