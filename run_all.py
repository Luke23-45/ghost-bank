"""Run the CIFAR-100 Class-IL benchmark for all four methods."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

RUNNER = str(ROOT / "studies/runner/cifar100/run.py")

METHODS: list[str] = ["baseline", "static_bank", "ed_gb", "pid_gb"]


def main() -> None:
    results: dict[str, bool] = {}

    try:
        for method in METHODS:
            name = f"cifar100/{method}"
            print(f"\n{'=' * 70}")
            print(f"  [{name}]")
            print(f"{'=' * 70}")
            proc = subprocess.run(
                [sys.executable, RUNNER, f"method={method}"],
                cwd=ROOT,
                env=ENV,
            )
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
