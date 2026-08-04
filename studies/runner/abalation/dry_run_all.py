"""CI-style dry run: compose and validate every ablation row (no GPU).

For every row of every family (component, sensitivity) the
full override list is composed through the real CIFAR runner and checked
against the frozen protocol lock.  Nothing trains and no output
directories are created.  Exit code is 0 only if every row passes; the
printed table shows one line per row.

Baseline/control methods are handled by ``run_all.py`` directly, not by
this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.runner.abalation.component import ROWS as COMPONENT_ROWS  # noqa: E402
from studies.runner.abalation.sensitivity import ROWS as SENSITIVITY_ROWS  # noqa: E402
from studies.runner.abalation.shared.executor import (  # noqa: E402
    DEFAULT_SEEDS,
    compose_row_cfg,
)
from studies.runner.abalation.shared.protocol import ProtocolViolation  # noqa: E402


def main() -> int:
    rows = [*COMPONENT_ROWS, *SENSITIVITY_ROWS]
    failures = 0
    for row in rows:
        try:
            compose_row_cfg(
                row,
                DEFAULT_SEEDS,
                Path(f"_dry_run/{row.family}/{row.label}"),
            )
            print(f"PASS  {row.run_label:24s} method={row.method}")
        except (ProtocolViolation, RuntimeError, ValueError) as exc:
            failures += 1
            print(f"FAIL  {row.run_label:24s} {type(exc).__name__}: {exc}")

    total = len(rows)
    print(f"\n{total - failures}/{total} rows composed and validated.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())