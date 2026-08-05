"""Sensitivity ablation family: s1..s4.

Resource/replay axes only (the method is a memory technique; the budget
and per-step replay curves are the essential sensitivity figures):

- memory budget:    500 / 4000 (anchor 2000)
- retrieval budget: 32 / 128  (anchor 64)

KD weight/temperature are deliberately not swept: the KD mechanism is
already isolated on/off by component row a1_no_kd, so tuning curves add
no conclusion.  For each axis the report adds a monotonicity check
against the anchor: forgetting must decrease (and avg_acc increase) as
the knob (budget / retrieval) grows.  Violations are flagged, never
silently ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.runner.abalation.shared.executor import RunOutcome, run_family_cli  # noqa: E402
from studies.runner.abalation.shared.report import (  # noqa: E402
    ACC_KEY,
    FORGET_KEY,
    aggregate_stats,
    fmt,
    mean_std,
    monotonic_violations,
)
from studies.runner.abalation.shared.rows import AblationRow, UNIFORM_REF_BASE  # noqa: E402

FAMILY = "sensitivity"

ROWS: list[AblationRow] = [
    AblationRow(
        label="s1_budget500",
        family=FAMILY,
        method="uniform_herding",
        overrides=[*UNIFORM_REF_BASE, "data.memory_total=500"],
        hypothesis="Memory budget 2000 -> 500 exemplars.",
        expected_direction="decrease",
    ),
    AblationRow(
        label="s2_budget4000",
        family=FAMILY,
        method="uniform_herding",
        overrides=[*UNIFORM_REF_BASE, "data.memory_total=4000"],
        hypothesis="Memory budget 2000 -> 4000 exemplars.",
        expected_direction="increase",
    ),
    AblationRow(
        label="s3_retr32",
        family=FAMILY,
        method="uniform_herding",
        overrides=[*UNIFORM_REF_BASE, "method.retrieval_budget=32"],
        hypothesis="Per-step replay 64 -> 32 exemplars.",
        expected_direction="decrease",
    ),
    AblationRow(
        label="s4_retr128",
        family=FAMILY,
        method="uniform_herding",
        overrides=[*UNIFORM_REF_BASE, "method.retrieval_budget=128"],
        hypothesis="Per-step replay 64 -> 128 exemplars.",
        expected_direction="increase",
    ),
]

# x-axis -> (x label, row label or None for the anchor) in increasing x order.
_AXES: dict[str, list[tuple[str, str | None]]] = {
    "memory_budget": [
        ("500", "s1_budget500"),
        ("2000 (ref)", None),
        ("4000", "s2_budget4000"),
    ],
    "retrieval_budget": [
        ("32", "s3_retr32"),
        ("64 (ref)", None),
        ("128", "s4_retr128"),
    ],
}


_AGGREGATE_KEY: dict[str, str] = {ACC_KEY: "avg_acc", FORGET_KEY: "forgetting"}


def _mean_by_label(
    outcomes: list[RunOutcome], ref_seeds: dict[int, dict[str, float]], key: str
) -> dict[str, float]:
    """Per-label metric means.

    ``aggregate_stats`` keys are ``avg_acc`` / ``forgetting`` while
    ``ACC_KEY`` / ``FORGET_KEY`` are the raw dot-paths used inside the
    per-seed metric dicts; map between the two namespaces here.
    """
    values: dict[str, float] = {}
    agg_key = _AGGREGATE_KEY[key]
    for outcome in outcomes:
        stats = aggregate_stats(outcome.aggregated)
        if stats:
            values[outcome.row.label] = stats[agg_key][0]
    if ref_seeds:
        ref_mean, _ = mean_std(
            [float(metrics[key]) for metrics in ref_seeds.values() if key in metrics]
        )
        values[None] = ref_mean
    return values


def _axis_section(
    title: str,
    axis: list[tuple[str, str | None]],
    acc_values: dict[str, float],
    forget_values: dict[str, float],
) -> tuple[str, str]:
    lines = [
        "| x | avg_acc | forgetting |",
        "|---|---------|------------|",
    ]
    for x_label, row_label in axis:
        acc = acc_values.get(row_label) if row_label is not None else acc_values.get(None)
        forget = forget_values.get(row_label) if row_label is not None else forget_values.get(None)
        lines.append(
            f"| {x_label} | {fmt(acc) if acc is not None else 'n/a'} | "
            f"{fmt(forget) if forget is not None else 'n/a'} |"
        )

    acc_points = [(x, acc_values.get(label if label else None)) for x, label in axis]
    forget_points = [(x, forget_values.get(label if label else None)) for x, label in axis]
    acc_violations = monotonic_violations(acc_points, higher_better=True)
    forget_violations = monotonic_violations(forget_points, higher_better=False)
    flags = []
    if acc_violations:
        flags.append(f"avg_acc not increasing: {'; '.join(acc_violations)}")
    if forget_violations:
        flags.append(f"forgetting not decreasing: {'; '.join(forget_violations)}")
    if not flags:
        flags.append("no monotonicity violations")
    return title, "\n".join(lines) + "\n\nFlags: " + "; ".join(flags)


def extra_sections(
    outcomes: list[RunOutcome],
    ref_seeds: dict[int, dict[str, float]],
) -> list[tuple[str, str]]:
    acc = _mean_by_label(outcomes, ref_seeds, ACC_KEY)
    forget = _mean_by_label(outcomes, ref_seeds, FORGET_KEY)
    return [
        _axis_section(f"Monotonicity: {axis}", points, acc, forget)
        for axis, points in _AXES.items()
    ]


def main() -> int:
    return run_family_cli(FAMILY, ROWS, extra_sections=extra_sections)


if __name__ == "__main__":
    sys.exit(main())