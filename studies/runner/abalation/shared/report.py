"""Family report generation for the ablation harness.

Every family script ends with a machine-readable trail in
``<base>/<family>/tables/``:

- ``<family>.md``   human-readable paper table (mean +- std, deltas vs
                    the locked reference, effect flags).
- ``<family>.csv``  same table, one row per ablation row.
- ``<family>.json`` structured data for downstream analysis.
- ``provenance.json`` reproducibility record: git commit/dirty, seeds,
  locked-schema sha256, per-row resolved-config fingerprints and the
  locked reference anchor.

All deltas are computed per seed against the reference run's per-seed
metrics (never mean-vs-mean), so the reported uncertainty is honest.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

from .executor import RunOutcome
from .protocol import REFERENCE_SHA256

ACC_KEY = "test/avg_acc"
FORGET_KEY = "test/forgetting"

# Reference anchor (published run debug_run/final_run/.../20260803_143548).
REF_ACC_MEAN = 0.4499
REF_ACC_STD = 0.0100
REF_FORGET_MEAN = 0.1385
REF_FORGET_STD = 0.0044

# |mean delta| below this is reported as "no effect" (~2x the anchor
# avg_acc std; smaller than any cross-config gap observed so far).
EFFECT_MIN_DELTA = 0.02


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var**0.5


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None or x != x:  # NaN guard (x != x)
        return "n/a"
    return f"{x:.{digits}f}"


def fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None or mean != mean or std is None or std != std:
        return "n/a"
    return f"{fmt(mean)} +- {fmt(std)}"


def aggregate_stats(aggregated: dict | None) -> dict[str, tuple[float, float]] | None:
    """Extract ``(mean, std)`` pairs for avg_acc / forgetting."""
    if not aggregated:
        return None

    def _pair(base: str) -> tuple[float, float] | None:
        mean, std = aggregated.get(f"{base}_mean"), aggregated.get(f"{base}_std")
        if mean is None or std is None:
            return None
        return float(mean), float(std)

    acc = _pair(ACC_KEY)
    forget = _pair(FORGET_KEY)
    if acc is None or forget is None:
        return None
    return {"avg_acc": acc, "forgetting": forget}


def per_seed_map(data: dict | None) -> dict[int, dict[str, float]]:
    """Index ``per_seed_metrics`` (from final_results.json) by seed."""
    if not data:
        return {}
    return {int(m["seed"]): m for m in data.get("per_seed_metrics", [])}


def delta_stats(
    row_seeds: dict[int, dict[str, float]],
    ref_seeds: dict[int, dict[str, float]],
    key: str,
) -> tuple[float, float]:
    """Per-seed delta (row - reference) mean and std over shared seeds."""
    common = sorted(set(row_seeds) & set(ref_seeds))
    if not common:
        return float("nan"), float("nan")
    deltas = [float(row_seeds[s][key]) - float(ref_seeds[s][key]) for s in common]
    return mean_std(deltas)


# ---------------------------------------------------------------------------
# Effect flags
# ---------------------------------------------------------------------------


def effect_label(
    delta_mean: float | None, expected_direction: str, key: str = ACC_KEY
) -> str:
    """Classify the row's effect against its declared hypothesis.

    ``delta_mean`` is the per-seed mean delta vs the reference.  For
    avg_acc positive is better, for forgetting negative is better.  The
    label reports consistency with the row's ``expected_direction``.
    """
    if delta_mean is None or delta_mean != delta_mean:
        return "n/a"
    if abs(delta_mean) < EFFECT_MIN_DELTA:
        return "~ (no effect)"
    improved = delta_mean > 0 if key == ACC_KEY else delta_mean < 0
    if expected_direction == "neutral":
        return "up" if improved else "down"
    consistent = improved == (expected_direction == "increase")
    return "H+ (supports)" if consistent else "H- (contradicts)"


# ---------------------------------------------------------------------------
# Monotonicity checks (sensitivity family)
# ---------------------------------------------------------------------------


def monotonic_violations(
    points: list[tuple[str, float | None]],
    *,
    higher_better: bool,
) -> list[str]:
    """Return step descriptions where the series moves against its trend.

    ``points`` is a list of ``(x_label, y_value)`` sorted by x; entries
    with ``None`` (missing runs) are skipped.
    """
    usable = [(label, value) for label, value in points if value is not None]
    violations: list[str] = []
    for i in range(1, len(usable)):
        prev_label, prev = usable[i - 1]
        label, value = usable[i]
        if higher_better:
            ok = value >= prev
        else:
            ok = value <= prev
        if not ok:
            violations.append(f"{prev_label} -> {label}")
    return violations


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _row_row(outcome: RunOutcome, ref_seeds: dict[int, dict[str, float]]) -> list[str]:
    stats = aggregate_stats(outcome.aggregated)
    acc_mean, acc_std = stats["avg_acc"] if stats else (None, None)
    forget_mean, forget_std = stats["forgetting"] if stats else (None, None)
    d_acc_mean, d_acc_std = delta_stats(
        per_seed_map(outcome.per_seed and {"per_seed_metrics": outcome.per_seed}),
        ref_seeds,
        ACC_KEY,
    )
    return [
        outcome.row.label,
        outcome.row.method,
        " ".join(outcome.row.overrides) if outcome.row.overrides else "(locked)",
        fmt_pm(acc_mean, acc_std),
        fmt_pm(forget_mean, forget_std),
        fmt_pm(d_acc_mean, d_acc_std) if d_acc_mean == d_acc_mean else "n/a",
        effect_label(d_acc_mean, outcome.row.expected_direction, ACC_KEY),
        outcome.status,
    ]


_MD_HEADER = [
    "row",
    "method",
    "overrides",
    "avg_acc",
    "forgetting",
    "delta_acc vs ref",
    "effect",
    "status",
]
_CSV_HEADER = [
    "row",
    "family",
    "method",
    "overrides",
    "avg_acc_mean",
    "avg_acc_std",
    "forgetting_mean",
    "forgetting_std",
    "delta_acc_mean",
    "delta_acc_std",
    "effect",
    "status",
    "run_dir",
]


def build_markdown(
    family: str,
    outcomes: list[RunOutcome],
    ref_seeds: dict[int, dict[str, float]],
    *,
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    lines = [
        f"# {family} family — ablation table",
        "",
        f"Built at {datetime.now().isoformat(timespec='seconds')}. "
        "Deltas are per-seed vs the locked reference run (when reference "
        "data exists).",
        "",
        "| " + " | ".join(_MD_HEADER) + " |",
        "|" + "---|" * len(_MD_HEADER),
    ]
    for outcome in outcomes:
        lines.append("| " + " | ".join(_row_row(outcome, ref_seeds)) + " |")
    for title, body in extra_sections or []:
        lines += ["", f"## {title}", "", body]
    lines += [
        "",
        "Status legend: RUN = executed now, SKIP = reused a completed run, "
        "READY = composed+validated only (dry-run), MISSING = no run found.",
    ]
    return "\n".join(lines)


def build_csv(
    family: str, outcomes: list[RunOutcome], ref_seeds: dict[int, dict[str, float]]
) -> str:
    rows = [",".join(_CSV_HEADER)]
    for outcome in outcomes:
        stats = aggregate_stats(outcome.aggregated)
        acc_mean, acc_std = stats["avg_acc"] if stats else (None, None)
        forget_mean, forget_std = stats["forgetting"] if stats else (None, None)
        d_acc_mean, d_acc_std = delta_stats(
            per_seed_map(outcome.per_seed and {"per_seed_metrics": outcome.per_seed}),
            ref_seeds,
            ACC_KEY,
        )
        rows.append(
            ",".join(
                [
                    outcome.row.label,
                    outcome.row.family,
                    outcome.row.method,
                    '"' + " ".join(outcome.row.overrides) + '"',
                    fmt(acc_mean, 6),
                    fmt(acc_std, 6),
                    fmt(forget_mean, 6),
                    fmt(forget_std, 6),
                    fmt(d_acc_mean, 6),
                    fmt(d_acc_std, 6),
                    effect_label(d_acc_mean, outcome.row.expected_direction, ACC_KEY),
                    outcome.status,
                    outcome.run_dir or "",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def build_json(outcomes: list[RunOutcome]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        stats = aggregate_stats(outcome.aggregated)
        entry: dict[str, Any] = {
            "label": outcome.row.label,
            "family": outcome.row.family,
            "method": outcome.row.method,
            "overrides": outcome.row.overrides,
            "hypothesis": outcome.row.hypothesis,
            "expected_direction": outcome.row.expected_direction,
            "status": outcome.status,
            "run_dir": outcome.run_dir,
            "resolved_sha256": outcome.resolved_sha256,
        }
        if stats:
            entry["avg_acc"] = {"mean": stats["avg_acc"][0], "std": stats["avg_acc"][1]}
            entry["forgetting"] = {
                "mean": stats["forgetting"][0],
                "std": stats["forgetting"][1],
            }
        if outcome.per_seed:
            entry["per_seed"] = outcome.per_seed
        rows.append(entry)
    return {"rows": rows}


def provenance(
    family: str,
    seeds: list[int],
    base_dir: str,
    outcomes: list[RunOutcome],
    git_commit: str | None,
    git_dirty: bool,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "family": family,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "base_dir": base_dir,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "reference_lock_sha256": REFERENCE_SHA256,
        "reference_anchor": {
            "avg_acc": {"mean": REF_ACC_MEAN, "std": REF_ACC_STD},
            "forgetting": {"mean": REF_FORGET_MEAN, "std": REF_FORGET_STD},
        },
        "rows": [
            {
                "label": outcome.row.label,
                "method": outcome.row.method,
                "overrides": outcome.row.overrides,
                "status": outcome.status,
                "run_dir": outcome.run_dir,
                "resolved_sha256": outcome.resolved_sha256,
            }
            for outcome in outcomes
        ],
    }
    if extra:
        doc.update(extra)
    return doc


def write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# Outcome comparison helper (e.g. c1 vs c2 within a family)
# ---------------------------------------------------------------------------


def compare_outcomes(
    a: RunOutcome | None,
    b: RunOutcome | None,
    key: str = ACC_KEY,
) -> tuple[float, float]:
    """Per-seed ``a - b`` delta (mean, std) for two outcomes."""
    if a is None or b is None:
        return float("nan"), float("nan")
    return delta_stats(
        per_seed_map(a.per_seed and {"per_seed_metrics": a.per_seed}),
        per_seed_map(b.per_seed and {"per_seed_metrics": b.per_seed}),
        key,
    )


def per_seed_by_label(outcomes: Iterable[RunOutcome]) -> dict[str, dict[str, float]]:
    return {o.row.label: per_seed_map(o.per_seed and {"per_seed_metrics": o.per_seed}) for o in outcomes}