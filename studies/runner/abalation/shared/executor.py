"""Row executor: compose, validate, run, skip/resume for ablation rows.

Every row runs in-process through the stable ``CIFAR100Runner`` with its
own isolated output tree:

    <base_dir>/<family>/<row_label>/cifar100/<method>/<timestamp>/

so each ablation type gets a dedicated folder under the ablation base,
exactly one row per directory, and never-merged timestamped run dirs.

Row-level resume: a row whose output tree already contains a completed
``results/final_results.json`` is skipped (unless ``--force``); the
family report then reuses the recorded metrics.  ``--report-only`` loads
existing runs without touching anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .rows import AblationRow

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from studies.output import OutputManager  # noqa: E402
from studies.runner.abalation.shared.protocol import (  # noqa: E402
    canonical_fingerprint,
    validate_resolved_config,
)
from studies.runner.cifar100.run import (  # noqa: E402
    BANK_MAP,
    CIFAR100Runner,
)

DEFAULT_SEEDS = [1993, 2023, 42]
DEFAULT_BASE_DIR = "debug_run/ablations"


# ---------------------------------------------------------------------------
# Outcome record
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    row: AblationRow
    status: str  # RUN | SKIP | READY (dry-run) | MISSING (report-only) | FAIL
    run_dir: str | None = None
    aggregated: dict | None = None
    per_seed: list[dict] | None = None
    resolved_sha256: str | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("RUN", "SKIP", "READY")


# ---------------------------------------------------------------------------
# Git / provenance helpers
# ---------------------------------------------------------------------------


def git_state() -> tuple[str | None, bool]:
    """Return (commit_sha, dirty_flag).  ``(None, False)`` if not a repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
        return (commit or None, dirty)
    except BaseException:
        return (None, False)


# ---------------------------------------------------------------------------
# Run discovery / result loading
# ---------------------------------------------------------------------------


def _iter_completed_run_dirs(row_root: Path) -> list[Path]:
    """All run dirs under ``row_root`` with a completed ``final_results.json``."""
    if not row_root.is_dir():
        return []
    return sorted(
        p.parent.parent for p in row_root.rglob("results/final_results.json")
        if p.is_file()
    )


def load_run_results(run_dir: Path) -> dict:
    """Load aggregated + per-seed results from a completed run dir."""
    path = run_dir / "results" / "final_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run_data(row_root: Path) -> dict | None:
    dirs = _iter_completed_run_dirs(row_root)
    if not dirs:
        return None
    return load_run_results(dirs[-1])


# ---------------------------------------------------------------------------
# Composition + validation
# ---------------------------------------------------------------------------


def declared_extras(row: AblationRow) -> list[tuple[str, str]]:
    """Synthetic value-claims so the lock check understands method/bank switches.

    The ``method.name`` (and, for reply-methods, ``bank.name``) of a row
    intentionally differ from the locked reference.  Declaring them here
    makes protocol validation value-verify them instead of treating a
    deliberate method swap as lock drift.
    """
    extras: list[tuple[str, str]] = [("method.name", row.method)]
    bank_target = BANK_MAP.get(row.method)
    if bank_target is not None:
        extras.append(("bank.name", bank_target))
    return extras


def compose_row_cfg(
    row: AblationRow,
    seeds: list[int],
    row_root: Path,
) -> tuple[DictConfig, str | None]:
    """Compose and validate the row's resolved config (no training).

    Raises :class:`ProtocolViolation` if the resolved config diverges
    from the locked reference beyond the row's declared overrides.
    """
    from .protocol import assert_declared_keys_valid

    assert_declared_keys_valid(row.overrides, row.label)
    runner = CIFAR100Runner(
        overrides=[
            f"runner.methods=[{row.method}]",
            *row.overrides,
            f"runner.seeds={list(seeds)}",
        ]
    )
    pairs = runner.compose_configs()
    if len(pairs) != 1:
        raise RuntimeError(
            f"Row {row.label!r}: compose_configs returned {len(pairs)} "
            "configs (expected exactly 1)."
        )
    cfg, run_name = pairs[0]
    cfg.output.base_dir = str(row_root).replace("\\", "/")
    validate_resolved_config(
        cfg, row.overrides, seeds, declared_extra=declared_extras(row)
    )
    return cfg, run_name


# ---------------------------------------------------------------------------
# Row execution
# ---------------------------------------------------------------------------


def _run_one_row(
    row: AblationRow,
    seeds: list[int],
    base_dir: Path,
    *,
    force: bool,
    dry_run: bool,
    report_only: bool,
    verbose: bool = True,
) -> RunOutcome:
    row_root = base_dir / row.family / row.label
    existing = _iter_completed_run_dirs(row_root)

    if report_only:
        if not existing:
            return RunOutcome(row, status="MISSING", message="no completed run found")
        data = load_run_results(existing[-1])
        return RunOutcome(
            row,
            status="SKIP",
            run_dir=str(existing[-1]),
            aggregated=data.get("aggregated"),
            per_seed=data.get("per_seed_metrics"),
            message="report-only: loaded existing run",
        )

    if existing and not force:
        data = load_run_results(existing[-1])
        if verbose:
            print(f"[{row.run_label}] SKIP (completed run exists: {existing[-1]})")
        return RunOutcome(
            row,
            status="SKIP",
            run_dir=str(existing[-1]),
            aggregated=data.get("aggregated"),
            per_seed=data.get("per_seed_metrics"),
            message="resumed from existing completed run",
        )

    cfg, run_name = compose_row_cfg(row, seeds, row_root)
    sha = canonical_fingerprint(cfg)
    if dry_run:
        if verbose:
            print(
                f"[{row.run_label}] READY (dry-run: composed + validated, "
                f"sha={sha[:12]}, method={run_name})"
            )
        return RunOutcome(row, status="READY", resolved_sha256=sha)

    mgr = OutputManager(
        experiment=cfg.runner.experiment_name,
        base_dir=cfg.output.base_dir,
        run_name=run_name,
    )
    mgr.initialize()
    mgr.save_config(OmegaConf.to_yaml(cfg))
    if verbose:
        print(f"[{row.run_label}] RUN -> {mgr.root}")
    try:
        runner_execute(cfg, mgr)
    except BaseException:
        mgr.fail()
        raise
    # Reload the persisted results so the outcome carries both aggregated
    # and per-seed metrics exactly as recorded (matching the SKIP path).
    record = load_run_results(Path(mgr.root))
    return RunOutcome(
        row,
        status="RUN",
        run_dir=mgr.root,
        aggregated=record.get("aggregated"),
        per_seed=record.get("per_seed_metrics"),
        resolved_sha256=sha,
    )


def runner_execute(cfg: DictConfig, mgr: OutputManager) -> dict:
    """Run the full multi-seed experiment for one composed config."""
    runner = CIFAR100Runner(overrides=[])
    return runner.run_experiment(cfg, output_manager=mgr)


# ---------------------------------------------------------------------------
# CLI plumbing shared by the family scripts
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seed list (default: 1993,2023,42).",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Output base directory (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose and validate every row config; run nothing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even if a completed run already exists.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only assemble the family report from existing runs.",
    )
    parser.add_argument(
        "--rows",
        default=None,
        help="Comma-separated row labels to run (subset of the family).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Refuse to run on a dirty git tree.",
    )


def parse_seeds(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def select_rows(rows: list[AblationRow], subset: str | None) -> list[AblationRow]:
    if not subset:
        return rows
    wanted = {label.strip() for label in subset.split(",") if label.strip()}
    missing = wanted - {row.label for row in rows}
    if missing:
        raise SystemExit(f"Unknown row labels in this family: {sorted(missing)}")
    return [row for row in rows if row.label in wanted]


def fingerprint_existing_run(run_dir: str | None) -> str | None:
    """Recompute the resolved-config fingerprint of an existing run for provenance."""
    if not run_dir:
        return None
    cfg_path = Path(run_dir) / "configs" / "resolved_config.yaml"
    if not cfg_path.is_file():
        return None
    from omegaconf import OmegaConf

    return canonical_fingerprint(OmegaConf.load(str(cfg_path)))


def run_family_cli(
    family: str,
    rows: list[AblationRow],
    *,
    gate=None,
    extra_sections=None,
) -> int:
    """Shared CLI entry point for every per-family runner script.

    Parameters
    ----------
    gate:
        Optional ``callable(list[RunOutcome]) -> tuple[bool, list[str]]``
        enforced as a hard gate.
    extra_sections:
        Optional ``callable(list[RunOutcome]) -> list[(title, body_md)]``
        appended under the row table (used by sensitivity).
    """
    from .report import (
        build_csv,
        build_json,
        build_markdown,
        per_seed_map,
        provenance,
        write_json,
    )

    parser = argparse.ArgumentParser(
        description=f"Run the {family} ablation family on 3 seeds per row.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    selected = select_rows(rows, args.rows)
    commit, dirty = git_state()
    if args.strict and dirty:
        print(f"[strict] refusing to run on a dirty git tree ({commit}).")
        return 2

    base = Path(args.base_dir)
    outcomes: list[RunOutcome] = []
    for row in selected:
        outcome = _run_one_row(
            row,
            seeds,
            base,
            force=args.force,
            dry_run=args.dry_run or args.report_only,
            report_only=args.report_only,
        )
        if outcome.resolved_sha256 is None and outcome.run_dir:
            outcome.resolved_sha256 = fingerprint_existing_run(Path(outcome.run_dir))
        outcomes.append(outcome)

    # Reference data (per-seed) for delta comparisons; absent when no
    # reference run exists, in which case deltas are reported as n/a.
    ref_data = find_latest_run_data(base / "reference" / "ref_full")
    ref_seeds = per_seed_map(ref_data)

    sections = list(extra_sections(outcomes, ref_seeds) if extra_sections else [])
    md = build_markdown(family, outcomes, ref_seeds, extra_sections=sections)
    csv = build_csv(family, outcomes, ref_seeds)
    js = build_json(outcomes)
    prov = provenance(
        family,
        seeds,
        str(base),
        outcomes,
        commit,
        dirty,
        extra={"reference_available": ref_data is not None},
    )

    tables_dir = base / family / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(tables_dir / "provenance.json"), prov)
    with open(tables_dir / f"{family}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(tables_dir / f"{family}.csv", "w", encoding="utf-8") as f:
        f.write(csv)
    write_json(str(tables_dir / f"{family}.json"), js)

    print(md)
    print(f"\nProvenance: {tables_dir / 'provenance.json'}")

    if gate is not None:
        if not any(o.status in ("RUN", "SKIP") for o in outcomes):
            print("[gate] skipped (no executed runs in this invocation)")
        else:
            ok, messages = gate(outcomes)
            for message in messages:
                print(f"[gate] {message}")
            if not ok:
                print("[gate] HARD FAIL -> exiting with code 1")
                return 1
    return 0