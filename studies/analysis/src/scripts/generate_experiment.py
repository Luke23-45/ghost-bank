"""Targeted regeneration for a single experiment.

Usage::

    python -m src.scripts.generate_experiment --experiment B3
    python -m src.scripts.generate_experiment --experiment a2 --include-family
    python -m src.scripts.generate_experiment --experiment s1 --tables

This gives per-experiment control: a modified figure for one run can be
regenerated without touching the other 10.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Bootstrap: ensure studies/analysis (the package parent) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import constants as C
from src.common.config import get_config, get_output_root
from src.common.data import RunResult, load_all_runs
from src.experiments import experiment_module, family_module, family_keys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_key(token: str) -> str:
    """Accept B3, b3, uniform_herding, or s1_budget500 style tokens."""
    token = token.strip().lower()
    if token in C.MASTER_ORDER:
        return token
    for key in C.MASTER_ORDER:
        if C.short_name(key).lower() == token:
            return key
    raise ValueError(
        f"unknown experiment '{token}'. Use a short name (B1, a2, s1) or a key "
        f"({', '.join(C.MASTER_ORDER)})"
    )


def _key_to_family(key: str) -> str:
    """Return the family name that owns this experiment key."""
    for fam in ["baselines", "component", "sensitivity"]:
        if key in family_keys(fam):
            return fam
    return "cross_cutting"


def regenerate_experiment(
    token: str,
    *,
    include_family: bool = False,
    run_tables: bool = False,
) -> List[Path]:
    key = resolve_key(token)
    family = _key_to_family(key)

    cfg = get_config()
    out_dir = get_output_root(cfg)

    logger.info("=" * 72)
    logger.info("TARGETED REGENERATION - %s (%s) via family '%s'", C.short_name(key), key, family)
    logger.info("=" * 72)

    runs = load_all_runs()
    outputs: List[Path] = []

    # Per-experiment bespoke figures
    mod = experiment_module(key)
    outputs += mod.generate_figures(runs, out_dir)
    logger.info("Regenerated per-experiment figures for '%s'.", key)

    # Optional: family-level comparison figures
    if include_family:
        fam_mod = family_module(family)
        outputs += fam_mod.generate_figures(runs, out_dir)
        logger.info("Regenerated ALL figures for family '%s'.", family)

    # Optional: family-level tables
    if run_tables:
        fam_mod = family_module(family)
        outputs += fam_mod.generate_tables(runs, out_dir)
        logger.info("Regenerated tables for family '%s'.", family)

    logger.info("=" * 72)
    for path in sorted(outputs):
        logger.info("    -> %s", path)
    logger.info("=" * 72)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures for one experiment.")
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment token, e.g. B3, a2, s1, uniform_herding.",
    )
    parser.add_argument(
        "--include-family", action="store_true",
        help="Also regenerate the family-level comparison figures.",
    )
    parser.add_argument("--tables", action="store_true", help="Also regenerate family tables.")
    args = parser.parse_args()

    try:
        t0 = time.perf_counter()
        regenerate_experiment(
            args.experiment,
            include_family=args.include_family,
            run_tables=args.tables,
        )
        logger.info("Elapsed: %.2f seconds", time.perf_counter() - t0)
    except Exception as exc:
        logger.error("Targeted regeneration failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
