"""Figure generation orchestrator for the Ghost Bank CIL paper."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Bootstrap: ensure studies/analysis (the package parent) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import get_config, get_output_root
from src.common.data import RunResult, load_all_runs
from src.experiments import FAMILIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def generate_all_figures(
    families: Optional[List[str]] = None,
    runs: Optional[Dict[str, RunResult]] = None,
) -> List[Path]:
    """Run all family figure generators and report outputs.

    Modules receive the output root (``out_dir``) and place files under
    their own family/type/experiment + figures/ sub-path.
    """
    cfg = get_config()
    out_dir = get_output_root(cfg)

    if runs is None:
        runs = load_all_runs()

    targets = families or list(FAMILIES.keys())
    logger.info("=" * 72)
    logger.info("GHOST BANK CIL - FIGURE GENERATION")
    logger.info("=" * 72)
    logger.info("Output root : %s", out_dir)
    logger.info("Families    : %s", ", ".join(targets))
    logger.info("-" * 72)

    outputs: List[Path] = []
    t0 = time.perf_counter()
    for family in targets:
        module = FAMILIES[family]
        f_t0 = time.perf_counter()
        family_out = module.generate_figures(runs, out_dir)
        outputs += family_out
        logger.info("[%s] %d files in %.1fs", family, len(family_out), time.perf_counter() - f_t0)

    logger.info("=" * 72)
    logger.info("FIGURE GENERATION COMPLETE")
    logger.info("  Total files written : %d", len(outputs))
    for path in sorted(outputs):
        logger.info("    -> %s", path)
    logger.info("  Elapsed             : %.2f seconds", time.perf_counter() - t0)
    logger.info("=" * 72)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument(
        "--families", nargs="*", default=None,
        choices=["baselines", "component", "sensitivity", "cross_cutting"],
        help="Restrict to specific experiment families (default: all).",
    )
    args = parser.parse_args()
    try:
        generate_all_figures(families=args.families)
    except Exception as exc:
        logger.error("Figure generation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
