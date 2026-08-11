"""Paper pipeline orchestrator: the approved manuscript figure/table set.

Generates exactly the approved artifacts:

    outputs/paper/main/figures/      Fig 1-5 (PDF + PNG)
    outputs/paper/appendix/figures/  Fig A1-A4 (PDF + PNG)
    outputs/paper/tables/            T1-T3, A1-A6 (.tex/.md = 18 files)

The former 38-figure library pipeline is removed; this script is the
single generation entrypoint (run ``verify_paper.py`` afterwards).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

# Bootstrap: ensure studies/analysis (the package parent) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import get_config, get_output_root
from src.common import constants as C
from src.common.data import load_all_runs
from src.paper import appendix_figures, main_figures, tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the approved paper figure/table set into outputs/paper/.",
    )
    parser.add_argument("--figures-only", action="store_true", help="Generate only figures.")
    parser.add_argument("--tables-only", action="store_true", help="Generate only tables.")
    args = parser.parse_args()

    run_figures = not args.tables_only
    run_tables = not args.figures_only

    cfg = get_config()
    out_dir = get_output_root(cfg)

    elapsed_start = time.perf_counter()
    logger.info("=" * 72)
    logger.info("GHOST BANK CIL PAPER - APPROVED ARTIFACT PIPELINE")
    logger.info("Output root : %s", out_dir)
    logger.info("=" * 72)

    runs = load_all_runs()
    missing = [k for k in C.MASTER_ORDER if k not in runs]
    if missing:
        logger.error("Missing runs: %s", missing)
        sys.exit(1)
    logger.info("Runs loaded: %d/12", len(runs))

    outputs: List[Path] = []

    if run_figures:
        t0 = time.perf_counter()
        logger.info("-" * 72)
        logger.info("MAIN FIGURES (Fig 1-5)")
        main_out = main_figures.generate_figures(runs, out_dir)
        outputs += main_out
        logger.info("Main figures: %d files in %.1fs", len(main_out), time.perf_counter() - t0)

        t0 = time.perf_counter()
        logger.info("-" * 72)
        logger.info("APPENDIX FIGURES (Fig A1-A4)")
        app_out = appendix_figures.generate_figures(runs, out_dir)
        outputs += app_out
        logger.info("Appendix figures: %d files in %.1fs", len(app_out), time.perf_counter() - t0)

    if run_tables:
        t0 = time.perf_counter()
        logger.info("-" * 72)
        logger.info("TABLES (T1-T3, A1-A6)")
        tbl_out = tables.generate_tables(runs, out_dir)
        outputs += tbl_out
        logger.info("Tables: %d files in %.1fs", len(tbl_out), time.perf_counter() - t0)

    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE - %d files", len(outputs))
    for path in sorted(outputs):
        logger.info("    -> %s", path)
    logger.info("Elapsed: %.2f seconds", time.perf_counter() - elapsed_start)
    logger.info("Next: run `python -m src.scripts.verify_paper`.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
