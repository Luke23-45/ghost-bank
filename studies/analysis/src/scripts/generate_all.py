"""Master CLI orchestrator for the Ghost Bank CIL paper analysis pipeline.

Generates BOTH per-experiment bespoke figures AND family-level comparison figures.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Bootstrap: ensure studies/analysis (the package parent) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import get_config, get_output_root
from src.common.data import load_all_runs
from src.experiments import all_experiment_keys, experiment_module, family_module, family_keys

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
        description="Ghost Bank CIL paper: generate all figures and tables from run artifacts.",
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
    logger.info("GHOST BANK CIL PAPER - COMPLETE ANALYSIS PIPELINE")
    logger.info("=" * 72)

    runs = load_all_runs()

    if run_figures:
        # ── Per-experiment bespoke figures ────────────────────────────
        t0 = time.perf_counter()
        fig_outputs = []
        logger.info("-" * 72)
        logger.info("PER-EXPERIMENT BESPOKE FIGURES")
        logger.info("-" * 72)
        for key in all_experiment_keys():
            mod = experiment_module(key)
            key_t0 = time.perf_counter()
            files = mod.generate_figures(runs, out_dir)
            fig_outputs += files
            logger.info("[%s] %d files in %.1fs", key, len(files), time.perf_counter() - key_t0)
        logger.info("Per-experiment figures: %d files in %.1fs", len(fig_outputs), time.perf_counter() - t0)

        # ── Family-level comparison figures ───────────────────────────
        t0 = time.perf_counter()
        fam_outputs = []
        logger.info("-" * 72)
        logger.info("FAMILY-LEVEL COMPARISON FIGURES")
        logger.info("-" * 72)
        for fam in ["baselines", "component", "sensitivity", "cross_cutting"]:
            mod = family_module(fam)
            fam_t0 = time.perf_counter()
            files = mod.generate_figures(runs, out_dir)
            fam_outputs += files
            logger.info("[%s] %d files in %.1fs", fam, len(files), time.perf_counter() - fam_t0)
        logger.info("Family-level figures: %d files in %.1fs", len(fam_outputs), time.perf_counter() - t0)
        logger.info("TOTAL FIGURES: %d files", len(fig_outputs) + len(fam_outputs))

    if run_tables:
        t0 = time.perf_counter()
        tbl_outputs = []
        logger.info("-" * 72)
        logger.info("TABLE GENERATION")
        logger.info("-" * 72)
        for fam in ["baselines", "component", "sensitivity", "cross_cutting"]:
            mod = family_module(fam)
            files = mod.generate_tables(runs, out_dir)
            tbl_outputs += files
            logger.info("[%s] %d files", fam, len(files))
        logger.info("TOTAL TABLES: %d files in %.1fs", len(tbl_outputs), time.perf_counter() - t0)

    logger.info("=" * 72)
    logger.info("ALL DONE - Total elapsed: %.2f seconds", time.perf_counter() - elapsed_start)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
