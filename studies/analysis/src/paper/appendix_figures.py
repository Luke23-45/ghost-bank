"""Appendix manuscript figures (Fig A1-A4) for the Ghost Bank CIL paper.

Reuses the verified plotting archetypes in ``src/common/plotting.py``;
data comes from the persisted artifacts via ``src/common/data.py``.
Composition follows ``docs/paper/analysis/analysis_plan.md`` Section 5.

Output: ``outputs/paper/appendix/figures/`` (PDF + PNG per figure).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import (
    add_colorbar,
    finish_axes,
    plot_evolution_heatmap,
    plot_method_task_forgetting_heatmap,
    plot_stability_slopes,
    styled_legend,
)
from src.common.style import apply_thesis_style, create_figure, save_figure

logger = logging.getLogger(__name__)

REF = C.reference_key()
APPENDIX_OUT = C.PAPER_APPENDIX_FIGURES_DIR


# ── Fig A1: method x task forgetting heatmap ─────────────────────────
def figA1_forgetting_heatmap(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """11x10 per-task forgetting matrix across all methods."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.78)
        im = plot_method_task_forgetting_heatmap(
            ax, runs, C.MASTER_ORDER,
            title="Per-task forgetting across all methods (pp)",
        )
        add_colorbar(fig, im, ax, label="Forgetting (pp)")
        finish_axes(ax)
        return save_figure(fig, out_dir / APPENDIX_OUT / "figA1_forgetting_heatmap")


# ── Fig A2: evolution heatmap, reference ─────────────────────────────
def figA2_evolution_reference(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """10x10 task-evolution matrix of the flagship (reference) method."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.78)
        im = plot_evolution_heatmap(
            ax, runs[REF],
            title="Uniform herding (reference) — task evolution",
        )
        add_colorbar(fig, im, ax)
        finish_axes(ax)
        return save_figure(fig, out_dir / APPENDIX_OUT / "figA2_evolution_reference")


# ── Fig A3: evolution heatmap, iCaRL ─────────────────────────────────
def figA3_evolution_icarl(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """10x10 task-evolution matrix of iCaRL (steady-erosion signature)."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.78)
        im = plot_evolution_heatmap(
            ax, runs["icarl"],
            title="iCaRL — task evolution (steady diagonal erosion)",
        )
        add_colorbar(fig, im, ax)
        finish_axes(ax)
        return save_figure(fig, out_dir / APPENDIX_OUT / "figA3_evolution_icarl")


# ── Fig A4: stability slopes, a1 ─────────────────────────────────────
def figA4_stability_slopes(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Intro-to-final paired drops: no-KD (a1) vs reference."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.58)
        plot_stability_slopes(
            ax, runs["a1_no_kd"], runs[REF],
            title="No-KD ablation (a1) — stability slopes",
            run_label=C.display_name("a1_no_kd"),
        )
        ax.set_ylim(0, 80)
        finish_axes(ax)
        styled_legend(ax, loc="upper left", fontsize=8.5, ncol=2)
        return save_figure(fig, out_dir / APPENDIX_OUT / "figA4_stability_slopes")


# ── Registry (keys match C.PAPER_APPENDIX_FIGURES) ───────────────────
BUILDERS: Dict[str, object] = {
    "figA1_forgetting_heatmap": figA1_forgetting_heatmap,
    "figA2_evolution_reference": figA2_evolution_reference,
    "figA3_evolution_icarl": figA3_evolution_icarl,
    "figA4_stability_slopes": figA4_stability_slopes,
}


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    for name in C.PAPER_APPENDIX_FIGURES:
        outputs += BUILDERS[name](runs, out_dir)
    return outputs
