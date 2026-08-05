"""B2 — Static bank baseline.

Data signature: the most pathological run of the study. Every task is learned
well when fresh (intro accuracies ~74-83%!) but collapses to ~20% by the end,
except T9 which holds 75.2%. 'Learns everything, keeps almost nothing.'

Figures:
  1. per-task final accuracy curve (t9 spike visible)
  2. trajectory fan chart — the signature: a uniform-high introduction fan
     that collapses toward ~20% while the newest task is anchored.
  3. forgetting-accumulation trace vs the reference (near-catastrophic climb)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import plot_forgetting_accumulation, plot_trajectory_fan, styled_legend
from src.common.style import APPLE, apply_thesis_style, create_figure
from src.experiments._shared import per_task_curve_figure, save_figure_pair

logger = logging.getLogger(__name__)

KEY = "static_bank"
OUT = C.experiment_output_subdir(KEY) / "figures"


def fig_per_task_curve(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return per_task_curve_figure(
        runs[KEY], None, out_dir, OUT, "per_task_curve",
        title="B2 — Static bank: early tasks collapse, T9 dominates (75.2%)",
    )


def fig_trajectory_fan(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: the collapse fan."""
    run = runs[KEY]
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.66)
        plot_trajectory_fan(
            ax, run,
            title="B2 — Static bank: learns every task ~80%, retains ~20%",
        )
        ax.axhspan(18, 24, color=APPLE["red"], alpha=0.05)
        ax.annotate(
            "retained level ≈ 20%",
            (8.6, 21.5), fontsize=9, color=APPLE["red"],
            ha="right", fontweight="bold",
        )
        ax.annotate(
            "T9 anchored at 75.2%",
            (8.7, 78.5), fontsize=9, color=APPLE["ink"], ha="right",
        )
        ax.set_ylim(0, 100)
        return save_figure_pair(fig, out_dir / OUT / "trajectory_fan")


def fig_forgetting_accumulation(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_forgetting_accumulation(
            ax, runs, ["static_bank", C.reference_key()],
            title="B2 — near-catastrophic erosion vs the reference",
            emphasize="static_bank",
        )
        ax.set_ylim(-0.5, 60)
        styled_legend(ax, loc="upper left", fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "forgetting_accumulation")


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_trajectory_fan(runs, out_dir)
    outputs += fig_forgetting_accumulation(runs, out_dir)
    return outputs
