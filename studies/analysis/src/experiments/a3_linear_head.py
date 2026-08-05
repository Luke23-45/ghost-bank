"""a3 — Linear head ablation (reference with linear classification head).

Data signature: the "no honest effect" experiment. Swapping the
cosine-margin NME head for a plain linear head gives only +1.4 pp
mean delta, and even −1.2 pp on T8 (negative forgetting). The
delta bars are the honest, small-effect chart.

Figures:
  1. per-task final accuracy curve vs reference
  2. per-task delta bars (small +1.4 pp mean; T8 negative)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import plot_per_task_curve, plot_per_task_delta_bars, styled_legend
from src.common.style import apply_thesis_style, create_figure
from src.experiments._shared import save_figure_pair

logger = logging.getLogger(__name__)

KEY = "a3_linear_head"
OUT = C.experiment_output_subdir(KEY) / "figures"


def fig_per_task_curve(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_per_task_curve(ax, runs[C.reference_key()], band=True)
        ax.lines[-1].set_linewidth(1.4)
        ax.lines[-1].set_alpha(0.75)
        plot_per_task_curve(ax, runs[KEY], band=True, annotate=True)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title("a3 — Linear head: marginal +1.4 pp effect (T8 negative)", loc="left")
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="lower center", ncol=2, fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "per_task_curve")


def fig_delta_bars(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: the honest small-effect delta chart."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.52)
        plot_per_task_delta_bars(
            ax, runs[KEY], runs[C.reference_key()],
            title="a3 — Per-task delta vs reference: mean +1.4 pp (T8: −1.2 pp)",
            annotate_mean=True,
        )
        return save_figure_pair(fig, out_dir / OUT / "delta_bars")


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_delta_bars(runs, out_dir)
    return outputs
