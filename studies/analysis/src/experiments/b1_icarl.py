"""B1 — iCaRL baseline.

Data signature: a flat, well-behaved final curve (37-48%) whose per-task
forgetting decays monotonically with task age (36.0 pp on T0 down to 5.3 pp
on T8). iCaRL does not collapse; it erodes steadily, task by task.

Figures:
  1. per-task final accuracy curve (with +-1 std band)
  2. task-evolution heatmap (gradual diagonal erosion)
  3. forgetting-accumulation trace vs the reference — 'steady erosion' is
     the signature; the reference's flatter trace shows why it wins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import plot_forgetting_accumulation, styled_legend
from src.common.style import apply_thesis_style, create_figure
from src.experiments._shared import evolution_heatmap_figure, per_task_curve_figure, save_figure_pair

logger = logging.getLogger(__name__)

KEY = "icarl"
OUT = C.experiment_output_subdir(KEY) / "figures"


def fig_per_task_curve(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return per_task_curve_figure(
        runs[KEY], None, out_dir, OUT, "per_task_curve",
        title="B1 — iCaRL: flat, well-behaved per-task accuracy",
    )


def fig_evolution_heatmap(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return evolution_heatmap_figure(
        runs[KEY], out_dir, OUT, "evolution",
        title="B1 — iCaRL: task evolution (gradual diagonal erosion)",
    )


def fig_forgetting_accumulation(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: iCaRL erodes steadily; the reference flattens it."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_forgetting_accumulation(
            ax, runs, ["icarl", C.reference_key()],
            title="B1 — iCaRL erodes steadily; the reference keeps forgetting flat",
            emphasize=C.reference_key(),
        )
        ax.set_ylim(-0.5, 45)
        styled_legend(ax, loc="upper left", fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "forgetting_accumulation")


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_evolution_heatmap(runs, out_dir)
    outputs += fig_forgetting_accumulation(runs, out_dir)
    return outputs
