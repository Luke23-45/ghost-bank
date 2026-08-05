"""B3 — Uniform herding reference (the locked anchor).

Data signature: the most uniform per-task curve of the study (34.7-54.5%),
a tight parallel trajectory fan, and low flat per-task forgetting
(~10-16 pp for all but T0). 'Uniform' is the headline; the figures must
make the uniformity self-evident.

Figures:
  1. per-task final accuracy curve with uniformity range band annotation
  2. trajectory fan — the signature: a stable, near-parallel bundle
  3. task-evolution heatmap
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import plot_per_task_curve, plot_trajectory_fan
from src.common.style import APPLE, apply_thesis_style, create_figure
from src.experiments._shared import evolution_heatmap_figure, save_figure_pair

logger = logging.getLogger(__name__)

KEY = "uniform_herding"
OUT = C.experiment_output_subdir(KEY) / "figures"


def fig_per_task_curve(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(6.5, 0.62 * 6.5), constrained_layout=True)
        run = runs[KEY]
        plot_per_task_curve(ax, run, band=True, annotate=True)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title("B3 — Uniform herding: the most uniform per-task curve (34.7-54.5%)", loc="left")
        ax.set_ylim(0, 100)
        ax.fill_between([-0.4, 9.4], 34.7, 54.5, color=APPLE["green"], alpha=0.05, linewidth=0)
        ax.annotate("uniformity band: 34.7–54.5 pp", (9.35, 55.0),
                    fontsize=8.5, color=APPLE["green"], ha="right", fontweight="bold")
        return save_figure_pair(fig, out_dir / OUT / "per_task_curve")


def fig_trajectory_fan(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: the stable parallel fan."""
    run = runs[KEY]
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.66)
        plot_trajectory_fan(
            ax, run,
            title="B3 — Uniform herding: tight, parallel task trajectories",
            label_every=3,
        )
        ax.set_ylim(0, 100)
        return save_figure_pair(fig, out_dir / OUT / "trajectory_fan")


def fig_evolution_heatmap(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return evolution_heatmap_figure(
        runs[KEY], out_dir, OUT, "evolution",
        title="B3 — Uniform herding: task evolution (uniform retention)",
    )


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_trajectory_fan(runs, out_dir)
    outputs += fig_evolution_heatmap(runs, out_dir)
    return outputs
