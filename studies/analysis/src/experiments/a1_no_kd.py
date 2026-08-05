"""a1 — No-KD ablation (reference without knowledge distillation).

Data signature: the KD-stability story in one chart. Removing KD lets
introduction accuracies HIGHER than the reference (74.3 pp vs 56.5 pp
at t1) but per-task forgetting doubles (32 pp vs 12 pp). The stability
slopes chart shows the "higher peaks, deeper troughs" trade-off.

Figures:
  1. per-task final accuracy curve vs the reference
  2. stability slope chart — intro-to-final lines for both runs;
     the steeper slopes of the no-KD run are the signature
  3. forgetting accumulation vs the reference — no-KD erodes faster
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import (
    plot_forgetting_accumulation,
    plot_per_task_curve,
    plot_stability_slopes,
    styled_legend,
)
from src.common.style import apply_thesis_style, create_figure
from src.experiments._shared import save_figure_pair

logger = logging.getLogger(__name__)

KEY = "a1_no_kd"
OUT = C.experiment_output_subdir(KEY) / "figures"


def fig_per_task_curve(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Per-task curve: no-KD vs reference; higher intro but deeper forgetting."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_per_task_curve(ax, runs[C.reference_key()], band=True)
        ax.lines[-1].set_linewidth(1.4)
        ax.lines[-1].set_alpha(0.75)
        plot_per_task_curve(ax, runs[KEY], band=True, annotate=True)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title("a1 — Without KD: higher introduction peaks, deeper forgetting", loc="left")
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="lower center", ncol=2, fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "per_task_curve")


def fig_stability_slopes(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: slope chart of intro-to-final drops."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.58)
        plot_stability_slopes(
            ax, runs[KEY], runs[C.reference_key()],
            title="a1 — Stability slopes: no-KD steeper on every task",
            run_label=C.display_name(KEY),
        )
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="upper left", fontsize=8.5, ncol=2)
        return save_figure_pair(fig, out_dir / OUT / "stability_slopes")


def fig_forgetting_accumulation(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """No-KD erodes faster: steeper accumulation trace."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_forgetting_accumulation(
            ax, runs, [KEY, C.reference_key()],
            title="a1 — Without KD: cumulative forgetting doubles vs reference",
            emphasize=KEY,
        )
        ax.set_ylim(-0.5, 45)
        styled_legend(ax, loc="upper left", fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "forgetting_accumulation")


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_stability_slopes(runs, out_dir)
    outputs += fig_forgetting_accumulation(runs, out_dir)
    return outputs
