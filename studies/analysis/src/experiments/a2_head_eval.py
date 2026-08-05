"""a2 — Head-logit evaluation ablation (reference without NME readout).

Data signature: the most dramatic ablation. Forgetting is a monotone
recency gradient: T0 loses 49.9 pp, T1 42.1 pp, … down to T9 at 0 pp.
The forgetting-by-age curve is the textbook signature of classifier
recency bias — the headline chart for this module.

Figures:
  1. per-task final accuracy curve vs reference (annotated t1/t9)
  2. forgetting by task age — the signature: a near-perfect monotone
     decline from T0 to T9 (49.9 → 0 pp)
  3. trajectory fan — visible collapse with new-task anchoring
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import (
    plot_forgetting_by_age,
    plot_per_task_curve,
    plot_trajectory_fan,
    styled_legend,
)
from src.common.style import APPLE, PALETTE, apply_thesis_style, create_figure
from src.experiments._shared import save_figure_pair

logger = logging.getLogger(__name__)

KEY = "a2_head_eval"
OUT = C.experiment_output_subdir(KEY) / "figures"


def fig_per_task_curve(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Per-task curve: the recency bias is visible as t1/t9 divergence."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_per_task_curve(ax, runs[C.reference_key()], band=True)
        ax.lines[-1].set_linewidth(1.4)
        ax.lines[-1].set_alpha(0.75)
        plot_per_task_curve(ax, runs[KEY], band=True, annotate=True)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title("a2 — Without NME readout: catastrophic recency bias", loc="left")
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="lower center", ncol=2, fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "per_task_curve")


def fig_forgetting_by_age(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: the monotone forgetting gradient."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.58)
        plot_forgetting_by_age(
            ax, runs[KEY], ref=runs[C.reference_key()],
            title="a2 — Forgetting by task age: 49.9 → 0 pp gradient (recency bias)",
            fill_gradient=True,
        )
        ax.set_ylim(-3, 58)
        styled_legend(ax, loc="upper right", fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "forgetting_by_age")


def fig_trajectory_fan(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Trajectory fan: the collapse is anchored to the newest task."""
    run = runs[KEY]
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.66)
        plot_trajectory_fan(
            ax, run,
            title="a2 — Without NME: trajectory collapse (old tasks anchor to zero)",
            label_every=2,
        )
        ax.axhspan(-1, 2, color=PALETTE["wine"], alpha=0.05)
        ax.annotate(
            "old tasks collapse toward 0%",
            (8.6, 3.0), fontsize=9, color=PALETTE["wine"],
            ha="right", fontweight="bold",
        )
        ax.set_ylim(-2, 100)
        return save_figure_pair(fig, out_dir / OUT / "trajectory_fan")


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_forgetting_by_age(runs, out_dir)
    outputs += fig_trajectory_fan(runs, out_dir)
    return outputs
