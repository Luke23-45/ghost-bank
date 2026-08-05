"""a4 — Random bank ablation (reference with random exemplar selection).

Data signature: the "herding doesn't matter" finding. Introduction
accuracies are identical to the reference, but final accuracy drops
uniformly −4.8 pp on every task. The per-seed consistency strip shows
the tightest cluster of the study (std 0.0017) — the uniformity IS
the finding.

Figures:
  1. per-task final accuracy curve vs reference
  2. per-task delta bars (uniform −4.8 pp)
  3. per-seed consistency strip (extremely low variance — the headline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.plotting import (
    plot_per_task_curve,
    plot_per_task_delta_bars,
    plot_seed_consistency_strip,
    styled_legend,
)
from src.common.style import apply_thesis_style, create_figure
from src.experiments._shared import save_figure_pair

logger = logging.getLogger(__name__)

KEY = "a4_random_bank"
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
        ax.set_title("a4 — Random selection: identical intro, uniformly −4.8 pp", loc="left")
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="lower center", ncol=2, fontsize=9)
        return save_figure_pair(fig, out_dir / OUT / "per_task_curve")


def fig_delta_bars(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.52)
        plot_per_task_delta_bars(
            ax, runs[KEY], runs[C.reference_key()],
            title="a4 — Per-task delta vs reference: uniform −4.8 pp across all tasks",
            annotate_mean=True,
        )
        return save_figure_pair(fig, out_dir / OUT / "delta_bars")


def fig_seed_consistency(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Signature figure: the extremely tight per-seed cluster."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.35)
        plot_seed_consistency_strip(
            ax, runs[KEY], runs[C.reference_key()],
            title="a4 — Per-seed consistency: std 0.0017 (extremely tight cluster)",
        )
        return save_figure_pair(fig, out_dir / OUT / "seed_consistency")


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_per_task_curve(runs, out_dir)
    outputs += fig_delta_bars(runs, out_dir)
    outputs += fig_seed_consistency(runs, out_dir)
    return outputs
