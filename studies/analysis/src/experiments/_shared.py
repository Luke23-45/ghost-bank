"""Shared composition helpers for per-experiment figure modules.

These are thin, purpose-built pieces (not wholesale templates): every
experiment module still decides which charts to draw, what to annotate
and how to lay them out. This file only removes the pure boilerplate
(figure creation, saving, standard axes finishing).

``out_dir`` is the analysis output root (``outputs/``); each module is
responsible for its own sub-path (family/type/experiment + figures/).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from src.common.data import RunResult
from src.common.plotting import add_colorbar, plot_evolution_heatmap, plot_per_task_curve, styled_legend
from src.common.style import apply_thesis_style, create_figure, save_figure


def save_figure_pair(fig: plt.Figure, path: Path) -> List[Path]:
    """Save pdf+png under the given base path and return both files."""
    return save_figure(fig, path, formats=("pdf", "png"))


def per_task_curve_figure(
    run: RunResult,
    ref: Optional[RunResult],
    out_dir: Path,
    subdir: Path,
    name: str,
    *,
    title: str,
    legend_ncol: int = 2,
    annotate: bool = False,
    ylim: Tuple[float, float] = (0, 100),
) -> List[Path]:
    """Per-task final-accuracy curve (CIL-standard spine), with or without
    a reference anchor and +-1 std bands."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        if ref is not None:
            plot_per_task_curve(ax, ref, band=True)
            ax.lines[-1].set_linewidth(1.4)
            ax.lines[-1].set_alpha(0.75)
        plot_per_task_curve(ax, run, band=True, annotate=annotate)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title(title, loc="left")
        ax.set_ylim(*ylim)
        if ref is not None:
            styled_legend(ax, loc="lower center", ncol=legend_ncol, fontsize=9)
        return save_figure_pair(fig, out_dir / subdir / name)


def evolution_heatmap_figure(
    run: RunResult,
    out_dir: Path,
    subdir: Path,
    name: str,
    *,
    title: str,
) -> List[Path]:
    """Task-evolution heatmap (row = evaluation time, column = task)."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.78)
        im = plot_evolution_heatmap(ax, run, title=title)
        add_colorbar(fig, im, ax)
        return save_figure_pair(fig, out_dir / subdir / name)
