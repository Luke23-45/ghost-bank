"""Baseline family-level comparison (B1/B2/B3 on one axis) + table.

Per-experiment bespoke figures live in b1_icarl.py, b2_static_bank.py,
b3_uniform_herding.py.  This module only handles the *overlay* comparison
and the family-level table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.common import constants as C
from src.common.data import RunResult
from src.common.latex import booktabs_table, format_mean_std, save_latex_table, save_markdown_table
from src.common.plotting import plot_per_task_curve, styled_legend
from src.common.style import apply_thesis_style, create_figure, save_figure

logger = logging.getLogger(__name__)

FAMILY = "baselines"
FIGURES_OUT = C.family_output_dir(FAMILY) / "figures"
TABLES_OUT = C.family_output_dir(FAMILY) / "tables"


def fig_baselines_comparison(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """B1/B2/B3 overlaid per-task curves (the headline comparison)."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        for key in C.BASELINE_KEYS:
            plot_per_task_curve(
                ax, runs[key],
                label=f"{C.short_name(key)} · {C.display_name(key)}",
            )
            lw = 2.4 if key == C.reference_key() else 1.6
            alpha = 1.0 if key == C.reference_key() else 0.85
            ax.lines[-1].set_linewidth(lw)
            ax.lines[-1].set_alpha(alpha)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title("Baselines — per-task accuracy at the end of training", loc="left")
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="lower center", ncol=3, fontsize=9)
        return save_figure(fig, out_dir / FIGURES_OUT / "baselines_comparison")


def table_baselines(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "avg_acc", "forgetting", "BWT"]
    rows = []
    for key in C.BASELINE_KEYS:
        run = runs[key]
        rows.append([
            C.short_name(key), C.display_name(key),
            format_mean_std(run.avg_acc, run.avg_acc_std),
            format_mean_std(run.forgetting, run.forgetting_std),
            format_mean_std(run.bwt, None),
        ])
    tex = booktabs_table(
        headers, rows,
        caption="Baseline results on CIFAR-100 class-incremental learning (mean $\\pm$ std over 3 seeds).",
        label="tab:baselines",
    )
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / "baselines.tex")
    md_path = save_markdown_table(headers, rows, out_dir / TABLES_OUT / "baselines.md", title="Baselines")
    return [tex_path, md_path]


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return fig_baselines_comparison(runs, out_dir)


def generate_tables(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return table_baselines(runs, out_dir)
