"""Cross-cutting figures and paper tables spanning all 11 runs.

Contains the master results table, per-seed and per-task tables, the
master accuracy comparison, the accuracy-vs-forgetting scatter, the
all-methods per-task figure, the method×task forgetting heatmap and
the ranking lollipop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common import constants as C
from src.common.data import RunResult
from src.common.latex import booktabs_table, format_mean_std, save_latex_table, save_markdown_table
from src.common.plotting import (
    add_colorbar,
    plot_method_task_forgetting_heatmap,
    plot_per_task_curve,
    plot_ranking_lollipop,
    styled_legend,
)
from src.common.style import APPLE, PALETTE, apply_thesis_style, create_figure, save_figure

logger = logging.getLogger(__name__)

FAMILY = "cross_cutting"
FIGURES_OUT = C.family_output_dir(FAMILY) / "figures"
TABLES_OUT = C.family_output_dir(FAMILY) / "tables"

REF = C.reference_key()


# ── Master accuracy comparison ───────────────────────────────────────
def fig_master_accuracy(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    keys = C.MASTER_ORDER
    means = np.asarray([runs[k].avg_acc for k in keys]) * 100.0
    stds = np.asarray([runs[k].avg_acc_std for k in keys]) * 100.0
    labels = [C.short_name(k) for k in keys]
    colors = [C.color_for(k) for k in keys]

    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.66)
        xs = np.arange(len(keys))
        bars = ax.bar(xs, means, yerr=stds, color=colors, edgecolor="white", linewidth=1.2,
                      capsize=3.5, error_kw={"elinewidth": 1.0, "ecolor": APPLE["muted"], "capthick": 1.0})
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.7, f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Average accuracy (%)"); ax.set_ylim(0, 100)
        ax.set_title("Master comparison — average accuracy over all tasks (3 seeds)", loc="left")
        bars[C.MASTER_ORDER.index(REF)].set_alpha(0.35)
        return save_figure(fig, out_dir / FIGURES_OUT / "master_accuracy_comparison")


# ── Accuracy vs forgetting scatter ───────────────────────────────────
def fig_acc_forgetting_scatter(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.66)
        for key in C.MASTER_ORDER:
            run = runs[key]; f = run.forgetting * 100.0; a = run.avg_acc * 100.0
            is_ref = key == REF
            ax.scatter(f, a, s=130 if is_ref else 85, color=C.color_for(key),
                       marker=C.marker_for(key), edgecolor="white", linewidth=1.2,
                       zorder=5 if is_ref else 3,
                       label=f"{C.short_name(key)} · {C.display_name(key)}")
        ax.set_xlabel("Forgetting (%)"); ax.set_ylabel("Average accuracy (%)")
        ax.set_title("Accuracy vs forgetting — upper-left is best", loc="left")
        ax.set_ylim(25, 55); ax.set_xlim(0, 60)
        ax.axhspan(40, 55, color=PALETTE["green"], alpha=0.06, zorder=0)
        styled_legend(ax, loc="lower left", ncol=2, fontsize=8)
        return save_figure(fig, out_dir / FIGURES_OUT / "accuracy_vs_forgetting")


# ── All methods per-task curves ──────────────────────────────────────
def fig_all_per_task(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        for key in C.MASTER_ORDER:
            plot_per_task_curve(ax, runs[key], label=f"{C.short_name(key)} · {C.display_name(key)}", band=False)
            lw = 2.6 if key == REF else 1.2
            alpha = 1.0 if key == REF else 0.85
            ax.lines[-1].set_linewidth(lw); ax.lines[-1].set_alpha(alpha)
        ax.set_xlabel("Task"); ax.set_ylabel("Final task accuracy (%)"); ax.set_ylim(0, 100)
        ax.set_title("All runs — final per-task accuracy", loc="left")
        styled_legend(ax, loc="lower center", ncol=3, fontsize=8)
        return save_figure(fig, out_dir / FIGURES_OUT / "all_methods_per_task")


# ── Method × task forgetting heatmap ─────────────────────────────────
def fig_forgetting_heatmap(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """11×10 heatmap of per-task forgetting across all methods."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.78)
        im = plot_method_task_forgetting_heatmap(
            ax, runs, C.MASTER_ORDER,
            title="Method × task forgetting heatmap (pp)",
        )
        add_colorbar(fig, im, ax, label="Forgetting (pp)")
        return save_figure(fig, out_dir / FIGURES_OUT / "forgetting_heatmap")


# ── Ranking lollipop ─────────────────────────────────────────────────
def fig_ranking_lollipop(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Sorted average-accuracy lollipop; the reference stem is emphasized."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        plot_ranking_lollipop(ax, runs, C.MASTER_ORDER, title="Ranking by average accuracy (%)")
        return save_figure(fig, out_dir / FIGURES_OUT / "ranking_lollipop")


# ── Tables ───────────────────────────────────────────────────────────
def table_master(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "avg_acc", "forgetting", "BWT"]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]
        rows.append([C.short_name(key), C.display_name(key),
                     format_mean_std(run.avg_acc, run.avg_acc_std),
                     format_mean_std(run.forgetting, run.forgetting_std),
                     format_mean_std(run.bwt, None)])
    tex = booktabs_table(headers, rows,
                         caption="Master results on CIFAR-100 class-incremental learning (10 tasks, mean $\\pm$ std over 3 seeds).",
                         label="tab:master_results")
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / "master_results.tex")
    md_path = save_markdown_table(headers, rows, out_dir / TABLES_OUT / "master_results.md",
                                  title="Master results — all runs")
    return [tex_path, md_path]


def table_per_seed(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "seed 1993", "seed 2023", "seed 42", "mean"]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]; accs = run.per_seed_avg_accs()
        rows.append([C.short_name(key), C.display_name(key),
                     f"{accs[1993]*100:.2f}", f"{accs[2023]*100:.2f}",
                     f"{accs[42]*100:.2f}", f"{run.avg_acc*100:.2f}"])
    tex = booktabs_table(headers, rows, caption="Per-seed average accuracies (%) for all runs.", label="tab:per_seed")
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / "per_seed_results.tex")
    md_path = save_markdown_table(headers, rows, out_dir / TABLES_OUT / "per_seed_results.md",
                                  title="Per-seed average accuracies (%)")
    return [tex_path, md_path]


def table_per_task(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment"] + [f"t{t}" for t in range(C.NUM_TASKS)]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]; accs = run.final_task_accs()
        rows.append([C.short_name(key), C.display_name(key)] + [f"{v:.1f}" for v in accs])
    tex = booktabs_table(headers, rows, caption="Final-state per-task accuracies (%, mean over 3 seeds).", label="tab:per_task")
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / "per_task_accuracies.tex")
    md_path = save_markdown_table(headers, rows, out_dir / TABLES_OUT / "per_task_accuracies.md",
                                  title="Final-state per-task accuracies (mean over seeds, %)")
    return [tex_path, md_path]


# ── Orchestrators ────────────────────────────────────────────────────
def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_master_accuracy(runs, out_dir)
    outputs += fig_acc_forgetting_scatter(runs, out_dir)
    outputs += fig_all_per_task(runs, out_dir)
    outputs += fig_forgetting_heatmap(runs, out_dir)
    outputs += fig_ranking_lollipop(runs, out_dir)
    return outputs


def generate_tables(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += table_master(runs, out_dir)
    outputs += table_per_seed(runs, out_dir)
    outputs += table_per_task(runs, out_dir)
    return outputs
