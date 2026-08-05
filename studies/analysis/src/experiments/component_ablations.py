"""Component-ablation family-level comparison + attribution chart + table.

Per-experiment bespoke figures live in a1_no_kd.py, a2_head_eval.py,
a3_linear_head.py, a4_random_bank.py.  This module only handles the
cross-ablation overlay, the attribution delta chart and the family table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common import constants as C
from src.common.data import RunResult, matched_deltas
from src.common.latex import booktabs_table, format_mean_std, save_latex_table, save_markdown_table
from src.common.plotting import plot_per_task_curve, styled_legend
from src.common.style import APPLE, apply_thesis_style, create_figure, save_figure

logger = logging.getLogger(__name__)

FAMILY = "component"
FIGURES_OUT = C.family_output_dir(FAMILY) / "figures"
TABLES_OUT = C.family_output_dir(FAMILY) / "tables"

REF = C.reference_key()
ABLATION_KEYS = [k for k in C.COMPONENT_KEYS if k != REF]


def fig_component_comparison(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """All component ablations + reference on one axis."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.62)
        for key in C.COMPONENT_KEYS:
            lw = 2.4 if key == REF else 1.5
            plot_per_task_curve(ax, runs[key], label=f"{C.short_name(key)} · {C.display_name(key)}")
            ax.lines[-1].set_linewidth(lw)
        ax.set_xlabel("Task")
        ax.set_ylabel("Final task accuracy (%)")
        ax.set_title("Component ablations — per-task accuracy vs reference", loc="left")
        ax.set_ylim(0, 100)
        styled_legend(ax, loc="lower center", ncol=3, fontsize=8)
        return save_figure(fig, out_dir / FIGURES_OUT / "component_comparison")


def fig_attribution(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """Component-attribution chart: per-seed matched delta accuracy vs reference."""
    deltas = matched_deltas(runs, REF)
    labels = [f"{C.display_name(k)}" for k in ABLATION_KEYS]
    acc_effects = [float(np.mean(list(deltas[k]["avg_acc"].values()))) * 100.0 for k in ABLATION_KEYS]
    for_effects = [float(np.mean(list(deltas[k]["forgetting"].values()))) * 100.0 for k in ABLATION_KEYS]

    with apply_thesis_style():
        fig, axes = create_figure(width="double", nrows=1, ncols=2, aspect=0.5)
        ax1, ax2 = axes

        order = np.argsort(acc_effects)
        colors = [C.color_for(ABLATION_KEYS[i]) for i in order]
        ys = np.arange(len(labels))
        ax1.barh(ys, np.asarray(acc_effects)[order], color=colors, edgecolor="white", linewidth=1.2)
        ax1.set_yticks(ys)
        ax1.set_yticklabels([f"{C.short_name(ABLATION_KEYS[i])} · {labels[i]}" for i in order], fontsize=9)
        ax1.axvline(0, color=APPLE["ink"], linewidth=1.0)
        ax1.set_title("(a) Delta average accuracy (pp)", loc="left")
        for y, e in zip(ys, np.asarray(acc_effects)[order]):
            ax1.text(e + (0.3 if e >= 0 else -0.3), y, f"{e:+.1f}",
                     va="center", ha="left" if e >= 0 else "right", fontsize=9)

        order2 = np.argsort(for_effects)
        colors2 = [C.color_for(ABLATION_KEYS[i]) for i in order2]
        ax2.barh(ys, np.asarray(for_effects)[order2], color=colors2, edgecolor="white", linewidth=1.2)
        ax2.set_yticks(ys)
        ax2.set_yticklabels([f"{C.short_name(ABLATION_KEYS[i])} · {labels[i]}" for i in order2], fontsize=9)
        ax2.axvline(0, color=APPLE["ink"], linewidth=1.0)
        ax2.set_title("(b) Delta forgetting (pp)", loc="left")
        for y, e in zip(ys, np.asarray(for_effects)[order2]):
            ax2.text(e + (0.3 if e >= 0 else -0.3), y, f"{e:+.1f}",
                     va="center", ha="left" if e >= 0 else "right", fontsize=9)

        for ax in (ax1, ax2):
            ax.grid(True, axis="x", color=APPLE["grid"], linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(APPLE["grid"])
            ax.spines["bottom"].set_color(APPLE["grid"])
            ax.tick_params(colors=APPLE["muted"])

        return save_figure(fig, out_dir / FIGURES_OUT / "attribution_deltas")


def table_component(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    deltas = matched_deltas(runs, REF)
    headers = ["#", "Experiment", "avg_acc", "forgetting", "Δ acc (pp)", "Δ forgetting (pp)"]
    rows = []
    for key in ABLATION_KEYS:
        run = runs[key]
        d = deltas[key]
        acc_eff = float(np.mean(list(d["avg_acc"].values()))) * 100.0
        for_eff = float(np.mean(list(d["forgetting"].values()))) * 100.0
        rows.append([
            C.short_name(key), C.display_name(key),
            format_mean_std(run.avg_acc, run.avg_acc_std),
            format_mean_std(run.forgetting, run.forgetting_std),
            f"${acc_eff:+.1f}$", f"${for_eff:+.1f}$",
        ])
    tex = booktabs_table(
        headers, rows,
        caption="Component ablations on CIFAR-100 (mean $\\pm$ std over 3 seeds; deltas are matched per-seed against the reference run).",
        label="tab:component_ablations",
    )
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / "component_ablations.tex")
    md_path = save_markdown_table(headers, rows, out_dir / TABLES_OUT / "component_ablations.md",
                                  title="Component ablations (per-seed matched deltas)")
    return [tex_path, md_path]


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_component_comparison(runs, out_dir)
    outputs += fig_attribution(runs, out_dir)
    return outputs


def generate_tables(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return table_component(runs, out_dir)
