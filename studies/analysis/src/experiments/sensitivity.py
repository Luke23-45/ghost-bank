"""Resource-sensitivity family-level curves (memory + retrieval) + table.

Per-experiment bespoke figures live in s1_budget500.py, s2_budget4000.py,
s3_retr32.py, s4_retr128.py.  This module only handles the resource
sensitivity curves and the family table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common import constants as C
from src.common.data import RunResult
from src.common.latex import booktabs_table, format_mean_std, save_latex_table, save_markdown_table
from src.common.style import APPLE, apply_thesis_style, create_figure, save_figure

logger = logging.getLogger(__name__)

FAMILY = "sensitivity"
FIGURES_OUT = C.family_output_dir(FAMILY) / "figures"
TABLES_OUT = C.family_output_dir(FAMILY) / "tables"

REF = C.reference_key()


def _resource_series(runs: Dict[str, RunResult], keys: List[str], axis: str):
    xs, accs, acc_stds, fors, for_stds = [], [], [], [], []
    for key in keys:
        run = runs[key]
        x = run.memory_budget() if axis == "memory" else run.retrieval_budget()
        if x is None:
            continue
        xs.append(float(x))
        accs.append(run.avg_acc * 100.0)
        acc_stds.append(run.avg_acc_std * 100.0)
        fors.append(run.forgetting * 100.0)
        for_stds.append(run.forgetting_std * 100.0)
    return (np.asarray(xs), np.asarray(accs), np.asarray(acc_stds),
            np.asarray(fors), np.asarray(for_stds))


def _resource_figure(runs, keys, axis, xlabel, title, out_dir, filename):
    xs, accs, acc_stds, fors, for_stds = _resource_series(runs, keys, axis)
    order = np.argsort(xs)
    xs, accs, acc_stds, fors, for_stds = (xs[order], accs[order], acc_stds[order],
                                           fors[order], for_stds[order])
    with apply_thesis_style():
        fig, axes = create_figure(width="double", nrows=1, ncols=2, aspect=0.5)
        ax1, ax2 = axes

        ax1.errorbar(xs, accs, yerr=acc_stds, color=C.color_for(REF), marker="o",
                     markersize=6, linewidth=2.0, capsize=4, elinewidth=1.2)
        for x, y in zip(xs, accs):
            ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=APPLE["ink"])
        ax1.set_xlabel(xlabel); ax1.set_ylabel("Average accuracy (%)")
        ax1.set_title("(a) Average accuracy", loc="left")
        pad = 0.15 * np.ptp(xs) if np.ptp(xs) > 0 else 1.0
        ax1.set_xlim(xs.min() - pad, xs.max() + pad)

        ax2.errorbar(xs, fors, yerr=for_stds, color=C.color_for("s1_budget500"), marker="s",
                     markersize=6, linewidth=2.0, capsize=4, elinewidth=1.2)
        for x, y in zip(xs, fors):
            ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=9, color=APPLE["ink"])
        ax2.set_xlabel(xlabel); ax2.set_ylabel("Forgetting (%)")
        ax2.set_title("(b) Forgetting", loc="left")
        ax2.set_xlim(xs.min() - pad, xs.max() + pad)

        for ax in (ax1, ax2):
            ax.grid(True, axis="y", color=APPLE["grid"], linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(APPLE["grid"]); ax.spines["bottom"].set_color(APPLE["grid"])
            ax.tick_params(colors=APPLE["muted"])
        fig.suptitle(title, fontsize=13, color=APPLE["ink"])
        return save_figure(fig, out_dir / FIGURES_OUT / filename)


def fig_memory_curve(runs, out_dir):
    return _resource_figure(runs, C.SENSITIVITY_MEMORY_KEYS, "memory",
                            "Memory budget (exemplars)", "Memory budget sensitivity",
                            out_dir, "memory_curve")


def fig_retrieval_curve(runs, out_dir):
    return _resource_figure(runs, C.SENSITIVITY_RETR_KEYS, "retrieval",
                            "Retrieval budget (exemplars / step)", "Retrieval budget sensitivity",
                            out_dir, "retrieval_curve")


def table_sensitivity(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "Axis", "Value", "avg_acc", "forgetting"]
    rows = []
    for key in C.SENSITIVITY_KEYS:
        run = runs[key]
        if run.memory_budget() is not None and key in C.SENSITIVITY_MEMORY_KEYS:
            axis, value = "Memory budget", run.memory_budget()
        else:
            axis, value = "Retrieval budget", run.retrieval_budget()
        rows.append([C.short_name(key), C.display_name(key), axis, str(value),
                     format_mean_std(run.avg_acc, run.avg_acc_std),
                     format_mean_std(run.forgetting, run.forgetting_std)])
    tex = booktabs_table(headers, rows,
                         caption="Resource-sensitivity results on CIFAR-100 (mean $\\pm$ std over 3 seeds).",
                         label="tab:sensitivity")
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / "sensitivity.tex")
    md_path = save_markdown_table(headers, rows, out_dir / TABLES_OUT / "sensitivity.md",
                                  title="Resource sensitivity (s1-s4)")
    return [tex_path, md_path]


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    outputs += fig_memory_curve(runs, out_dir)
    outputs += fig_retrieval_curve(runs, out_dir)
    return outputs


def generate_tables(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    return table_sensitivity(runs, out_dir)
