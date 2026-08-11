"""Main manuscript figures (Fig 1-5) for the Ghost Bank CIL paper.

Every builder reads directly from the persisted run artifacts via the
verified data layer (``src/common/data.py``) — no hard-coded numbers.
The composition (panels, titles, emphases) follows the approved plan
(``docs/paper/analysis/analysis_plan.md``, Section 4).

Output: ``outputs/paper/main/figures/`` (PDF + PNG per figure).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common import constants as C
from src.common.data import RunResult, matched_deltas
from src.common.plotting import (
    finish_axes,
    plot_forgetting_by_age,
    plot_per_task_curve,
    styled_legend,
)
from src.common.style import (
    APPLE,
    PALETTE,
    FONT_SIZE_TICK,
    FONT_SIZE_LABEL,
    FONT_SIZE_ANNOT,
    apply_thesis_style,
    create_figure,
    save_figure,
)

logger = logging.getLogger(__name__)

REF = C.reference_key()
MAIN_OUT = C.PAPER_MAIN_FIGURES_DIR


# ── Fig 1: per-task final accuracy ───────────────────────────────────
def fig1_per_task_accuracy(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """2 panels: (a) baselines B0/B1/B2/B3; (b) reference vs a1-a4.

    A single figure-level legend sits below both panels so every entry
    appears exactly once.  The reference line (B3 / uniform herding) is
    emphasised with a heavier stroke; non-reference baselines are softer.
    """
    with apply_thesis_style():
        # Aspect reduced from 0.72 → 0.55 so the double-column canvas
        # Aspect reduced from 0.72 → 0.45 so the double-column canvas
        # (6.5" wide) yields a ~3.6" tall panel pair — better balanced and
        # less vertically stretched once the shared legend is added below.
        fig, axes = create_figure(width="double", nrows=1, ncols=2, aspect=0.45)
        ax_a, ax_b = axes

        # Panel (a): baselines -------------------------------------------------
        # No shaded std bands in the paper main figures — keep the lines
        # clean and uncluttered (uncertainty is reported in Fig 3 / appendix).
        for key in C.BASELINE_KEYS:
            lw = 2.4 if key == REF else 1.6
            alpha_val = 1.0 if key == REF else 0.85
            line = plot_per_task_curve(
                ax_a, runs[key],
                label=C.display_name(key),
                band=False,
            )
            line.set_linewidth(lw)
            line.set_alpha(alpha_val)
        ax_a.set_xlabel("Task\n", color=APPLE["ink"])
        ax_a.set_ylabel("Final task accuracy (%)", color=APPLE["ink"])
        ax_a.set_title("(a) Baselines", loc="center")
        ax_a.set_ylim(0, 80)
        finish_axes(ax_a)

        # Panel (b): component ablations ---------------------------------------
        # B3 already labelled in panel (a); plot it unlabelled here so the
        # reference line is visible but the legend stays clean.
        for key in C.COMPONENT_KEYS:
            lw = 2.4 if key == REF else 1.5
            lbl = None if key == REF else C.display_name(key)
            line = plot_per_task_curve(
                ax_b, runs[key],
                label=lbl,
                band=False,
            )
            line.set_linewidth(lw)
        ax_b.set_xlabel("Task\n", color=APPLE["ink"])
        # Shared y-label from panel (a); omit to save horizontal space
        ax_b.set_title("(b) Component ablations", loc="center")
        ax_b.set_ylim(0, 80)
        finish_axes(ax_b)

        # Unified legend below both panels (all unique handles) ----------------
        handles_a, labels_a = ax_a.get_legend_handles_labels()
        handles_b, labels_b = ax_b.get_legend_handles_labels()
        all_handles = handles_a + handles_b
        all_labels = labels_a + labels_b
        fig.subplots_adjust(bottom=-0.35)
        fig.legend(
            all_handles, all_labels,
            loc="outside lower center",
            ncol=4,
            bbox_to_anchor=(0.54, -0.18),
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            edgecolor="#D1D1D6",
            fancybox=True,
            columnspacing=1.5,
            handletextpad=0.6,
        )

        return save_figure(fig, out_dir / MAIN_OUT / "fig1_per_task_accuracy", bbox_inches="tight")


# ── Fig 2: component attribution ─────────────────────────────────────
def fig2_component_attribution(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """2 panels: matched per-seed delta accuracy and delta forgetting (a1-a4)."""
    ablation_keys = [k for k in C.COMPONENT_KEYS if k != REF]
    deltas = matched_deltas(runs, REF)
    acc_effects = [
        float(np.mean(list(deltas[k]["avg_acc"].values()))) * 100.0 for k in ablation_keys
    ]
    for_effects = [
        float(np.mean(list(deltas[k]["forgetting"].values()))) * 100.0 for k in ablation_keys
    ]

    with apply_thesis_style():
        # Panel (a) range: -14 to 2 (16 units). Panel (b) range: 0 to 38 (38 units).
        # Ratio = 16 : 38 = 1 : 2.375 to enforce a 1:1 physical aspect ratio for the units.
        fig, axes = create_figure(
            width="double", nrows=1, ncols=2, aspect=0.55,
            gridspec_kw={"width_ratios": [1, 2.375]}
        )
        ax1, ax2 = axes

        order = np.argsort(acc_effects)
        ys = np.arange(len(ablation_keys))
        
        # Panel (a)
        ax1.barh(ys, np.asarray(acc_effects)[order],
                 color=[C.color_for(ablation_keys[i]) for i in order],
                 edgecolor="white", linewidth=1.2)
        ax1.set_yticks(ys)
        ax1.set_yticklabels(
            [C.display_name(ablation_keys[i]) for i in order],
            fontsize=FONT_SIZE_LABEL,
        )
        ax1.axvline(0, color=APPLE["ink"], linewidth=1.2)
        ax1.set_xlabel("Delta (pp)", x=-0.1)
        ax1.set_title("(a) Delta average accuracy (pp)", x=-0.1)
        ax1.set_xlim(-14, 2)
        ax1.set_xticks([-10, -5, 0])
        for y, e in zip(ys, np.asarray(acc_effects)[order]):
            ax1.text(e + (0.3 if e >= 0 else -0.3), y, f"{e:+.1f}",
                     va="center", ha="left" if e >= 0 else "right",
                     fontsize=FONT_SIZE_TICK)

        # Panel (b) - Uses same 'order' for consistent categorical comparison
        ax2.barh(ys, np.asarray(for_effects)[order],
                 color=[C.color_for(ablation_keys[i]) for i in order],
                 edgecolor="white", linewidth=1.2)
        ax2.set_yticks(ys)
        ax2.set_yticklabels([])
        ax2.axvline(0, color=APPLE["ink"], linewidth=1.2)
        ax2.set_xlabel("Delta (pp)")
        ax2.set_title("(b) Delta forgetting (pp)", loc="center")
        ax2.set_xlim(0, 38)
        ax2.set_xticks([0, 10, 20, 30])
        for y, e in zip(ys, np.asarray(for_effects)[order]):
            ax2.text(e + (0.4 if e >= 0 else -0.4), y, f"{e:+.1f}",
                     va="center", ha="left" if e >= 0 else "right",
                     fontsize=FONT_SIZE_TICK)

        for ax in (ax1, ax2):
            finish_axes(ax)
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

        return save_figure(fig, out_dir / MAIN_OUT / "fig2_component_attribution")



# ── Fig 3: resource sensitivity (2x2) ────────────────────────────────
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
    xs = np.asarray(xs)
    order = np.argsort(xs)
    return (xs[order], np.asarray(accs)[order], np.asarray(acc_stds)[order],
            np.asarray(fors)[order], np.asarray(for_stds)[order])


def fig3_resource_sensitivity(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """2x2 grid: {memory, retrieval} x {accuracy, forgetting}."""
    panels = [
        ("memory", "Memory budget (exemplars)", C.SENSITIVITY_MEMORY_KEYS),
        ("retrieval", "Retrieval budget (exemplars / step)", C.SENSITIVITY_RETR_KEYS),
    ]
    series = {axis: _resource_series(runs, keys, axis) for axis, _, keys in panels}

    with apply_thesis_style():
        fig, axes = create_figure(width="double", nrows=2, ncols=2, aspect=0.95)
        for row, (axis, xlabel, _keys) in enumerate(panels):
            xs, accs, acc_stds, fors, for_stds = series[axis]
            pad = 0.15 * np.ptp(xs) if np.ptp(xs) > 0 else 1.0

            ax = axes[row, 0]
            ax.errorbar(xs, accs, yerr=acc_stds, color=C.color_for(REF),
                        marker="o", markersize=6, linewidth=2.0, capsize=4, elinewidth=1.2)
            ax.set_xlabel(xlabel + "\n" if row == 0 else xlabel)
            ax.set_ylabel("Average accuracy (%)")
            ax.set_title(f"({'a' if axis == 'memory' else 'c'}) {axis.title()} budget — Accuracy", loc="center")
            if axis == "retrieval":
                ax.set_xscale("log", base=2)
                ax.set_xlim(24, 170)
                ax.set_xticks([32, 64, 128])
                ax.set_xticklabels(["32", "64", "128"])
                ax.set_ylim(40, 50)
            elif axis == "memory":
                ax.set_xlim(xs.min() - pad, xs.max() + pad)
                ax.set_xticks([500, 2000, 4000])

            ax = axes[row, 1]
            ax.errorbar(xs, fors, yerr=for_stds, color=PALETTE["wine"],
                        marker="s", markersize=6, linewidth=2.0, capsize=4, elinewidth=1.2)
            ax.set_xlabel(xlabel + "\n" if row == 0 else xlabel)
            ax.set_ylabel("Forgetting (%)")
            ax.set_title(f"({'b' if axis == 'memory' else 'd'}) {axis.title()} budget — Forgetting", loc="center")
            if axis == "retrieval":
                ax.set_xscale("log", base=2)
                ax.set_xlim(24, 170)
                ax.set_xticks([32, 64, 128])
                ax.set_xticklabels(["32", "64", "128"])
                ax.set_ylim(11, 19)
                ax.set_yticks([12, 14, 16, 18])
            elif axis == "memory":
                ax.set_xlim(xs.min() - pad, xs.max() + pad)
                ax.set_xticks([500, 2000, 4000])

            finish_axes(axes[row, 0])
            finish_axes(axes[row, 1])

        return save_figure(fig, out_dir / MAIN_OUT / "fig3_resource_sensitivity")


# ── Fig 4: accuracy vs forgetting scatter with error bars ────────────
def fig4_acc_forgetting_scatter(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """All 12 runs: forgetting (x) vs average accuracy (y) trade-off space."""
    with apply_thesis_style():
        fig, ax = create_figure(width="double", aspect=0.9)
        
        # 1. Collect points
        points = []
        for key in C.MASTER_ORDER:
            f = runs[key].forgetting * 100.0
            a = runs[key].avg_acc * 100.0
            points.append((f, a, key))
            
        # 2. Scatter plot without error bars
        for f, a, key in points:
            is_ref = (key == REF)
            marker = C.marker_for(key)
            
            # Boost size of spiky markers to match visual weight of dense shapes
            if marker in ['*', 'X', 'x', '+', 'P']:
                size = 300
            elif is_ref:
                size = 280
            else:
                size = 220
                
            # Explicitly elevate zorder of overlapping/tiny markers
            z = 5 if key == "a3_linear_head" else (3 if is_ref else 4)
            
            ax.scatter(f, a, s=size, color=C.color_for(key),
                       marker=marker, edgecolor="white", linewidth=1.2,
                       zorder=z,
                       label=C.display_name(key))
                       
        # 3. Add directional arrow
        ax.annotate("Better performance", xy=(0.02, 0.95), xytext=(0.10, 0.85),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", color=APPLE["muted"], lw=1.5),
                    fontsize=FONT_SIZE_ANNOT, color=APPLE["muted"], ha="center", va="center")

        ax.set_xlabel("Forgetting (%)")
        ax.set_ylabel("Average accuracy (%)")
        ax.set_title("Accuracy vs. Forgetting Trade-off", loc="center")
        ax.set_ylim(25, 55)
        ax.set_xlim(0, 60)
        finish_axes(ax)
        
        # 4. Logical Legend Grouping
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {lbl: hdl for lbl, hdl in zip(labels, handles)}
        
        from matplotlib.lines import Line2D
        dummy_handle = Line2D([], [], linestyle='none')
        label_to_handle[""] = dummy_handle
        
        # We want 3 columns: Core, Ablations, Sweeps
        target_cols = [
            ["Uniform Herding", "No replay", "iCaRL", "Static bank"],
            ["Without KD", "Head-logit evaluation", "Linear head", "Random selection"],
            ["Active budget 500", "Active budget 4000", "Retrieval 32", "Retrieval 128"]
        ]
        
        # Matplotlib natively fills legends column-by-column in a top-down fashion.
        # By providing a flat list of exactly 12 items (padded with our dummy handle in Col 1),
        # it will perfectly slice them into 3 columns of 4 rows.
        ordered_labels = target_cols[0] + target_cols[1] + target_cols[2]
            
        ordered_handles = [label_to_handle[lbl] for lbl in ordered_labels if lbl in label_to_handle]
        ordered_labels_present = [lbl for lbl in ordered_labels if lbl in label_to_handle]
        
        # Use tighter columnspacing to prevent collision, and labelspacing/markerscale to fix vertical overlap
        styled_legend(ax, handles=ordered_handles, labels=ordered_labels_present, loc="upper right", ncol=3, fontsize=8, columnspacing=1.0, labelspacing=1.5, markerscale=0.7)
        
        return save_figure(fig, out_dir / MAIN_OUT / "fig4_acc_forgetting_scatter")


# ── Fig 5: failure modes — forgetting by task age ────────────────────
def fig5_forgetting_by_age(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    """2 panels: (a) baselines; (b) component ablations.
    
    Mirrors the structure of Figure 1, but plots per-task forgetting
    instead of final accuracy.
    """
    with apply_thesis_style():
        fig, axes = create_figure(width="double", nrows=1, ncols=2, aspect=0.45)
        ax_a, ax_b = axes

        # Panel (a): baselines -------------------------------------------------
        for i, key in enumerate(C.BASELINE_KEYS):
            lw = 2.4 if key == REF else 1.6
            alpha_val = 1.0 if key == REF else 0.85
            line = plot_forgetting_by_age(
                ax_a, runs[key],
                label=C.display_name(key),
                fill_gradient=(i == 0),
                linewidth=lw,
                alpha=alpha_val,
                show_xlabel=False,
                show_ylabel=False,
            )
        ax_a.set_xlabel("Task\n", color=APPLE["ink"])
        ax_a.set_ylabel("Forgetting (pp)", color=APPLE["ink"])
        ax_a.set_title("(a) Baselines", loc="center")
        ax_a.set_ylim(-3, 65)
        finish_axes(ax_a)

        # Panel (b): component ablations ---------------------------------------
        for i, key in enumerate(C.COMPONENT_KEYS):
            lw = 2.4 if key == REF else 1.5
            lbl = None if key == REF else C.display_name(key)
            line = plot_forgetting_by_age(
                ax_b, runs[key],
                label=lbl,
                fill_gradient=(i == 0),
                linewidth=lw,
                show_xlabel=False,
                show_ylabel=False,
            )
        ax_b.set_xlabel("Task\n", color=APPLE["ink"])
        ax_b.set_title("(b) Component ablations", loc="center")
        ax_b.set_ylim(-3, 65)
        finish_axes(ax_b)

        # Unified legend below both panels -------------------------------------
        handles_a, labels_a = ax_a.get_legend_handles_labels()
        handles_b, labels_b = ax_b.get_legend_handles_labels()
        all_handles = handles_a + handles_b
        all_labels = labels_a + labels_b
        
        fig.subplots_adjust(bottom=-0.35)
        fig.legend(
            all_handles, all_labels,
            loc="outside lower center",
            ncol=4,
            bbox_to_anchor=(0.54, -0.18),
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            edgecolor="#D1D1D6",
            fancybox=True,
            columnspacing=1.5,
            handletextpad=0.6,
        )

        return save_figure(fig, out_dir / MAIN_OUT / "fig5_forgetting_by_age", bbox_inches="tight")


# ── Registry (keys match C.PAPER_MAIN_FIGURES) ───────────────────────
BUILDERS: Dict[str, object] = {
    "fig1_per_task_accuracy": fig1_per_task_accuracy,
    "fig2_component_attribution": fig2_component_attribution,
    "fig3_resource_sensitivity": fig3_resource_sensitivity,
    "fig4_acc_forgetting_scatter": fig4_acc_forgetting_scatter,
    "fig5_forgetting_by_age": fig5_forgetting_by_age,
}


def generate_figures(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    for name in C.PAPER_MAIN_FIGURES:
        outputs += BUILDERS[name](runs, out_dir)
    return outputs
