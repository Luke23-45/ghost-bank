"""Publication-grade plotting primitives used by the approved paper figures.

These are the ONLY archetypes referenced by ``src/paper`` (Fig 1-5, Fig A1-A4).
The former 38-figure library archetypes (trajectory fans, delta bars, seed
strips, accumulation traces, ranking lollipops, ...) were removed together
with the old per-experiment figure pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.common import constants as C
from src.common.data import RunResult
from src.common.style import APPLE, PALETTE

logger = logging.getLogger(__name__)

TASK_TICKS = list(range(C.NUM_TASKS))

# Sequential colormap for evolution heatmaps (colorblind-safe, sky -> indigo)
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ghost_bank_heat",
    ["#FFFFFF", PALETTE["sky"], PALETTE["indigo"]],
    N=256,
)

# Diverging colormap for forgetting (white -> rose -> wine; dark = catastrophic)
FORGET_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ghost_bank_forget",
    ["#FFFFFF", PALETTE["rose"], PALETTE["wine"]],
    N=256,
)

# Sequential colormap for task-age coloring (old = sky, new = indigo)
AGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ghost_bank_age",
    [PALETTE["sky"], PALETTE["indigo"]],
    N=256,
)


def styled_legend(ax: plt.Axes, **kwargs) -> None:
    """Legend with the reference framework's frame styling."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 0.92)
    kwargs.setdefault("edgecolor", "#D1D1D6")
    kwargs.setdefault("fancybox", True)
    ax.legend(**kwargs)


def finish_axes(ax: plt.Axes) -> None:
    """Apply consistent spine, grid, and tick styling to an axes.

    Call on every axes in every figure to guarantee visual coherence
    across the entire paper (main + appendix).
    """
    ax.grid(True, axis="y", color=APPLE["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(APPLE["grid"])
    ax.spines["bottom"].set_color(APPLE["grid"])
    ax.tick_params(colors=APPLE["grid"], labelcolor=APPLE["ink"])


# ── Per-task final accuracy curve (line + mean/std band) ─────────────
def plot_per_task_curve(
    ax: plt.Axes,
    run: RunResult,
    *,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    band: bool = True,
) -> mpl.lines.Line2D:
    """Final-state per-task accuracy curve with +-1 std band (mean over seeds).

    Pass ``label=None`` to suppress the legend entry (useful when the same
    series is plotted in multiple panels and should only appear once).
    """
    color = color or C.color_for(run.key)
    marker = marker or C.marker_for(run.key)
    resolved_label: Optional[str]
    if label is None:
        resolved_label = "_nolegend_"
    elif label == "":
        resolved_label = C.display_name(run.key)
    else:
        resolved_label = label
    means = run.final_task_accs()
    stds = run.final_task_stds()
    xs = np.asarray(TASK_TICKS, dtype=float)

    if band:
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.20, linewidth=0)
    (line,) = ax.plot(
        xs, means,
        color=color, marker=marker, markersize=7,
        linewidth=1.8, label=resolved_label,
    )
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlim(TASK_TICKS[0] - 0.4, TASK_TICKS[-1] + 0.4)
    return line


# ── Evolution heatmap (10x10 lower-triangular task matrix) ───────────
def plot_evolution_heatmap(
    ax: plt.Axes,
    run: RunResult,
    *,
    title: str,
    vmax: Optional[float] = None,
    fmt: str = ".0f",
    annotate: bool = True,
) -> mpl.image.AxesImage:
    """Heatmap of the lower-triangular task-accuracy evolution matrix.

    Row i = evaluation performed after training task i; column j = task j
    accuracy at that point (percent).
    """
    matrix = run.accuracy_matrix * 100.0 if run.accuracy_matrix is not None else None
    if matrix is None:
        raise ValueError(f"{run.key}: aggregated_accuracy_matrix.csv missing")
    vmax = vmax or float(np.nanmax(matrix))
    im = ax.imshow(
        matrix,
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(TASK_TICKS)
    ax.set_yticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_yticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Task")
    ax.set_ylabel("Evaluated after training")
    ax.set_title(title, loc="center")
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j, i, f"{v:{fmt}}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if v > 0.5 * vmax else APPLE["ink"],
                )
    ax.set_xticks(TASK_TICKS, minor=False)
    return im


def add_colorbar(fig: plt.Figure, im: mpl.image.AxesImage, ax: plt.Axes, label: str = "Task accuracy (%)") -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=APPLE["muted"], labelsize=9)


# ── Stability slope chart (introduction -> final) ────────────────────
def plot_stability_slopes(
    ax: plt.Axes,
    run: RunResult,
    ref: RunResult,
    *,
    title: str,
    run_label: str,
    ref_label: str = "Reference",
) -> None:
    """Paired per-task dots: accuracy at introduction vs accuracy at the end.

    Each task contributes a left dot (fresh accuracy) and a right dot (final
    accuracy); the connecting line's slope is that task's forgetting. The
    reference is drawn light underneath so the comparison reads instantly.
    """
    xs = np.asarray(TASK_TICKS, dtype=float)
    ref_intro, ref_final = ref.intro_accs(), ref.final_task_accs()
    run_intro, run_final = run.intro_accs(), run.final_task_accs()

    for j, x in enumerate(xs):
        ax.plot(
            [x, x + 0.28], [ref_intro[j], ref_final[j]],
            color=C.color_for(ref.key), alpha=0.35, linewidth=1.2, zorder=2,
        )
        ax.plot(
            [x + 0.34, x + 0.62], [run_intro[j], run_final[j]],
            color=C.color_for(run.key), linewidth=1.6, zorder=3,
        )
    ax.scatter(xs, ref_intro, color=C.color_for(ref.key), s=26, alpha=0.5, zorder=4, label=f"{ref_label} — at introduction")
    ax.scatter(xs + 0.28, ref_final, color=C.color_for(ref.key), s=26, alpha=0.5, zorder=4, label=f"{ref_label} — final")
    ax.scatter(xs + 0.34, run_intro, color=C.color_for(run.key), s=42, zorder=5, label=f"{run_label} — at introduction")
    ax.scatter(xs + 0.62, run_final, color=C.color_for(run.key), s=42, zorder=5, label=f"{run_label} — final")
    ax.set_xticks(xs + 0.31)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlim(-0.5, len(xs) - 0.3)
    ax.set_ylabel("Task accuracy (%)")
    ax.set_title(title, loc="center")
    return None


# ── Forgetting by task age ───────────────────────────────────────────
def plot_forgetting_by_age(
    ax: plt.Axes,
    run: RunResult,
    ref: Optional[RunResult] = None,
    *,
    title: str,
    fill_gradient: bool = True,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    """Per-task forgetting vs task index (0 = oldest).

    A monotone decline toward the last task is the steady-erosion signature
    of classifier recency bias (iCaRL); a high plateau across the old tasks
    with a tail on the newest three is the recency-anchored collapse (a2).
    """
    xs = np.asarray(TASK_TICKS, dtype=float)
    if ref is not None:
        ax.plot(xs, ref.per_task_forgetting(), color=C.color_for(ref.key),
                marker="o", markersize=5, linewidth=1.5, alpha=0.7,
                label=C.display_name(ref.key), zorder=4)
    f = run.per_task_forgetting()
    ax.plot(xs, f, color=C.color_for(run.key), marker="s", markersize=6, linewidth=2.0,
            label=C.display_name(run.key), zorder=3)
    if fill_gradient:
        for j in TASK_TICKS:
            ax.axvspan(j - 0.4, j + 0.4, color=AGE_CMAP((j + 1) / len(TASK_TICKS)), alpha=0.06)
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    if show_xlabel:
        ax.set_xlabel("Task")
    if show_ylabel:
        ax.set_ylabel("Forgetting (pp)")
    ax.set_title(title, loc="center")
    ax.plot([xs[0], xs[-1]], [0, 0], color=APPLE["ink"], linewidth=0.8, linestyle=":", zorder=1)
    return None


# ── Method x task forgetting heatmap ─────────────────────────────────
def plot_method_task_forgetting_heatmap(
    ax: plt.Axes,
    runs: Dict[str, RunResult],
    keys: Sequence[str],
    *,
    title: str,
) -> mpl.image.AxesImage:
    """11 x 10 heatmap of per-task forgetting (pp) across all methods.

    Rows are methods (paper ordering), columns are tasks. The dark band
    across B2/a2 rows is the 'catastrophic retention' story at a glance.
    """
    matrix = np.vstack([runs[k].per_task_forgetting() for k in keys])
    im = ax.imshow(matrix, cmap=FORGET_CMAP, vmin=0, vmax=60, aspect="auto")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([C.display_name(k) for k in keys], fontsize=9)
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Task")
    ax.set_title(title, loc="center")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color=APPLE["ink"] if v < 30 else "white")
    ref_row = keys.index(C.reference_key())
    ax.axhline(ref_row - 0.5, color=APPLE["ink"], linewidth=1.6)
    ax.axhline(ref_row + 0.5, color=APPLE["ink"], linewidth=1.6)
    return im
