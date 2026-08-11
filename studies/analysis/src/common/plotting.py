"""Publication-grade plotting primitives used by the approved paper figures.

These are the ONLY archetypes referenced by ``src/paper`` (Fig 1-5, Fig A1-A4).
The former 38-figure library archetypes (trajectory fans, delta bars, seed
strips, accumulation traces, ranking lollipops, ...) were removed together
with the old per-experiment figure pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.common import constants as C
from src.common.data import RunResult
from src.common.style import APPLE, PALETTE
import seaborn as sns

logger = logging.getLogger(__name__)

TASK_TICKS = list(range(C.NUM_TASKS))

# ── Curated colormaps (no green/cyan tails) ──────────────────────────
# Evolution heatmaps: cream → steel-blue → deep navy (no mako green tail)
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ghost_bank_evo",
    ["#F7F7F0", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    N=256,
)

# Forgetting heatmap: warm cream → salmon → crimson → deep maroon
FORGET_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ghost_bank_forget",
    ["#FFF5EB", "#FDD0A2", "#F16913", "#D62728", "#67000D"],
    N=256,
)

# Sequential colormap for task-age coloring (old = sky, new = indigo)
AGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ghost_bank_age",
    [PALETTE["sky"], PALETTE["indigo"]],
    N=256,
)


# ── Luminance-based contrast (WCAG-grade) ────────────────────────────
def _relative_luminance(hex_color: str) -> float:
    """Compute relative luminance per WCAG 2.0 from a hex color."""
    r, g, b = mcolors.to_rgb(hex_color)
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _text_color_for_bg(cmap, norm_value: float, vmin: float = 0.0, vmax: float = 1.0, threshold: float = 0.40) -> str:
    """Return white or dark text for maximum readability on a colormap cell."""
    normed = (norm_value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    rgba = cmap(np.clip(normed, 0.0, 1.0))
    bg_hex = mcolors.to_hex(rgba[:3])
    lum = _relative_luminance(bg_hex)
    return "#FFFFFF" if lum < threshold else "#1A1A1A"


def styled_legend(ax: plt.Axes, **kwargs) -> None:
    """Legend with the reference framework's frame styling."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 0.92)
    kwargs.setdefault("edgecolor", "#D1D1D6")
    kwargs.setdefault("fancybox", True)
    ax.legend(**kwargs)


def finish_axes(ax: plt.Axes, *, heatmap: bool = False) -> None:
    """Apply consistent spine, grid, and tick styling to an axes.

    Call on every axes in every figure to guarantee visual coherence
    across the entire paper (main + appendix).

    Parameters
    ----------
    heatmap : bool
        When True, skips the y-grid overlay and preserves all four spines
        so that heatmap cell borders are not disrupted.
    """
    if not heatmap:
        ax.grid(True, axis="y", color=APPLE["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        # Heatmaps manage their own grid via minor ticks; just clean spines
        ax.grid(which="major", visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
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

    # Balance visual weight: a square (s) has more area than a circle/diamond of the same size.
    ms = 5.5 if marker == "s" else 7

    if band:
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.20, linewidth=0)
    (line,) = ax.plot(
        xs, means,
        color=color, marker=marker, markersize=ms,
        markeredgecolor="white", markeredgewidth=0.6,
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
    nan_fill: str = "#F0F0F2",
) -> mpl.image.AxesImage:
    """Heatmap of the lower-triangular task-accuracy evolution matrix.

    Row i = evaluation performed after training task i; column j = task j
    accuracy at that point (percent).  NaN cells (upper triangle) are
    filled with a soft neutral to avoid stark white gaps.
    """
    matrix = run.accuracy_matrix * 100.0 if run.accuracy_matrix is not None else None
    if matrix is None:
        raise ValueError(f"{run.key}: aggregated_accuracy_matrix.csv missing")
    vmax = vmax or float(np.nanmax(matrix))

    # Use a copy with NaN masked so imshow leaves those cells transparent
    cmap_copy = HEATMAP_CMAP.copy()
    cmap_copy.set_bad(color=(0, 0, 0, 0))  # transparent for NaN
    display = np.ma.masked_invalid(matrix)
    im = ax.imshow(
        display,
        cmap=cmap_copy,
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
                txt_color = "#FFFFFF"  # Force white text for premium contrast on blues
                ax.text(
                    j, i, f"{v:{fmt}}",
                    ha="center", va="center",
                    fontsize=9, fontweight="medium",
                    color=txt_color,
                )

    # Clean white cell separators
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def add_colorbar(
    fig: plt.Figure,
    im: mpl.image.AxesImage,
    ax: plt.Axes,
    label: str = "Task accuracy (%)",
    shrink: float = 0.92,
) -> mpl.colorbar.Colorbar:
    """Add a refined colorbar matched to the axes height."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=shrink)
    cbar.set_label(label, fontsize=10)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(APPLE["grid"])
    cbar.ax.tick_params(colors=APPLE["muted"], labelsize=9, width=0.4)
    return cbar


# ── Stability slope chart (introduction -> final) ────────────────────
def plot_stability_slopes(
    ax: plt.Axes,
    run: RunResult,
    ref: RunResult,
    *,
    title: str,
    run_label: str,
    ref_label: str = "Uniform Herding",
) -> None:
    """Paired per-task dots: accuracy at introduction vs accuracy at the end.

    Each task contributes a left dot (fresh accuracy) and a right dot (final
    accuracy); the connecting line's slope is that task's forgetting. The
    reference is drawn light underneath so the comparison reads instantly.

    Markers: ● circle = at introduction, ▼ triangle-down = final state.
    This visual encoding makes the forgetting direction immediately obvious.
    """
    xs = np.asarray(TASK_TICKS, dtype=float)
    ref_intro, ref_final = ref.intro_accs(), ref.final_task_accs()
    run_intro, run_final = run.intro_accs(), run.final_task_accs()

    ref_color = C.color_for(ref.key)
    run_color = C.color_for(run.key)

    # ── Connecting slopes (T0..T8) ──
    for j, x in enumerate(xs[:-1]):
        ax.plot(
            [x, x + 0.28], [ref_intro[j], ref_final[j]],
            color=ref_color, alpha=1.0, linewidth=1.2, zorder=1,
        )
        ax.plot(
            [x + 0.38, x + 0.66], [run_intro[j], run_final[j]],
            color=run_color, linewidth=1.8, zorder=1,
        )

    # ── Reference markers (lighter, recedes) ──
    ax.scatter(xs[:-1], ref_intro[:-1],
              color=ref_color, marker="o", s=36, alpha=1.0, zorder=2,
              edgecolors="white", linewidths=0.5,
              label=f"{ref_label} — intro")
    ax.scatter(xs[:-1] + 0.28, ref_final[:-1],
              color=ref_color, marker="v", s=36, alpha=1.0, zorder=2,
              edgecolors="white", linewidths=0.5,
              label=f"{ref_label} — final")

    # ── Ablation markers (prominent) ──
    ax.scatter(xs[:-1] + 0.38, run_intro[:-1],
              color=run_color, marker="o", s=52, zorder=2,
              edgecolors="white", linewidths=0.6,
              label=f"{run_label} — intro")
    ax.scatter(xs[:-1] + 0.66, run_final[:-1],
              color=run_color, marker="v", s=52, zorder=2,
              edgecolors="white", linewidths=0.6,
              label=f"{run_label} — final")

    # ── T9 Single Evaluation Points ──
    # For T9, intro == final. Plot a single centered marker per method to avoid zero-length dumbbells.
    t9 = xs[-1]
    ax.scatter([t9 + 0.14], [ref_intro[-1]], color=ref_color, marker="o", s=36, alpha=1.0, zorder=2, edgecolors="white", linewidths=0.5)
    ax.scatter([t9 + 0.52], [run_intro[-1]], color=run_color, marker="o", s=52, zorder=2, edgecolors="white", linewidths=0.6)

    ax.set_xticks(xs + 0.33)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlim(-0.4, len(xs) - 0.1)
    ax.set_xlabel("Task")
    ax.set_ylabel("Task accuracy (%)")
    ax.set_title(title, loc="center")
    return None


# ── Forgetting by task age ───────────────────────────────────────────
def plot_forgetting_by_age(
    ax: plt.Axes,
    run: RunResult,
    ref: Optional[RunResult] = None,
    *,
    title: Optional[str] = None,
    fill_gradient: bool = True,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    linewidth: float = 2.0,
    alpha: float = 1.0,
) -> mpl.lines.Line2D:
    """Per-task forgetting vs task index (0 = oldest)."""
    xs = np.asarray(TASK_TICKS, dtype=float)
    if ref is not None:
        ax.plot(xs, ref.per_task_forgetting(), color=C.color_for(ref.key),
                marker="o", markersize=5, linewidth=1.5, alpha=0.7,
                label=C.display_name(ref.key), zorder=4)
    f = run.per_task_forgetting()
    
    color = color or C.color_for(run.key)
    marker = marker or C.marker_for(run.key)
    if label is None:
        resolved_label = "_nolegend_"
    elif label == "":
        resolved_label = C.display_name(run.key)
    else:
        resolved_label = label

    ms = 5.5 if marker == "s" else 6
    (line,) = ax.plot(xs, f, color=color, marker=marker, markersize=ms,
            markeredgecolor="white", markeredgewidth=0.6,
            linewidth=linewidth, label=resolved_label, alpha=alpha, zorder=3)
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
    vmin_f, vmax_f = 0, 60
    matrix = np.vstack([runs[k].per_task_forgetting() for k in keys])
    im = ax.imshow(matrix, cmap=FORGET_CMAP, vmin=vmin_f, vmax=vmax_f, aspect="auto")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([C.display_name(k) for k in keys], fontsize=9)
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Task")
    ax.set_title(title, loc="center")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt_color = _text_color_for_bg(FORGET_CMAP, v, vmin=vmin_f, vmax=vmax_f, threshold=0.18)
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=9, fontweight="medium", color=txt_color)

    # Clean white cell separators
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Highlight reference row with bold typography
    ref_row = keys.index(C.reference_key())
    ax.get_yticklabels()[ref_row].set_weight("bold")
    ax.get_yticklabels()[ref_row].set_color("#1A1A1A")

    return im
