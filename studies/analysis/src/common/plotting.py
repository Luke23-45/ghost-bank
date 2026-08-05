"""Reusable publication-grade plotting primitives for the Ghost Bank paper.

These builders are shared by every experiment module so that all figures
share the exact same style, axes finishing and annotation conventions.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.common import constants as C
from src.common.data import RunResult
from src.common.style import APPLE, PALETTE, SERIES_COLORS, finish_axes

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


def percent_axis(ax: plt.Axes, ylim: Optional[Tuple[float, float]] = None) -> None:
    """Format a 0-100 accuracy axis with a percent formatter."""
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    if ylim is not None:
        ax.set_ylim(*ylim)


def styled_legend(ax: plt.Axes, **kwargs) -> None:
    """Legend with the reference framework's frame styling."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 0.92)
    kwargs.setdefault("edgecolor", "#D1D1D6")
    kwargs.setdefault("fancybox", True)
    ax.legend(**kwargs)


def annotate_best(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, key: str) -> None:
    """Label the peak value of a series with its value and key short name."""
    i = int(np.nanargmax(ys))
    ax.annotate(
        f"{ys[i]:.1f}%",
        (xs[i], ys[i]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color=APPLE["ink"],
    )


# ── Per-task final accuracy curve (line + mean/std band) ─────────────
def plot_per_task_curve(
    ax: plt.Axes,
    run: RunResult,
    *,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    band: bool = True,
    annotate: bool = False,
) -> None:
    """Final-state per-task accuracy curve with +-1 std band (mean over seeds)."""
    color = color or C.color_for(run.key)
    marker = marker or C.marker_for(run.key)
    means = run.final_task_accs()
    stds = run.final_task_stds()
    xs = np.asarray(TASK_TICKS, dtype=float)

    if band:
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)
    (line,) = ax.plot(
        xs, means,
        color=color, marker=marker, markersize=5,
        linewidth=1.8, label=label or C.display_name(run.key),
    )
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    if annotate:
        annotate_best(ax, xs, means, run.key)
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
    ax.set_title(title, loc="left")
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
                    color=APPLE["ink"] if v > 0.6 * vmax else "white",
                )
    ax.set_xticks(TASK_TICKS, minor=False)
    return im


def add_colorbar(fig: plt.Figure, im: mpl.image.AxesImage, ax: plt.Axes, label: str = "Task accuracy (%)") -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=APPLE["muted"], labelsize=9)


# ── Grouped bar chart of mean +- std ─────────────────────────────────
def plot_grouped_bars(
    ax: plt.Axes,
    keys: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    *,
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    bar_labels: bool = True,
) -> List[plt.Rectangle]:
    """Vertical bars with error caps, value labels, reference emphasized."""
    colors = list(colors) if colors is not None else [C.color_for(k) for k in keys]
    labels = list(labels) if labels is not None else [C.display_name(k) for k in keys]
    xs = np.arange(len(keys))
    bars = ax.bar(
        xs, means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        capsize=3.5,
        error_kw={"elinewidth": 1.0, "ecolor": APPLE["muted"], "capthick": 1.0},
    )
    if bar_labels:
        for x, m, s in zip(xs, means, stds):
            ax.text(
                x, m + s + 0.8,
                f"{m:.1f}",
                ha="center", va="bottom",
                fontsize=9, color=APPLE["ink"],
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    return bars


# ── Horizontal effect bars (positive/negative deltas) ────────────────
def plot_effect_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    effects: Sequence[float],
    *,
    reference_label: str = "Reference",
) -> None:
    """Horizontal bars of matched per-seed effect sizes (delta accuracy)."""
    order = np.argsort(effects)
    ys = np.arange(len(labels))
    colors = [
        C.color_for(C.ATTRIBUTION_ORDER[int(i)][1]) if i < len(C.ATTRIBUTION_ORDER) else PALETTE["grey"]
        for i in order
    ]
    ax.barh(ys, np.asarray(effects)[order], color=colors, edgecolor="white", linewidth=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[i] for i in order])
    ax.axvline(0, color=APPLE["ink"], linewidth=1.0)
    for y, e in zip(ys, np.asarray(effects)[order]):
        ax.text(
            e + (0.004 if e >= 0 else -0.004),
            y,
            f"{e:+.1f}pp",
            va="center",
            ha="left" if e >= 0 else "right",
            fontsize=9,
            color=APPLE["ink"],
        )
    return None


# ── Trajectory fan chart ─────────────────────────────────────────────
def plot_trajectory_fan(
    ax: plt.Axes,
    run: RunResult,
    *,
    title: str,
    label_every: int = 3,
    annotate_peak: bool = False,
) -> None:
    """One line per task: its accuracy across evaluation times 0..9.

    Each trace starts at its introduction accuracy (bold start marker) and
    ends at its final accuracy (end marker). A fan that collapses reveals
    catastrophic forgetting; a tight parallel bundle reveals stability.
    """
    xs = np.asarray(TASK_TICKS, dtype=float)
    traces = run.trajectories()
    colors = AGE_CMAP(np.linspace(0.15, 1.0, len(traces)))
    for j, trace in enumerate(traces):
        valid = ~np.isnan(trace)
        color = colors[j]
        (line,) = ax.plot(
            xs[valid], trace[valid],
            color=color, linewidth=1.3, alpha=0.9,
        )
        ax.scatter(
            [j], [trace[j]],
            color=color, marker="o", s=42, zorder=5,
            edgecolor="white", linewidth=0.8,
        )
        ax.scatter(
            [len(traces) - 1], [trace[-1]],
            color=color, marker="D", s=34, zorder=5,
            edgecolor="white", linewidth=0.6,
        )
        if j % label_every == 0:
            ax.annotate(
                f"T{j}",
                (j, trace[j]),
                textcoords="offset points",
                xytext=(-4, 8),
                fontsize=8,
                color=color,
            )
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Evaluation time (after training task)")
    ax.set_ylabel("Task accuracy (%)")
    ax.set_title(title, loc="left")
    ax.set_xlim(-0.4, len(traces) - 0.6)
    if annotate_peak:
        pass
    return None


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
    ax.set_title(title, loc="left")
    return None


# ── Forgetting by task age ───────────────────────────────────────────
def plot_forgetting_by_age(
    ax: plt.Axes,
    run: RunResult,
    ref: Optional[RunResult] = None,
    *,
    title: str,
    fill_gradient: bool = True,
) -> None:
    """Per-task forgetting vs task index (0 = oldest). A monotone decline
    toward the last task is the textbook signature of classifier recency bias."""
    xs = np.asarray(TASK_TICKS, dtype=float)
    if ref is not None:
        ax.plot(xs, ref.per_task_forgetting(), color=C.color_for(ref.key),
                marker="o", markersize=5, linewidth=1.5, alpha=0.7,
                label=C.display_name(ref.key))
    f = run.per_task_forgetting()
    ax.plot(xs, f, color=C.color_for(run.key), marker="s", markersize=6, linewidth=2.0,
            label=C.display_name(run.key))
    if fill_gradient:
        for j in TASK_TICKS:
            ax.axvspan(j - 0.4, j + 0.4, color=AGE_CMAP((j + 1) / len(TASK_TICKS)), alpha=0.06)
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting (pp)")
    ax.set_title(title, loc="left")
    ax.axhline(0, color=APPLE["ink"], linewidth=0.8, linestyle=":")
    return None


# ── Per-task delta bars vs reference ─────────────────────────────────
def plot_per_task_delta_bars(
    ax: plt.Axes,
    run: RunResult,
    ref: RunResult,
    *,
    title: str,
    color_negative: bool = True,
    annotate_mean: bool = True,
) -> None:
    """Per-task final-accuracy delta (run minus reference, pp) as bars."""
    xs = np.asarray(TASK_TICKS, dtype=float)
    deltas = run.per_task_delta_vs(ref)
    colors = [C.color_for(run.key) if d >= 0 else PALETTE["rose"] for d in deltas] \
        if color_negative else [C.color_for(run.key)] * len(deltas)
    ax.bar(xs, deltas, color=colors, edgecolor="white", linewidth=1.0)
    for x, d in zip(xs, deltas):
        ax.text(x, d + (0.4 if d >= 0 else -0.4), f"{d:+.1f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
    ax.axhline(0, color=APPLE["ink"], linewidth=1.0)
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Task")
    ax.set_ylabel("Delta accuracy vs reference (pp)")
    ax.set_title(title, loc="left")
    if annotate_mean:
        mean = float(np.mean(deltas))
        ax.text(
            0.01, 0.95, f"mean = {mean:+.1f} pp",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=APPLE["ink"],
            bbox=dict(boxstyle="round,pad=0.3", fc=APPLE["panel"], ec="none"),
        )
    return None


# ── Per-seed consistency strip ───────────────────────────────────────
def plot_seed_consistency_strip(
    ax: plt.Axes,
    run: RunResult,
    ref: RunResult,
    *,
    title: str,
) -> None:
    """Jittered strip of matched per-seed deltas vs the reference.

    For experiments whose penalty is extremely consistent across seeds
    (e.g. a4, seed std 0.0017), the tight cluster IS the finding.
    """
    rng = np.random.default_rng(7)
    deltas = {s: run.per_seed_avg_accs()[s] - ref.per_seed_avg_accs()[s] for s in C.SEEDS}
    vals = np.asarray([deltas[s] * 100.0 for s in C.SEEDS], dtype=float)
    y = rng.uniform(-0.25, 0.25, size=len(vals))
    ax.scatter(vals, y, s=90, color=C.color_for(run.key), edgecolor="white", linewidth=1.2, zorder=5)
    for v, s in zip(vals, C.SEEDS):
        ax.annotate(f"seed {s}", (v, y[np.where(vals == v)[0][0]]),
                    textcoords="offset points", xytext=(10, 0), fontsize=8, color=APPLE["muted"])
    mean, std = float(vals.mean()), float(vals.std())
    ax.axvline(0, color=APPLE["ink"], linewidth=1.2, linestyle=":")
    ax.axvline(mean, color=C.color_for(run.key), linewidth=1.8)
    ax.axvspan(mean - std, mean + std, color=C.color_for(run.key), alpha=0.12)
    ax.text(mean, 0.55, f"mean {mean:+.1f} pp  ($\\sigma$={std:.2f})",
            ha="center", fontsize=10, color=APPLE["ink"],
            bbox=dict(boxstyle="round,pad=0.3", fc=APPLE["panel"], ec="none"))
    ax.set_xlim(-12, 2)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Per-seed delta average accuracy vs reference (pp)")
    ax.set_title(title, loc="left")
    ax.spines["left"].set_visible(False)
    return None


# ── Forgetting accumulation over training ────────────────────────────
def plot_forgetting_accumulation(
    ax: plt.Axes,
    runs: Dict[str, RunResult],
    keys: Sequence[str],
    *,
    title: str,
    emphasize: Optional[str] = None,
) -> None:
    """Cumulative mean forgetting vs evaluation time — the erosion trace."""
    xs = np.asarray(TASK_TICKS, dtype=float)
    for key in keys:
        run = runs[key]
        acc = run.forgetting_accumulation()
        if key == emphasize:
            ax.plot(xs, acc, color=C.color_for(key), marker="o", markersize=6,
                    linewidth=2.4, label=C.display_name(key))
            ax.fill_between(xs, acc, color=C.color_for(key), alpha=0.06)
        else:
            ax.plot(xs, acc, color=C.color_for(key), marker="o", markersize=4,
                    linewidth=1.4, alpha=0.8, label=C.display_name(key))
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Training progress (after task)")
    ax.set_ylabel("Cumulative mean forgetting (pp)")
    ax.set_title(title, loc="left")
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
    ax.set_yticklabels([f"{C.short_name(k)}" for k in keys], fontsize=9)
    ax.set_xticks(TASK_TICKS)
    ax.set_xticklabels([f"T{t}" for t in TASK_TICKS])
    ax.set_xlabel("Task")
    ax.set_title(title, loc="left")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color=APPLE["ink"] if v < 30 else "white")
    ref_row = keys.index(C.reference_key())
    ax.axhline(ref_row - 0.5, color=APPLE["ink"], linewidth=1.6)
    ax.axhline(ref_row + 0.5, color=APPLE["ink"], linewidth=1.6)
    return im


# ── Ranking lollipop ─────────────────────────────────────────────────
def plot_ranking_lollipop(
    ax: plt.Axes,
    runs: Dict[str, RunResult],
    keys: Sequence[str],
    *,
    title: str,
) -> None:
    """Sorted average-accuracy lollipop; the reference stem is emphasized."""
    means = {k: runs[k].avg_acc * 100.0 for k in keys}
    stds = {k: runs[k].avg_acc_std * 100.0 for k in keys}
    order = sorted(keys, key=lambda k: means[k])
    ys = np.arange(len(order))
    for k, y in zip(order, ys):
        is_ref = k == C.reference_key()
        ax.plot([means[k], means[k]], [y - 0.22, y + 0.22],
                color=C.color_for(k), linewidth=3.0 if is_ref else 1.4)
        ax.scatter([means[k]], [y], s=110 if is_ref else 55,
                   color=C.color_for(k), edgecolor="white", linewidth=1.0,
                   zorder=5)
        ax.errorbar([means[k]], [y], xerr=[stds[k]], capsize=2.5,
                    ecolor=APPLE["muted"], linewidth=0.8, zorder=4)
        ax.text(means[k] + 1.0, y, f"{means[k]:.1f}", va="center",
                fontsize=8.5, color=APPLE["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{C.short_name(k)} · {C.display_name(k)}" for k in order], fontsize=9)
    ax.set_xlabel("Average accuracy (%)")
    ax.set_title(title, loc="left")
    ax.set_xlim(20, 55)
    ax.grid(True, axis="x", color=APPLE["grid"], linewidth=0.8, alpha=0.8)
    return None
