"""Publication-grade matplotlib style system for the Ghost Bank CIL paper.

The palette, typography and layout presets are the exact style used by the
referenced analysis framework (studies/analysis/references/analysis1/common/style.py):

- Colorblind-safe Wong & Tol qualitative palette.
- Apple-inspired ink/grid/panel neutrals.
- Serif (Times) or sans-serif (Aptos/Segoe UI) typography.
- Single-column (4.5"), double-column (6.5") and full-width (13.2") presets.
- 300 DPI raster + vector (PDF) exports with embedded (type-42) fonts.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── Column widths (inches) ───────────────────────────────────────────
COL_WIDTH = 4.5     # single column
TEXT_WIDTH = 6.5    # double column / full text width
FULL_WIDTH = 13.2   # wide multi-panel canvas
DPI = 300

# ── Font sizes (pt) ──────────────────────────────────────────────────
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOT = 9

# ── Colorblind-safe palettes (Wong & Tol) ────────────────────────────
PALETTE = {
    "indigo":   "#332288",
    "teal":     "#44AA99",
    "green":    "#117733",
    "olive":    "#999933",
    "sand":     "#DDCC77",
    "rose":     "#CC6677",
    "wine":     "#882255",
    "purple":   "#AA4499",
    "sky":      "#88CCEE",
    "pink":     "#EE3377",
    "grey":     "#BBBBBB",
    "dark":     "#333333",
}

# Apple-inspired neutrals (matching the reference framework)
APPLE = {
    "ink":    "#1D1D1F",
    "muted":  "#6E6E73",
    "grid":   "#E5E5EA",
    "blue":   "#007AFF",
    "teal":   "#00A6A6",
    "green":  "#34C759",
    "orange": "#FF9500",
    "red":    "#FF3B30",
    "indigo": "#5856D6",
    "pink":   "#FF2D55",
    "panel":  "#F5F5F7",
}

# Canonical ordered color cycle for series in this paper (colorblind-safe)
SERIES_COLORS = [
    PALETTE["indigo"],
    PALETTE["teal"],
    PALETTE["green"],
    PALETTE["rose"],
    PALETTE["olive"],
    PALETTE["wine"],
    PALETTE["sky"],
    PALETTE["purple"],
    PALETTE["pink"],
    PALETTE["sand"],
    PALETTE["dark"],
]

# Standard style dictionary (mirrors the reference framework)
THESIS_RC = {
    # Fonts
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Aptos", "Segoe UI", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size":            FONT_SIZE_LABEL,
    "mathtext.fontset":     "cm",

    # Axes
    "axes.titlesize":       FONT_SIZE_TITLE,
    "axes.labelsize":       FONT_SIZE_LABEL,
    "axes.labelcolor":      APPLE["ink"],
    "axes.titleweight":     "normal",
    "axes.titlepad":        16,
    "axes.labelpad":        8,
    "axes.linewidth":       0.8,
    "axes.edgecolor":       APPLE["ink"],
    "axes.facecolor":       "white",
    "axes.grid":            True,
    "axes.grid.which":      "major",
    "axes.axisbelow":       True,
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # Grid
    "grid.color":           APPLE["grid"],
    "grid.linewidth":       0.4,
    "grid.alpha":           0.7,
    "grid.linestyle":       "-",

    # Ticks
    "xtick.labelsize":      FONT_SIZE_TICK,
    "ytick.labelsize":      FONT_SIZE_TICK,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.pad":      4,
    "ytick.major.pad":      4,

    # Legend
    "legend.fontsize":      FONT_SIZE_LEGEND,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#D1D1D6",
    "legend.fancybox":      True,
    "legend.borderpad":     0.5,
    "legend.handlelength":  1.8,
    "legend.handletextpad": 0.5,

    # Lines & Markers
    "lines.linewidth":      1.5,
    "lines.markersize":     5,

    # Figure
    "figure.facecolor":     "white",
    "figure.dpi":           DPI,
    "figure.constrained_layout.use": True,

    # Saving
    # NOTE: bbox_inches is intentionally NOT set to "tight": with
    # constrained_layout enabled, the tight-bbox pass re-runs the layout
    # solver at a degenerate size, emitting "constrained_layout not applied
    # because axes sizes collapsed to zero" and inflating the exported
    # canvas beyond the designed figsize (e.g. fig1 8.03" vs 6.5").
    # constrained_layout already fits every artist inside the canvas, so
    # exports are the exact designed dimensions (verified empirically).
    "savefig.dpi":          DPI,
    "savefig.pad_inches":   0.18,
    "savefig.facecolor":    "white",
    "savefig.transparent":  False,

    # PDF / PostScript font embedding
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

THESIS_RC_SERIF = dict(THESIS_RC, **{
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
})


@contextlib.contextmanager
def apply_thesis_style(serif: bool = False):
    """Context manager that applies the publication style for the duration."""
    rc = THESIS_RC_SERIF if serif else THESIS_RC
    with mpl.rc_context(rc):
        yield


def create_figure(
    width: str = "single",
    aspect: float = 0.618,
    nrows: int = 1,
    ncols: int = 1,
    height_override: Optional[float] = None,
    squeeze: bool = True,
    **kwargs,
) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, np.ndarray]]:
    """Create a figure with paper-appropriate dimensions.

    Parameters
    ----------
    width : {"single", "double", "full"}
        Column width preset (4.5", 6.5", 13.2").
    aspect : float
        Height/width ratio (default: golden ratio 0.618).
    nrows, ncols : int
        Subplot grid dimensions.
    height_override : float, optional
        Explicit height in inches (overrides aspect).
    squeeze : bool
        Whether to squeeze singleton dimensions.
    **kwargs
        Additional arguments passed to plt.subplots().
    """
    if width == "full":
        w = FULL_WIDTH
    elif width == "double":
        w = TEXT_WIDTH
    else:
        w = COL_WIDTH
    h = height_override if height_override is not None else w * aspect

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(w, h),
        squeeze=squeeze,
        constrained_layout=True,
        **kwargs
    )
    return fig, axes


def save_figure(
    fig: plt.Figure,
    path: Union[Path, str],
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
    dpi: int = DPI,
    bbox_inches: Optional[str] = None,
    **kwargs
) -> List[Path]:
    """Save a figure to the target formats and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : Path
        Base output path without extension (e.g. outputs/figures/my_figure).
    formats : sequence of str
        Target file formats to export (default: pdf, png).
    close : bool
        If True, closes the figure to reclaim memory.
    dpi : int
        Raster export resolution (default 300).
    """
    path = Path(path)
    written: List[Path] = []
    for fmt in formats:
        out_path = path.parent / f"{path.name}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if bbox_inches is not None:
            fig.savefig(str(out_path), format=fmt, dpi=dpi, bbox_inches="tight", **kwargs)
        else:
            fig.savefig(str(out_path), format=fmt, dpi=dpi, **kwargs)
        written.append(out_path)
        logger.debug("Saved figure: %s", out_path)

    if close:
        plt.close(fig)
    return written
