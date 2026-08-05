"""LaTeX table generation utilities (booktabs).

Mirrors the referenced framework's latex helpers, extended with
mean +- std cell formatting and per-seed delta tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


def format_value(
    value: Union[float, int, str, None],
    *,
    precision: int = 4,
    use_math: bool = False,
) -> str:
    """Format a single numeric value for LaTeX."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        s = "—"
    elif isinstance(value, int):
        s = f"{value}"
    elif isinstance(value, str):
        return value
    else:
        s = f"{{:.{precision}f}}".format(value)
    return f"${s}$" if use_math else s


def format_mean_std(
    mean: float,
    std: Optional[float] = None,
    *,
    precision: int = 4,
    bold: bool = False,
    math: bool = True,
) -> str:
    """Format ``mean +- std`` (population std) the way the report prints it."""
    if std is None:
        s = f"{mean:.{precision}f}"
    else:
        s = f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"
    if math and std is None:
        s = f"${s}$"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    if "\\" in text or "$" in text:
        return text
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def booktabs_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str,
    label: str,
    col_spec: Optional[str] = None,
    alignment: Optional[str] = None,
) -> str:
    """Build a complete booktabs table environment (no document wrapper)."""
    n = len(headers)
    col_spec = col_spec or ("l" + "r" * (n - 1))
    header = " & ".join(f"\\textbf{{{latex_escape(str(h))}}}" for h in headers)
    body_lines = []
    for row in rows:
        cells = " & ".join(latex_escape(str(c)) for c in row)
        body_lines.append(f"    {cells} \\\\")
    midrule = "    \\midrule"
    if alignment:
        midrule = f"    \\cmidrule(lr){{{alignment}}}"
    tex = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{latex_escape(caption)}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"    {header} \\\\\n"
        f"{midrule}\n"
        f"{'\n'.join(body_lines)}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    return tex


def save_latex_table(
    tex_str: str,
    path: Union[str, Path],
    *,
    standalone_header: bool = True,
) -> Path:
    """Save a LaTeX table to a .tex file (optionally wrapped in a compilable doc)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = tex_str
    if standalone_header:
        preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{margin=1in}\n"
            "\\begin{document}\n\n"
        )
        postamble = "\n\n\\end{document}\n"
        content = preamble + content + postamble
    path.write_text(content, encoding="utf-8")
    logger.debug("Saved LaTeX table: %s", path)
    return path.resolve()


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: Optional[str] = None,
    align_right: bool = True,
) -> str:
    """Build a GitHub-style markdown table."""
    lines: List[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    sep = ["---"] + (["---:"] if align_right else ["---"]) * (len(headers) - 1)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def save_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    path: Union[str, Path],
    *,
    title: Optional[str] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(headers, rows, title=title), encoding="utf-8")
    return path.resolve()
