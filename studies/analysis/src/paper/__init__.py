"""Paper pipeline package: the approved manuscript figure/table set.

- ``main_figures``      — Fig 1-5 (outputs/paper/main/figures/)
- ``appendix_figures``  — Fig A1-A4 (outputs/paper/appendix/figures/)
- ``tables``            — T1-T3, A1-A6 (outputs/paper/tables/, .tex/.md)

This package replaces the former 38-figure library pipeline; only the
approved artifacts are generated (see ``docs/paper/analysis/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from src.common import constants as C
from src.common.data import RunResult

Generator = Callable[[Dict[str, RunResult], Path], List[Path]]

from src.paper import appendix_figures, main_figures, tables  # noqa: E402


def main_figure_builder(name: str) -> Generator:
    return main_figures.BUILDERS[name]


def appendix_figure_builder(name: str) -> Generator:
    return appendix_figures.BUILDERS[name]


def table_builder(name: str) -> Generator:
    return tables.BUILDERS[name]
