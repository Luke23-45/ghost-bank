"""Canonical experiment registry for the Ghost Bank CIL paper analysis.

Single source of truth for:

- Experiment keys (B1-B3 baselines, a1-a4 component ablations, s1-s4 sensitivity).
- On-disk run locations (relative to the repository's ``experiment_output`` root).
- Display names, colors, markers and group ordering for every figure/table.

New runs (e.g. an extended memory curve) only need a new entry here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.common.style import SERIES_COLORS

# The analysis package lives at studies/analysis/src/common/constants.py
REPO_ROOT = Path(__file__).resolve().parents[4]             # repo root
EXPERIMENT_OUTPUT_ROOT = REPO_ROOT / "experiment_output"

SEEDS = (1993, 2023, 42)
NUM_TASKS = 10

# ── Display metadata ─────────────────────────────────────────────────
DISPLAY_NAMES: Dict[str, str] = {
    "icarl":               "iCaRL",
    "static_bank":         "Static bank",
    "uniform_herding":     "Uniform herding (Reference)",
    "a1_no_kd":            "Ref. without KD",
    "a2_head_eval":        "Ref. head-logit eval",
    "a3_linear_head":      "Ref. linear head",
    "a4_random_bank":      "Ref. random selection",
    "s1_budget500":        "Memory 500",
    "s2_budget4000":       "Memory 4000",
    "s3_retr32":           "Retrieval 32",
    "s4_retr128":          "Retrieval 128",
}

SHORT_NAMES: Dict[str, str] = {
    "icarl":               "B1",
    "static_bank":         "B2",
    "uniform_herding":     "B3",
    "a1_no_kd":            "a1",
    "a2_head_eval":        "a2",
    "a3_linear_head":      "a3",
    "a4_random_bank":      "a4",
    "s1_budget500":        "s1",
    "s2_budget4000":       "s2",
    "s3_retr32":           "s3",
    "s4_retr128":          "s4",
}

# ── Run location patterns (relative to EXPERIMENT_OUTPUT_ROOT) ───────
# The loader auto-discovers the newest timestamped run directory inside.
RUN_PATTERNS: Dict[str, Tuple[Path, str]] = {
    "icarl":           (Path("final_baseline_run/output/cifar100/icarl"), "icarl"),
    "static_bank":     (Path("final_baseline_run/output/cifar100/static_bank"), "static_bank"),
    "uniform_herding": (Path("final_baseline_run/output/cifar100/uniform_herding"), "uniform_herding"),
    "a1_no_kd":        (Path("abalations/component/a1_no_kd/cifar100/uniform_herding"), "uniform_herding"),
    "a2_head_eval":    (Path("abalations/component/a2_head_eval/cifar100/uniform_herding"), "uniform_herding"),
    "a3_linear_head":  (Path("abalations/component/a3_linear_head/cifar100/uniform_herding"), "uniform_herding"),
    "a4_random_bank":  (Path("abalations/component/a4_random_bank/cifar100/uniform_herding"), "uniform_herding"),
    "s1_budget500":    (Path("abalations/sensitivity/s1_budget500/cifar100/uniform_herding"), "uniform_herding"),
    "s2_budget4000":   (Path("abalations/sensitivity/s2_budget4000/cifar100/uniform_herding"), "uniform_herding"),
    "s3_retr32":       (Path("abalations/sensitivity/s3_retr32/cifar100/uniform_herding"), "uniform_herding"),
    "s4_retr128":      (Path("abalations/sensitivity/s4_retr128/cifar100/uniform_herding"), "uniform_herding"),
}

# ── Group membership ─────────────────────────────────────────────────
BASELINE_KEYS: List[str] = ["icarl", "static_bank", "uniform_herding"]
COMPONENT_KEYS: List[str] = [
    "uniform_herding", "a1_no_kd", "a2_head_eval", "a3_linear_head", "a4_random_bank",
]
SENSITIVITY_MEMORY_KEYS: List[str] = ["s1_budget500", "uniform_herding", "s2_budget4000"]
SENSITIVITY_RETR_KEYS: List[str] = ["s3_retr32", "uniform_herding", "s4_retr128"]
SENSITIVITY_KEYS: List[str] = [
    "s1_budget500", "s2_budget4000", "s3_retr32", "s4_retr128",
]
ALL_KEYS: List[str] = (
    BASELINE_KEYS + ["a1_no_kd", "a2_head_eval", "a3_linear_head", "a4_random_bank"]
    + SENSITIVITY_KEYS
)

# Key ordering used everywhere (paper ordering, from the master table)
MASTER_ORDER: List[str] = [
    "icarl",
    "static_bank",
    "uniform_herding",
    "a1_no_kd",
    "a2_head_eval",
    "a3_linear_head",
    "a4_random_bank",
    "s1_budget500",
    "s2_budget4000",
    "s3_retr32",
    "s4_retr128",
]

# ── Style assignment (stable, colorblind-safe) ───────────────────────
COLORS: Dict[str, str] = {
    "icarl":           SERIES_COLORS[3],   # rose
    "static_bank":     "#007AFF",          # Apple blue (darker than sky for contrast)
    "uniform_herding": SERIES_COLORS[0],   # indigo (reference; always emphasized)
    "a1_no_kd":        "#E8912D",          # warm amber (replaces olive for contrast)
    "a2_head_eval":    SERIES_COLORS[5],   # wine
    "a3_linear_head":  SERIES_COLORS[7],   # purple
    "a4_random_bank":  SERIES_COLORS[8],   # pink
    "s1_budget500":    SERIES_COLORS[9],   # sand
    "s2_budget4000":   SERIES_COLORS[1],   # teal
    "s3_retr32":       SERIES_COLORS[2],   # green
    "s4_retr128":      SERIES_COLORS[10],  # dark
}

MARKERS: Dict[str, str] = {
    "icarl": "o", "static_bank": "s", "uniform_herding": "D",
    "a1_no_kd": "p", "a2_head_eval": "P", "a3_linear_head": "*", "a4_random_bank": "X",
    "s1_budget500": "P", "s2_budget4000": "*", "s3_retr32": "X", "s4_retr128": "h",
}

# Headline pairs for the resource story (x value -> experiment key)
MEMORY_X: Dict[float, str] = {500: "s1_budget500", 2000: "uniform_herding", 4000: "s2_budget4000"}
RETRIEVAL_X: Dict[float, str] = {32: "s3_retr32", 64: "uniform_herding", 128: "s4_retr128"}


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def short_name(key: str) -> str:
    return SHORT_NAMES.get(key, key)


def color_for(key: str) -> str:
    return COLORS[key]


def marker_for(key: str) -> str:
    return MARKERS.get(key, "o")


def run_root(key: str) -> Path:
    """Absolute root of the run directory pattern for an experiment key."""
    rel, _ = RUN_PATTERNS[key]
    return EXPERIMENT_OUTPUT_ROOT / rel


def reference_key() -> str:
    return "uniform_herding"


# ── Paper output layout ──────────────────────────────────────────────
# The manuscript pipeline writes only the approved figure/table set:
#   outputs/paper/main/figures/      (Fig 1-5)
#   outputs/paper/appendix/figures/  (Fig A1-A4)
#   outputs/paper/tables/            (T1-T3, A1-A6; .tex/.md pairs)
PAPER_MAIN_FIGURES_DIR: Path = Path("paper/main/figures")
PAPER_APPENDIX_FIGURES_DIR: Path = Path("paper/appendix/figures")
PAPER_TABLES_DIR: Path = Path("paper/tables")

# ── Paper figure registry (single source of truth for the manuscript) ─
PAPER_MAIN_FIGURES: List[str] = [
    "fig1_per_task_accuracy",
    "fig2_component_attribution",
    "fig3_resource_sensitivity",
    "fig4_acc_forgetting_scatter",
    "fig5_forgetting_by_age",
]

PAPER_APPENDIX_FIGURES: List[str] = [
    "figA1_forgetting_heatmap",
    "figA2_evolution_reference",
    "figA3_evolution_icarl",
    "figA4_stability_slopes",
]

PAPER_TABLES: List[str] = [
    "T1_master_results",
    "T2_component_ablations",
    "T3_resource_sensitivity",
    "A1_protocol",
    "A2_per_task_accuracies",
    "A3_per_task_forgetting",
    "A4_per_seed_metrics",
    "A5_compute_cost",
    "A6_bank_sizes",
]
