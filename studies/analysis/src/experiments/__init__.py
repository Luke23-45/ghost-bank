"""Per-experiment modules with bespoke figures + family-level cross-experiment modules."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Dict, List

from src.common import constants as C
from src.common.data import RunResult

# ── Per-experiment modules (one key → one module) ───────────────────
_EXPERIMENT_MODULES = {
    "icarl":           "src.experiments.b1_icarl",
    "static_bank":     "src.experiments.b2_static_bank",
    "uniform_herding": "src.experiments.b3_uniform_herding",
    "a1_no_kd":        "src.experiments.a1_no_kd",
    "a2_head_eval":    "src.experiments.a2_head_eval",
    "a3_linear_head":  "src.experiments.a3_linear_head",
    "a4_random_bank":  "src.experiments.a4_random_bank",
    "s1_budget500":    "src.experiments.s1_budget500",
    "s2_budget4000":   "src.experiments.s2_budget4000",
    "s3_retr32":       "src.experiments.s3_retr32",
    "s4_retr128":      "src.experiments.s4_retr128",
}

# ── Family-level modules (cross-experiment comparison figures) ───────
_FAMILY_MODULES = {
    "baselines":  "src.experiments.baselines",
    "component":  "src.experiments.component_ablations",
    "sensitivity": "src.experiments.sensitivity",
    "cross_cutting": "src.experiments.cross_cutting",
}

Generator = Callable[[Dict[str, RunResult], Path], List[Path]]


def _load(modpath: str):
    return importlib.import_module(modpath)


def experiment_module(key: str):
    """Return the per-experiment module for *key*."""
    return _load(_EXPERIMENT_MODULES[key])


def family_module(family: str):
    """Return the family-level module for *family*."""
    return _load(_FAMILY_MODULES[family])


def figures_for(key: str) -> Generator:
    """Return the generate_figures callable for a single experiment key."""
    mod = experiment_module(key)
    return mod.generate_figures


def family_figures_for(family: str) -> Generator:
    """Return the generate_figures callable for a family-level module."""
    mod = family_module(family)
    return mod.generate_figures


def all_experiment_keys() -> List[str]:
    return list(_EXPERIMENT_MODULES.keys())


def all_family_keys() -> List[str]:
    return list(_FAMILY_MODULES.keys())


def family_keys(family: str) -> List[str]:
    """Return the experiment keys belonging to a family (for loading runs)."""
    return getattr(C, {
        "baselines": "BASELINE_KEYS",
        "component": "COMPONENT_KEYS",
        "sensitivity": "SENSITIVITY_KEYS",
        "cross_cutting": "MASTER_ORDER",
    }[family])


# ── Lazy-loaded FAMILIES dict (backward compat for generate_figures/tables) ──
class _LazyFamilies(dict):
    """Dict that lazily imports family modules on first access."""
    _loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            for fam, modpath in _FAMILY_MODULES.items():
                super().__setitem__(fam, _load(modpath))
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def keys(self):
        self._ensure_loaded()
        return super().keys()


FAMILIES: Dict[str, object] = _LazyFamilies()
