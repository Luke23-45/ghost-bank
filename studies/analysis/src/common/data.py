"""Run artifact loading for the Ghost Bank CIL analysis.

Every persisted run has the same layout::

    <run_root>/<timestamp>/
        results/final_results.json          aggregated + per-seed metrics
        metrics/aggregated_accuracy_matrix.csv    10x10 lower-triangular (mean over seeds)
        metrics/aggregated_accuracy_matrix_std.csv 10x10 (std over seeds)
        metrics/aggregated_metrics.csv      wide mean/std CSV
        metrics/seed_<seed>_accuracy_matrix.csv
        configs/resolved_config.yaml        resolved experiment config
        run_meta.json                       hardware/git metadata

The loader auto-discovers the newest timestamped run inside each pattern
root, so newly added runs are picked up without code changes.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.common import constants as C

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """All data of one persisted run, resolved for analysis."""

    key: str
    root: Path
    config: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    final: Dict[str, Any] = field(default_factory=dict)          # final_results.json
    agg_metrics: Dict[str, Any] = field(default_factory=dict)    # aggregated_metrics.csv (wide)
    accuracy_matrix: Optional[np.ndarray] = None                 # mean over seeds, nan lower-left
    accuracy_matrix_std: Optional[np.ndarray] = None
    seed_matrices: Dict[int, np.ndarray] = field(default_factory=dict)

    # ── Convenience accessors ────────────────────────────────────────
    @property
    def avg_acc(self) -> float:
        return float(self.final["aggregated"]["test/avg_acc_mean"])

    @property
    def avg_acc_std(self) -> float:
        return float(self.final["aggregated"]["test/avg_acc_std"])

    @property
    def forgetting(self) -> float:
        return float(self.final["aggregated"]["test/forgetting_mean"])

    @property
    def forgetting_std(self) -> float:
        return float(self.final["aggregated"]["test/forgetting_std"])

    @property
    def bwt(self) -> float:
        return float(self.final["aggregated"]["test/backward_transfer_mean"])

    @property
    def wall_time_s(self) -> float:
        return float(self.final["aggregated"]["wall_time_s_mean"])

    @property
    def method(self) -> str:
        return str(self.final.get("method") or self.meta.get("method") or self.key)

    def final_task_accs(self, use_percent: bool = True) -> np.ndarray:
        """Final-state per-task accuracies (mean over seeds), tasks 0..9."""
        agg = self.final["aggregated"]
        vals = [agg[f"test/task_{t}_final_acc_mean"] for t in range(C.NUM_TASKS)]
        arr = np.asarray(vals, dtype=float)
        return arr * 100.0 if use_percent else arr

    def final_task_stds(self, use_percent: bool = True) -> np.ndarray:
        agg = self.final["aggregated"]
        vals = [agg[f"test/task_{t}_final_acc_std"] for t in range(C.NUM_TASKS)]
        arr = np.asarray(vals, dtype=float)
        return arr * 100.0 if use_percent else arr

    def per_seed_avg_accs(self) -> Dict[int, float]:
        return {
            int(s["seed"]): float(s["test/avg_acc"])
            for s in self.final["per_seed_metrics"]
        }

    def per_seed_forgetting(self) -> Dict[int, float]:
        return {
            int(s["seed"]): float(s["test/forgetting"])
            for s in self.final["per_seed_metrics"]
        }

    def per_seed_final_task_accs(self, seed: int, use_percent: bool = True) -> np.ndarray:
        """Per-task final accuracies for one seed."""
        for s in self.final["per_seed_metrics"]:
            if int(s["seed"]) == seed:
                vals = [s[f"test/task_{t}_final_acc"] for t in range(C.NUM_TASKS)]
                arr = np.asarray(vals, dtype=float)
                return arr * 100.0 if use_percent else arr
        raise KeyError(f"seed {seed} not found in {self.key}")

    def memory_budget(self) -> Optional[int]:
        cfg = self.config.get("data", {})
        return cfg.get("memory_total")

    def retrieval_budget(self) -> Optional[int]:
        return self.config.get("method", {}).get("retrieval_budget")

    # ── Derived quantities from the evolution matrix ─────────────────
    @property
    def matrix(self) -> np.ndarray:
        """Evolution matrix in percent; raises if the artifact is missing."""
        if self.accuracy_matrix is None:
            raise FileNotFoundError(f"{self.key}: aggregated_accuracy_matrix.csv missing")
        return self.accuracy_matrix * 100.0

    def intro_accs(self, use_percent: bool = True) -> np.ndarray:
        """Accuracy of each task at the moment it was introduced (matrix diagonal)."""
        m = self.accuracy_matrix
        arr = np.asarray([m[t, t] for t in range(C.NUM_TASKS)], dtype=float)
        return arr * 100.0 if use_percent else arr

    def per_task_forgetting(self, use_percent: bool = True) -> np.ndarray:
        """Per-task forgetting (pp): accuracy at introduction minus final accuracy."""
        m = self.accuracy_matrix
        arr = np.asarray([m[t, t] - m[-1, t] for t in range(C.NUM_TASKS)], dtype=float)
        return arr * 100.0 if use_percent else arr

    def per_task_delta_vs(self, ref: "RunResult", use_percent: bool = True) -> np.ndarray:
        """Per-task final-accuracy delta vs a reference run (this minus ref)."""
        arr = self.final_task_accs(use_percent=False) - ref.final_task_accs(use_percent=False)
        return arr * 100.0 if use_percent else arr

    def forgetting_accumulation(self, use_percent: bool = True) -> np.ndarray:
        """Cumulative mean forgetting (pp) over evaluation times 0..9.

        At evaluation time t, forgetting of task j (j <= t) is its introduction
        accuracy minus its accuracy at time t; the curve averages over the tasks
        seen so far. This is the 'how fast does the model erode' trace.
        """
        m = self.accuracy_matrix
        diag = np.asarray([m[t, t] for t in range(C.NUM_TASKS)], dtype=float)
        out = np.full(C.NUM_TASKS, np.nan, dtype=float)
        for t in range(C.NUM_TASKS):
            forget = diag[: t + 1] - m[t, : t + 1]
            out[t] = float(np.mean(forget))
        return out * 100.0 if use_percent else out

    def trajectories(self) -> List[np.ndarray]:
        """Per-task accuracy traces: trace[j][i] = accuracy of task j at eval time i.

        trace[j] is defined for i in [j, 9], np.nan elsewhere. The first valid
        point is the introduction accuracy (matrix diagonal).
        """
        m = self.accuracy_matrix
        traces: List[np.ndarray] = []
        for j in range(C.NUM_TASKS):
            trace = np.full(C.NUM_TASKS, np.nan, dtype=float)
            trace[j:] = m[j:, j] * 100.0
            traces.append(trace)
        return traces


def _discover_latest_run(pattern_root: Path) -> Path:
    """Return the newest timestamped run directory (YYYYMMDD_HHMMSS)."""
    if not pattern_root.exists():
        raise FileNotFoundError(f"run pattern root missing: {pattern_root}")
    run_dirs = sorted(
        (p for p in pattern_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    if not run_dirs:
        raise FileNotFoundError(f"no timestamped runs under {pattern_root}")
    return run_dirs[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_wide_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    return {k: _num(v) for k, v in row.items()} if row else {}


def _num(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def _load_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [[_num(x) for x in line.split(",")] for line in f if line.strip()]
    n = max(len(r) for r in rows)
    arr = np.full((n, n), np.nan, dtype=float)
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row
    return arr


def load_run(key: str, *, pattern_root: Optional[Path] = None) -> RunResult:
    """Load the (latest) persisted run for an experiment key."""
    pattern_root = pattern_root or C.run_root(key)
    run_dir = _discover_latest_run(pattern_root)

    result = RunResult(key=key, root=run_dir)
    result.meta = _load_json(run_dir / "run_meta.json")
    cfg_path = run_dir / "configs" / "resolved_config.yaml"
    if cfg_path.exists():
        result.config = _load_yaml(cfg_path)

    final_path = run_dir / "results" / "final_results.json"
    if final_path.exists():
        result.final = _load_json(final_path)

    metrics_dir = run_dir / "metrics"
    result.agg_metrics = _load_wide_csv(metrics_dir / "aggregated_metrics.csv")
    result.accuracy_matrix = _load_matrix(metrics_dir / "aggregated_accuracy_matrix.csv")
    std_path = metrics_dir / "aggregated_accuracy_matrix_std.csv"
    if std_path.exists():
        result.accuracy_matrix_std = _load_matrix(std_path)
    for seed in C.SEEDS:
        m_path = metrics_dir / f"seed_{seed}_accuracy_matrix.csv"
        if m_path.exists():
            result.seed_matrices[seed] = _load_matrix(m_path)

    return result


def load_all_runs(keys: Optional[List[str]] = None) -> Dict[str, RunResult]:
    """Load all registered runs, keyed by experiment key."""
    keys = keys or C.MASTER_ORDER
    runs: Dict[str, RunResult] = {}
    for key in keys:
        try:
            runs[key] = load_run(key)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", key, exc)
    return runs


def matched_deltas(runs: Dict[str, RunResult], ref_key: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Per-seed matched deltas of each run vs the reference run.

    Returns {key: {"avg_acc": {seed: delta}, "forgetting": {seed: delta}}}.
    """
    ref = runs[ref_key]
    ref_acc = ref.per_seed_avg_accs()
    ref_for = ref.per_seed_forgetting()
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for key, run in runs.items():
        if key == ref_key:
            continue
        out[key] = {
            "avg_acc": {
                s: run.per_seed_avg_accs()[s] - ref_acc[s] for s in ref_acc
            },
            "forgetting": {
                s: run.per_seed_forgetting()[s] - ref_for[s] for s in ref_for
            },
        }
    return out
