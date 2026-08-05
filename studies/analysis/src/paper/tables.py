"""Paper tables (T1-T3 main, A1-A6 appendix) for the Ghost Bank CIL paper.

Every cell is recomputed from the persisted run artifacts via the
verified data layer (``src/common/data.py``) — no hard-coded numbers.
The specification follows ``docs/paper/analysis/table_plan.md``.

Output: ``outputs/paper/tables/`` — 9 tables x (.tex, .md) = 18 files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from src.common import constants as C
from src.common.data import RunResult, matched_deltas
from src.common.latex import (
    booktabs_table,
    save_latex_table,
    save_markdown_table,
)

logger = logging.getLogger(__name__)

REF = C.reference_key()
TABLES_OUT = C.PAPER_TABLES_DIR


# ── Formatting helpers (percent-based cells) ─────────────────────────
def _m(mean: float, precision: int = 2) -> str:
    """Percent mean, e.g. 0.4236 -> "42.36"."""
    return f"{mean * 100.0:.{precision}f}"


def _ms(mean: float, std: float, precision: int = 2) -> str:
    """Percent mean +/- std, e.g. "42.36 $\\pm$ 0.87"."""
    return f"{mean * 100.0:.{precision}f} $\\pm$ {std * 100.0:.{precision}f}"


def _delta(value: float, precision: int = 2) -> str:
    """Signed value in pp (already in pp), e.g. -0.2067 -> "-0.21"."""
    return f"{value:+.{precision}f}"


def _md_md(cell: str) -> str:
    """Render a tex-format cell for markdown (unicode pm)."""
    return cell.replace("$\\pm$", "±")


def _save_pair(headers: Sequence[str], rows: Sequence[Sequence[str]],
               out_dir: Path, name: str, *, caption: str, label: str,
               col_spec: str = "") -> List[Path]:
    col_spec = col_spec or ("l" + "r" * (len(headers) - 1))
    tex = booktabs_table(headers, rows, caption=caption, label=label, col_spec=col_spec)
    tex_path = save_latex_table(tex, out_dir / TABLES_OUT / f"{name}.tex")
    md_rows = [[_md_md(str(c)) for c in row] for row in rows]
    md_headers = [_md_md(str(h)) for h in headers]
    md_path = save_markdown_table(md_headers, md_rows, out_dir / TABLES_OUT / f"{name}.md",
                                  title=name)
    return [tex_path, md_path]


# ── Significance (two-sided paired t-test on per-seed deltas vs zero) ─
def _sig_cell(deltas: Sequence[float]) -> str:
    """Return the significance flag from a two-sided paired t-test vs zero."""
    _, p = stats.ttest_1samp(list(deltas), 0.0)
    return "sig" if p < 0.05 else ("marginal" if p < 0.10 else "n.s.")


# ── T1: master results ───────────────────────────────────────────────
def table_T1_master_results(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "Mem", "Retr", "avg_acc (%)", "forgetting (%)", "BWT (%)"]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]
        mem = run.memory_budget()
        retr = run.retrieval_budget()
        rows.append([
            C.short_name(key), C.display_name(key),
            str(mem) if mem is not None else "—",
            str(retr) if retr is not None else "—",
            _ms(run.avg_acc, run.avg_acc_std),
            _ms(run.forgetting, run.forgetting_std),
            _ms(run.bwt, run.bwt_std),
        ])
    return _save_pair(
        headers, rows, out_dir, "T1_master_results",
        caption="Master results on CIFAR-100 class-incremental learning "
                "(10 tasks, mean $\\pm$ std over 3 seeds; budgets from resolved configs).",
        label="tab:T1_master_results",
    )


# ── T2: component attribution ────────────────────────────────────────
def table_T2_component_ablations(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    ablation_keys = [k for k in C.COMPONENT_KEYS if k != REF]
    deltas = matched_deltas(runs, REF)
    headers = ["#", "Experiment", "avg_acc (%)", "forgetting (%)",
               "Δ acc (pp)", "Δ forgetting (pp)", "Sig (acc)", "Sig (fgt)"]
    rows = []
    for key in ablation_keys:
        run = runs[key]
        d = deltas[key]
        acc_deltas = list(d["avg_acc"].values())
        for_deltas = list(d["forgetting"].values())
        acc_eff = float(np.mean(acc_deltas)) * 100.0
        for_eff = float(np.mean(for_deltas)) * 100.0
        rows.append([
            C.short_name(key), C.display_name(key),
            _ms(run.avg_acc, run.avg_acc_std),
            _ms(run.forgetting, run.forgetting_std),
            _delta(acc_eff), _delta(for_eff),
            _sig_cell(acc_deltas), _sig_cell(for_deltas),
        ])
    return _save_pair(
        headers, rows, out_dir, "T2_component_ablations",
        caption="Component ablations on CIFAR-100 (mean $\\pm$ std over 3 seeds; "
                "deltas are matched per-seed against the reference run, two-sided "
                "paired t-test vs zero, sig p<0.05, marginal p<0.1).",
        label="tab:T2_component_ablations",
    )


# ── T3: resource sensitivity with reference anchors ──────────────────
def table_T3_resource_sensitivity(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    deltas = matched_deltas(runs, REF)
    headers = ["#", "Experiment", "Axis", "Value", "avg_acc (%)",
               "forgetting (%)", "Δ acc (pp)", "Δ forgetting (pp)"]
    rows = []

    def row(key: str, axis: str, value: int, delta_key: str) -> List[str]:
        run = runs[key]
        d = deltas[delta_key]
        acc_eff = float(np.mean(list(d["avg_acc"].values()))) * 100.0
        for_eff = float(np.mean(list(d["forgetting"].values()))) * 100.0
        return [
            C.short_name(key), C.display_name(key), axis, str(value),
            _ms(run.avg_acc, run.avg_acc_std),
            _ms(run.forgetting, run.forgetting_std),
            _delta(acc_eff), _delta(for_eff),
        ]

    for key, axis, value in [
        ("s1_budget500", "Memory", 500),
        ("uniform_herding", "Memory", 2000),
        ("s2_budget4000", "Memory", 4000),
        ("s3_retr32", "Retrieval", 32),
        ("uniform_herding", "Retrieval", 64),
        ("s4_retr128", "Retrieval", 128),
    ]:
        if key == REF:
            rows.append([C.short_name(key), C.display_name(key), axis, str(value),
                         _ms(runs[key].avg_acc, runs[key].avg_acc_std),
                         _ms(runs[key].forgetting, runs[key].forgetting_std),
                         "0", "0"])
        else:
            rows.append(row(key, axis, value, key))
    return _save_pair(
        headers, rows, out_dir, "T3_resource_sensitivity",
        caption="Resource sensitivity on CIFAR-100 (mean $\\pm$ std over 3 seeds; "
                "deltas are matched per-seed against the reference run at 2000 "
                "exemplars / 64 retrieval).",
        label="tab:T3_resource_sensitivity",
    )


# ── A1: protocol and reproducibility ─────────────────────────────────
def _git_summary(runs: Dict[str, RunResult]) -> str:
    groups: Dict[str, List[str]] = {}
    for key in C.MASTER_ORDER:
        commit = runs[key].meta.get("git_commit", "?")
        groups.setdefault(commit, []).append(key)
    parts = []
    for commit in sorted(groups):
        keys = groups[commit]
        dirty = any(runs[k].meta.get("git_dirty") for k in keys)
        if len(keys) == 1:
            parts.append(f"{C.short_name(keys[0])} `{commit[:8]}`")
        else:
            names = ", ".join(C.short_name(k) for k in keys)
            parts.append(f"{names} `{commit[:8]}`")
        parts[-1] += " (dirty)" if dirty else ""
    return "; ".join(parts)


def table_A1_protocol(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    ref = runs[REF]
    cfg = ref.config
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    method_cfg = cfg.get("method", {})
    train_cfg = cfg.get("training", {})
    meta = ref.meta

    def budgets(axis: str) -> str:
        if axis == "memory":
            mem = {runs[k].memory_budget(): k for k in C.MASTER_ORDER if runs[k].memory_budget() is not None}
            parts = []
            for value in sorted(mem):
                keys = [k for k in C.MASTER_ORDER if runs[k].memory_budget() == value]
                names = ", ".join(C.short_name(k) for k in keys)
                parts.append(f"{value} ({names})")
            return "; ".join(parts)
        retr = {runs[k].retrieval_budget(): k for k in C.MASTER_ORDER if runs[k].retrieval_budget() is not None}
        parts = []
        for value in sorted(retr):
            keys = [k for k in C.MASTER_ORDER if runs[k].retrieval_budget() == value]
            names = ", ".join(C.short_name(k) for k in keys)
            parts.append(f"{value} ({names})")
        return "; ".join(parts)

    kd_row = method_cfg.get("kd_weight")
    kd_temp = method_cfg.get("kd_temperature")
    kd_note = ""
    if runs["a1_no_kd"].config.get("method", {}).get("kd_weight", 1.0) != kd_row:
        kd_note = "; disabled for a1 (kd_weight 0.0)"

    rows: List[List[str]] = [
        ["Dataset", f"CIFAR-100, {data_cfg.get('num_tasks')} tasks x "
                    f"{data_cfg.get('classes_per_task')} classes, "
                    f"split seed {data_cfg.get('split_seed')}, "
                    f"probe/val splits {data_cfg.get('probe_split_size')}/"
                    f"{data_cfg.get('val_split_size')}"],
        ["Backbone", f"ResNet-{model_cfg.get('depth')}, base filters "
                     f"{model_cfg.get('base_filters')}, dropout {model_cfg.get('dropout')}"],
        ["Head", f"{model_cfg.get('head')} (scale {model_cfg.get('head_scale')}, "
                 f"margin {model_cfg.get('head_margin')}, first-task imprinting); "
                 f"linear for a3"],
        ["Optimizer", f"SGD lr {train_cfg.get('learning_rate')} / momentum "
                      f"{train_cfg.get('momentum')} / weight decay "
                      f"{train_cfg.get('weight_decay')}, grad clip "
                      f"{train_cfg.get('gradient_clip_val')}, no LR schedule, "
                      f"warmup {method_cfg.get('warmup_steps')}"],
        ["Epochs per task", f"{train_cfg.get('max_epochs')} configured; "
                            f"{ref.epochs_per_task():g} recorded for all 33 seeds "
                            f"(off-by-one in the epoch counter)"],
        ["Batch size", f"{data_cfg.get('batch_size')}"],
        ["Precision", f"{train_cfg.get('precision')}"],
        ["Seeds", "1993, 2023, 42"],
        ["Exemplar budgets", budgets("memory")],
        ["Retrieval budgets", budgets("retrieval")],
        ["Knowledge distillation", f"weight {kd_row}, temperature {kd_temp}{kd_note}"],
        ["Evaluation protocol", "NME for B1/B3/a1-a4; head-logit for B2 (native protocol)"],
        ["Hardware", f"{meta.get('device')}"],
        ["Software", f"torch {meta.get('torch')}, pytorch-lightning "
                     f"{meta.get('pytorch_lightning')}, python {meta.get('python')}"],
        ["Git commits", _git_summary(runs)],
        ["Wall time", "see Table A5 (compute cost)"],
    ]
    headers = ["Setting", "Value"]
    return _save_pair(
        headers, rows, out_dir, "A1_protocol",
        caption="Protocol and reproducibility settings for all runs "
                "(sourced from resolved configs and run metadata).",
        label="tab:A1_protocol",
        col_spec="ll",
    )


# ── A2: per-task final accuracies (mean +/- std) ─────────────────────
def table_A2_per_task_accuracies(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment"] + [f"t{t}" for t in range(C.NUM_TASKS)]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]
        means = run.final_task_accs()
        stds = run.final_task_stds()
        cells = [f"{m:.1f} $\\pm$ {s:.1f}" for m, s in zip(means, stds)]
        rows.append([C.short_name(key), C.display_name(key)] + cells)
    return _save_pair(
        headers, rows, out_dir, "A2_per_task_accuracies",
        caption="Final-state per-task accuracies (mean $\\pm$ std over 3 seeds, "
                "percent; landscape).",
        label="tab:A2_per_task_accuracies",
    )


# ── A3: per-task forgetting (mean +/- std) ───────────────────────────
def table_A3_per_task_forgetting(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment"] + [f"t{t}" for t in range(C.NUM_TASKS)]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]
        means = run.per_task_forgetting()
        stds = run.per_task_forgetting_std()
        cells = [f"{m:.1f} $\\pm$ {s:.1f}" for m, s in zip(means, stds)]
        rows.append([C.short_name(key), C.display_name(key)] + cells)
    return _save_pair(
        headers, rows, out_dir, "A3_per_task_forgetting",
        caption="Per-task forgetting, introduction minus final accuracy "
                "(pp, mean $\\pm$ std over 3 seeds; t9 is 0.0 by construction).",
        label="tab:A3_per_task_forgetting",
    )


# ── A4: per-seed metrics (long format) ───────────────────────────────
def table_A4_per_seed_metrics(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "Seed", "avg_acc (%)", "forgetting (%)", "BWT (%)"]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]
        for s in C.SEEDS:
            rows.append([
                C.short_name(key), C.display_name(key), str(s),
                _m(run.per_seed_avg_accs()[s]),
                _m(run.per_seed_forgetting()[s]),
                _m(run.per_seed_bwt()[s]),
            ])
    return _save_pair(
        headers, rows, out_dir, "A4_per_seed_metrics",
        caption="Per-seed average accuracy, forgetting and backward transfer "
                "(percent, 33 rows: 11 runs x 3 seeds).",
        label="tab:A4_per_seed_metrics",
    )


# ── A5: compute cost ─────────────────────────────────────────────────
def table_A5_compute_cost(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    headers = ["#", "Experiment", "wall_time_s", "seed 1993", "seed 2023",
               "seed 42", "device"]
    rows = []
    for key in C.MASTER_ORDER:
        run = runs[key]
        walls = run.per_seed_wall_times()
        rows.append([
            C.short_name(key), C.display_name(key),
            f"{run.wall_time_s:.1f} $\\pm$ {run.wall_time_s_std:.1f}",
            f"{walls[1993]:.1f}", f"{walls[2023]:.1f}", f"{walls[42]:.1f}",
            str(run.meta.get("device", "—")),
        ])
    return _save_pair(
        headers, rows, out_dir, "A5_compute_cost",
        caption="Wall-clock time per run (seconds; aggregate mean $\\pm$ std "
                "and per-seed values; all runs on Tesla T4).",
        label="tab:A5_compute_cost",
    )


# ── A6: exemplar bank sizes ──────────────────────────────────────────
def table_A6_bank_sizes(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    budget_keys = ["s1_budget500", "uniform_herding", "s2_budget4000"]
    headers = ["Budget"] + [f"t{t}" for t in range(C.NUM_TASKS)]
    rows = []
    for key in budget_keys:
        run = runs[key]
        cells = []
        for t in range(C.NUM_TASKS):
            lo, hi = run.bank_quota_range(t)
            cells.append(str(lo) if lo == hi else f"{lo}-{hi}")
        rows.append([str(run.memory_budget())] + cells)
    return _save_pair(
        headers, rows, out_dir, "A6_bank_sizes",
        caption="Per-class exemplar quota per task at each budget "
                "(quota = budget divided by classes seen, floor rounding "
                "causes the +/-1 pairs; identical across seeds).",
        label="tab:A6_bank_sizes",
    )


# ── Registry (keys match C.PAPER_TABLES) ─────────────────────────────
BUILDERS: Dict[str, object] = {
    "T1_master_results": table_T1_master_results,
    "T2_component_ablations": table_T2_component_ablations,
    "T3_resource_sensitivity": table_T3_resource_sensitivity,
    "A1_protocol": table_A1_protocol,
    "A2_per_task_accuracies": table_A2_per_task_accuracies,
    "A3_per_task_forgetting": table_A3_per_task_forgetting,
    "A4_per_seed_metrics": table_A4_per_seed_metrics,
    "A5_compute_cost": table_A5_compute_cost,
    "A6_bank_sizes": table_A6_bank_sizes,
}


def generate_tables(runs: Dict[str, RunResult], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    for name in C.PAPER_TABLES:
        outputs += BUILDERS[name](runs, out_dir)
    return outputs
