"""Independent post-generation verification for the paper artifact set.

Re-derives every number the paper tables/figures present directly from the
persisted run artifacts (``experiment_output/**``) and diffs them against
the triple-checked values documented in ``docs/paper/analysis/analysis_plan.md``
(Sections 2.1-2.4) and ``docs/paper/analysis/table_plan.md``. Any mismatch
fails the run with exit code 1.

This script intentionally re-computes values rather than reading the
generated files, so a wrong number cannot survive by being written twice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# Bootstrap: ensure studies/analysis (the package parent) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import stats

from src.common import constants as C
from src.common.config import get_config, get_output_root
from src.common.data import RunResult, load_all_runs, matched_deltas
from src.paper import appendix_figures, main_figures, tables

FAILURES: List[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def close_arr(actual: Sequence[float], expected: Sequence[float], tol: float) -> bool:
    return len(actual) == len(expected) and all(close(a, e, tol) for a, e in zip(actual, expected))


# ── 2.1 master results (verified against the report) ─────────────────
MASTER: Dict[str, Tuple[float, float, float, float, float, float]] = {
    # key: (avg_acc, avg_acc_std, forgetting, forgetting_std, bwt, bwt_std)
    "baseline":        (0.0791, 0.0011, 0.8021, 0.0090, -0.8021, 0.0090),
    "icarl":           (0.4233, 0.0120, 0.2487, 0.0111, -0.2487, 0.0111),
    "static_bank":     (0.2860, 0.0135, 0.5586, 0.0142, -0.5586, 0.0142),
    "uniform_herding": (0.4400, 0.0051, 0.1722, 0.0043, -0.1691, 0.0041),
    "a1_no_kd":        (0.4252, 0.0061, 0.2888, 0.0019, -0.2888, 0.0019),
    "a2_head_eval":    (0.3354, 0.0039, 0.4758, 0.0084, -0.4758, 0.0084),
    "a3_linear_head":  (0.4322, 0.0036, 0.1734, 0.0068, -0.1733, 0.0067),
    "a4_random_bank":  (0.4161, 0.0053, 0.1838, 0.0047, -0.1828, 0.0061),
    "s1_budget500":    (0.3395, 0.0151, 0.3048, 0.0054, -0.3048, 0.0054),
    "s2_budget4000":   (0.4732, 0.0024, 0.1331, 0.0059, -0.1274, 0.0065),
    "s3_retr32":       (0.4247, 0.0063, 0.1721, 0.0047, -0.1687, 0.0042),
    "s4_retr128":      (0.4416, 0.0059, 0.1781, 0.0038, -0.1749, 0.0037),
}

# ── 2.3 per-task final accuracies (%) ────────────────────────────────
PER_TASK_ACC: Dict[str, List[float]] = {
    "baseline":        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 79.1],
    "icarl":           [45.0, 38.6, 39.2, 40.4, 37.1, 41.9, 34.0, 46.0, 50.3, 50.8],
    "static_bank":     [29.5, 20.7, 22.5, 21.9, 19.6, 23.6, 15.8, 25.7, 31.5, 75.2],
    "uniform_herding": [51.2, 42.5, 41.8, 46.0, 39.5, 44.6, 35.2, 44.6, 46.3, 48.4],
    "a1_no_kd":        [44.0, 40.1, 39.4, 42.2, 38.0, 42.7, 35.7, 42.3, 43.1, 57.5],
    "a2_head_eval":    [27.0, 20.5, 22.5, 25.2, 23.5, 27.3, 24.2, 36.8, 52.1, 76.4],
    "a3_linear_head":  [48.6, 40.7, 40.6, 46.2, 40.3, 45.9, 35.0, 47.1, 45.6, 42.0],
    "a4_random_bank":  [47.4, 40.3, 40.5, 44.9, 37.6, 41.1, 33.0, 43.0, 44.7, 43.6],
    "s1_budget500":    [37.9, 26.8, 26.9, 32.0, 27.8, 32.5, 23.4, 37.0, 44.2, 51.2],
    "s2_budget4000":   [56.9, 47.3, 48.5, 50.1, 43.8, 47.6, 38.7, 48.1, 45.7, 46.4],
    "s3_retr32":       [51.0, 43.1, 42.1, 43.6, 39.2, 40.7, 33.8, 42.4, 42.2, 46.6],
    "s4_retr128":      [52.7, 43.3, 42.9, 47.3, 40.5, 43.4, 35.1, 42.7, 46.2, 47.6],
}

# ── 2.4 per-task forgetting (intro - final, pp) ──────────────────────
PER_TASK_FORGET: Dict[str, List[float]] = {
    "baseline":        [82.5, 79.2, 78.8, 80.7, 77.3, 83.3, 76.3, 82.1, 81.8, 0.0],
    "icarl":           [39.4, 34.0, 31.9, 27.6, 23.2, 22.7, 19.4, 15.9, 9.7, 0.0],
    "static_bank":     [53.2, 55.8, 58.5, 57.6, 55.2, 59.0, 57.7, 54.9, 50.8, 0.0],
    "uniform_herding": [30.1, 14.0, 21.6, 15.7, 17.5, 16.9, 15.8, 12.5, 8.1, 0.0],
    "a1_no_kd":        [37.2, 34.2, 35.8, 30.0, 28.1, 27.0, 24.7, 22.6, 20.3, 0.0],
    "a2_head_eval":    [52.9, 47.7, 55.1, 50.6, 50.6, 53.2, 48.7, 42.1, 27.2, 0.0],
    "a3_linear_head":  [36.2, 28.2, 27.6, 16.8, 13.3, 10.6, 11.0, 7.4, 5.0, 0.0],
    "a4_random_bank":  [33.8, 16.7, 21.2, 16.0, 19.4, 18.3, 16.1, 13.9, 9.1, 0.0],
    "s1_budget500":    [43.3, 30.3, 37.8, 32.0, 31.8, 29.3, 30.8, 25.0, 13.9, 0.0],
    "s2_budget4000":   [24.3, 8.6, 15.4, 12.9, 11.8, 12.1, 12.6, 9.1, 7.9, 0.0],
    "s3_retr32":       [30.3, 11.8, 20.8, 16.9, 16.8, 16.7, 14.8, 13.3, 10.5, 0.0],
    "s4_retr128":      [28.6, 14.7, 21.2, 15.5, 17.4, 17.9, 16.4, 15.6, 10.0, 0.0],
}

# ── 2.2 matched per-seed deltas (pp) ─────────────────────────────────
MATCHED_DELTAS: Dict[str, Tuple[float, float]] = {
    "a1_no_kd":        (-1.480000, 11.659259),
    "a2_head_eval":    (-10.460000, 30.355557),
    "a3_linear_head":  (-0.786667, 0.122223),
    "a4_random_bank":  (-2.390000, 1.159260),
    "s1_budget500":    (-10.053334, 13.259260),
    "s2_budget4000":   (3.320000, -3.911111),
    "s3_retr32":       (-1.530001, -0.007407),
    "s4_retr128":      (0.160000, 0.592593),
}

# ── A6 bank sizes ────────────────────────────────────────────────────
BANK_SIZES: Dict[int, List[Tuple[int, int]]] = {
    500:  [(50, 50), (25, 25), (16, 17), (12, 13), (10, 10),
           (8, 9), (7, 8), (6, 7), (5, 6), (5, 5)],
    2000: [(200, 200), (100, 100), (66, 67), (50, 50), (40, 40),
           (33, 34), (28, 29), (25, 25), (22, 23), (20, 20)],
    4000: [(400, 400), (200, 200), (133, 134), (100, 100), (80, 80),
           (66, 67), (57, 58), (50, 50), (44, 45), (40, 40)],
}

# ── A5 wall-time means (s) ───────────────────────────────────────────
WALL_TIMES: Dict[str, float] = {
    "baseline": 2022.45, "icarl": 3649.27, "static_bank": 2756.36, "uniform_herding": 4635.36,
    "a1_no_kd": 3446.05, "a2_head_eval": 4332.37, "a3_linear_head": 3957.35,
    "a4_random_bank": 4336.75, "s1_budget500": 4236.96, "s2_budget4000": 4309.41,
    "s3_retr32": 3516.54, "s4_retr128": 5285.69,
}

# ── reference per-task forgetting std (population over seeds) ────────
REF_FORGET_STD = [3.7, 1.4, 0.9, 1.9, 2.0, 3.5, 2.3, 2.6, 3.4, 0.0]

# ── T2 significance flags (two-sided paired t-test on per-seed deltas) ─
# Expected flags per plan table_plan.md T2 / report Section 5:
#   sig: p < 0.05 | marginal: p < 0.10 | n.s.: else
SIG_FLAGS: Dict[str, Tuple[str, str]] = {
    "a1_no_kd":       ("n.s.", "sig"),
    "a2_head_eval":   ("sig", "sig"),
    "a3_linear_head": ("marginal", "n.s."),
    "a4_random_bank": ("sig", "n.s."),
}


def verify_master(runs: Dict[str, RunResult]) -> None:
    for key, (ea, esa, ef, esf, eb, esb) in MASTER.items():
        r = runs[key]
        check(f"2.1 {key} avg_acc", close(r.avg_acc, ea, 5e-5), f"{r.avg_acc:.6f}")
        check(f"2.1 {key} avg_acc_std", close(r.avg_acc_std, esa, 5e-5), f"{r.avg_acc_std:.6f}")
        check(f"2.1 {key} forgetting", close(r.forgetting, ef, 5e-5), f"{r.forgetting:.6f}")
        check(f"2.1 {key} forgetting_std", close(r.forgetting_std, esf, 5e-5), f"{r.forgetting_std:.6f}")
        check(f"2.1 {key} bwt", close(r.bwt, eb, 5e-5), f"{r.bwt:.6f}")
        check(f"2.1 {key} bwt_std", close(r.bwt_std, esb, 5e-5), f"{r.bwt_std:.6f}")
    # aggregation consistency: bwt == -mean of per-seed? (check mean over seeds)
    for key in C.MASTER_ORDER:
        r = runs[key]
        mean_seed_bwt = float(np.mean(list(r.per_seed_bwt().values())))
        check(f"2.1 {key} bwt==mean(per-seed bwt)",
              close(mean_seed_bwt, r.bwt, 1e-4), f"seeds {mean_seed_bwt:.6f}")


def verify_per_task(runs: Dict[str, RunResult]) -> None:
    for key, exp in PER_TASK_ACC.items():
        got = runs[key].final_task_accs()
        check(f"2.3 {key} per-task acc", close_arr(got, exp, 0.05),
              f"t1={got[1]:.1f} t9={got[9]:.1f}")
    for key, exp in PER_TASK_FORGET.items():
        got = runs[key].per_task_forgetting()
        check(f"2.4 {key} per-task forgetting", close_arr(got, exp, 0.05),
              f"t1={got[1]:.1f} t9={got[9]:.1f}")
    # t9 == 0.0 by construction
    for key in C.MASTER_ORDER:
        got = runs[key].per_task_forgetting()
        check(f"2.4 {key} t9=0", close(got[9], 0.0, 1e-9), f"{got[9]:.4f}")
    # a2 correction guard: T1 = 47.7, NOT 42.1 (regression guard)
    a2 = runs["a2_head_eval"].per_task_forgetting()
    check("3 a2 T1 == 47.7 (corrected)", close(a2[1], 47.7, 0.05), f"{a2[1]:.2f}")
    check("3 a2 plateau T0-T6 >= 47", all(47.0 <= v <= 55.5 for v in a2[:7]),
          f"{[round(v,1) for v in a2[:7]]}")


def verify_deltas(runs: Dict[str, RunResult]) -> None:
    deltas = matched_deltas(runs, C.reference_key())
    for key, (e_acc, e_for) in MATCHED_DELTAS.items():
        acc = float(np.mean(list(deltas[key]["avg_acc"].values()))) * 100.0
        for_ = float(np.mean(list(deltas[key]["forgetting"].values()))) * 100.0
        check(f"2.2 {key} delta acc", close(acc, e_acc, 1e-4), f"{acc:.6f}")
        check(f"2.2 {key} delta forgetting", close(for_, e_for, 1e-4), f"{for_:.6f}")


def verify_banks(runs: Dict[str, RunResult]) -> None:
    for budget, expected in BANK_SIZES.items():
        key = {500: "s1_budget500", 2000: "uniform_herding", 4000: "s2_budget4000"}[budget]
        r = runs[key]
        got = []
        for t in range(C.NUM_TASKS):
            lo, hi = r.bank_quota_range(t)
            got.append((lo, hi))
            exp = expected[t]
            if (lo, hi) != exp:
                check(f"A6 budget {budget} t{t}", False, f"got {(lo, hi)} expected {exp}")
        check(f"A6 budget {budget} all tasks", got == expected, str(got))
    # identical across seeds for the same budget
    for key in ["s1_budget500", "uniform_herding", "s2_budget4000"]:
        r = runs[key]
        raw = {s: r.bank_sizes[s] for s in r.bank_sizes}
        same = all(raw[s] == raw[list(raw)[0]] for s in raw)
        check(f"A6 {key} seed-invariant", same)


def verify_meta(runs: Dict[str, RunResult]) -> None:
    for key in C.MASTER_ORDER:
        r = runs[key]
        check(f"A5 {key} wall_time", close(r.wall_time_s, WALL_TIMES[key], 0.15),
              f"{r.wall_time_s:.2f}")
        check(f"A5 {key} per-seed walls present", len(r.per_seed_wall_times()) == 3)
        check(f"A1 {key} epochs==71", close(r.epochs_per_task(), 71.0, 1e-9),
              f"{r.epochs_per_task():g}")
        check(f"A1 {key} device present", bool(r.meta.get("device")))
        check(f"A1 {key} git_commit present", bool(r.meta.get("git_commit")))
    ref = runs[C.reference_key()]
    check("A3 ref forgetting std", close_arr(ref.per_task_forgetting_std(), REF_FORGET_STD, 0.05),
          str([round(v, 1) for v in ref.per_task_forgetting_std()]))
    # A4: per-seed means match aggregated
    for key in C.MASTER_ORDER:
        r = runs[key]
        mean_seed = float(np.mean(list(r.per_seed_avg_accs().values())))
        check(f"A4 {key} mean(seed acc)==agg", close(mean_seed, r.avg_acc, 1e-6),
              f"{mean_seed:.6f}")


def verify_significance(runs: Dict[str, RunResult]) -> None:
    print("  (Table 2 significance — two-sided paired t-test, per-seed deltas vs 0)")
    deltas = matched_deltas(runs, C.reference_key())
    for key in [k for k in C.COMPONENT_KEYS if k != C.reference_key()]:
        acc = list(deltas[key]["avg_acc"].values())
        for_ = list(deltas[key]["forgetting"].values())
        p_acc = float(stats.ttest_1samp(acc, 0.0).pvalue)
        p_for = float(stats.ttest_1samp(for_, 0.0).pvalue)
        flag = lambda p: "sig" if p < 0.05 else ("marginal" if p < 0.10 else "n.s.")
        f_acc, f_for = flag(p_acc), flag(p_for)
        check(f"T2 {key} sig flags valid", f_acc in {"sig", "marginal", "n.s."}
              and f_for in {"sig", "marginal", "n.s."},
              f"p_acc={p_acc:.4f} ({f_acc}), p_for={p_for:.4f} ({f_for})")
        exp_acc, exp_for = SIG_FLAGS[key]
        check(f"T2 {key} sig matches plan",
              f_acc == exp_acc and f_for == exp_for,
              f"computed ({f_acc}/{f_for}) vs plan ({exp_acc}/{exp_for}) "
              f"p_acc={p_acc:.4f} p_for={p_for:.4f}")


def verify_artifacts(out_dir: Path) -> None:
    expected = []
    expected += [out_dir / C.PAPER_MAIN_FIGURES_DIR / f"{n}.png" for n in C.PAPER_MAIN_FIGURES]
    expected += [out_dir / C.PAPER_MAIN_FIGURES_DIR / f"{n}.pdf" for n in C.PAPER_MAIN_FIGURES]
    expected += [out_dir / C.PAPER_APPENDIX_FIGURES_DIR / f"{n}.png" for n in C.PAPER_APPENDIX_FIGURES]
    expected += [out_dir / C.PAPER_APPENDIX_FIGURES_DIR / f"{n}.pdf" for n in C.PAPER_APPENDIX_FIGURES]
    for t in C.PAPER_TABLES:
        expected += [out_dir / C.PAPER_TABLES_DIR / f"{t}.tex", out_dir / C.PAPER_TABLES_DIR / f"{t}.md"]
    for p in expected:
        check(f"artifact {p.name}", p.exists() and p.stat().st_size > 0, str(p.relative_to(out_dir)))
    check("artifact count", len(expected) == 9 * 2 + 18,
          f"expected 36 files (9 figures x 2 formats + 18 tables), got {len(expected)}")


def verify_registry() -> None:
    check("registry main complete",
          set(main_figures.BUILDERS) == set(C.PAPER_MAIN_FIGURES))
    check("registry appendix complete",
          set(appendix_figures.BUILDERS) == set(C.PAPER_APPENDIX_FIGURES))
    check("registry tables complete",
          set(tables.BUILDERS) == set(C.PAPER_TABLES))


def main() -> None:
    cfg = get_config()
    out_dir = get_output_root(cfg)
    runs = load_all_runs()
    missing = [k for k in C.MASTER_ORDER if k not in runs]
    if missing:
        print(f"[FATAL] missing runs: {missing}")
        sys.exit(1)

    print("=" * 72)
    print("PAPER ARTIFACT VERIFICATION (independent re-derivation)")
    print("=" * 72)
    verify_registry()
    verify_master(runs)
    verify_per_task(runs)
    verify_deltas(runs)
    verify_banks(runs)
    verify_meta(runs)
    verify_significance(runs)
    verify_artifacts(out_dir)
    print("=" * 72)
    print(f"Checks: {CHECKS} | Failures: {len(FAILURES)}")
    if FAILURES:
        print("FAILED CHECKS:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
