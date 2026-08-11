python -m src.scripts.generate_paper


# Ghost Bank CIL Paper — Analysis Framework

Generates all figures and tables for the CIFAR-100 class-incremental learning
paper from the persisted run artifacts in `experiment_output/`.

## Architecture

```
studies/analysis/
├── configs/base.yaml          # central path config (OmegaConf)
└── src/
    ├── common/
    │   ├── style.py           # exact reference palette + RC params, figure/save helpers
    │   ├── constants.py       # experiment registry (keys, paths, colors, ordering)
    │   ├── data.py            # run discovery + artifact loaders (RunResult)
    │   ├── plotting.py        # shared plot primitives (curves, heatmaps, bars)
    │   ├── latex.py           # booktabs tables (.tex) + markdown tables
    │   ├── io.py              # json/csv/yaml/text I/O
    │   └── config.py          # OmegaConf-backed config loader
    ├── experiments/           # ONE MODULE PER EXPERIMENT + family-level modules
    │   ├── _shared.py         # thin composition helpers (figure spine + saving)
    │   ├── b1_icarl.py        # B1 — per-task curve, heatmap, accumulation trace
    │   ├── b2_static_bank.py  # B2 — per-task curve, trajectory fan, accumulation
    │   ├── b3_uniform_herding.py  # B3 — per-task curve, fan, heatmap (reference)
    │   ├── a1_no_kd.py        # a1 — curve, stability slopes, accumulation
    │   ├── a2_head_eval.py    # a2 — curve, forgetting-by-age, trajectory fan
    │   ├── a3_linear_head.py  # a3 — curve + per-task delta bars
    │   ├── a4_random_bank.py  # a4 — curve, delta bars, seed-consistency strip
    │   ├── s1_budget500.py    # s1 — curve + delta bars
    │   ├── s2_budget4000.py   # s2 — curve + delta bars
    │   ├── s3_retr32.py       # s3 — curve + delta bars
    │   ├── s4_retr128.py      # s4 — curve + delta bars
    │   ├── baselines.py       # family: B1/B2/B3 overlay comparison
    │   ├── component_ablations.py  # family: overlay + attribution deltas
    │   ├── sensitivity.py     # family: memory/retrieval resource curves
    │   └── cross_cutting.py   # all-runs master charts + paper tables
    └── scripts/               # CLI entrypoints
        ├── generate_all.py           # full pipeline
        ├── generate_figures.py       # figures only (--families ...)
        ├── generate_tables.py        # tables only (--families ...)
        └── generate_experiment.py    # regenerate ONE experiment
```

Every experiment module exposes `generate_figures(runs, out_dir) -> List[Path]`;
every family module exposes both `generate_figures(runs, out_dir)` and
`generate_tables(runs, out_dir)`. `out_dir` is the analysis output root;
each module is responsible for its own sub-path (family/type/experiment +
`figures/` or `tables/`), so outputs mirror the run-directory layout.

## Usage

Run everything (from `studies/analysis/`):

```powershell
python -m src.scripts.generate_paper             # figures + tables
python -m src.scripts.generate_figures             # figures only
python -m src.scripts.generate_tables              # tables only
python -m src.scripts.generate_all --families component sensitivity
```

Per-experiment control (touch one experiment, leave the other 10 untouched):

```powershell
python -m src.scripts.generate_experiment --experiment B3
python -m src.scripts.generate_experiment --experiment a2 --include-family
python -m src.scripts.generate_experiment --experiment s1 --tables
```

Accepted tokens: short names (`B1`, `a2`, `s3`) or registry keys
(`uniform_herding`, `s2_budget4000`).

## Outputs

Outputs mirror the run-directory layout of `experiment_output/`: one folder
per family, one folder per experiment inside it, with `figures/` (and, at
family level, `tables/`) sub-directories:

```
outputs/
├── baseline/                       # mirrors final_baseline_run/
│   ├── figures/                    # baselines_comparison.{png,pdf}
│   ├── tables/                     # baselines.{md,tex}
│   ├── icarl/figures/              # B1 bespoke figures
│   ├── static_bank/figures/        # B2 bespoke figures
│   └── uniform_herding/figures/    # B3 bespoke figures (reference)
├── ablation/
│   ├── component/                  # mirrors abalations/component/
│   │   ├── figures/                # component_comparison, attribution_deltas
│   │   ├── tables/                 # component_ablations.{md,tex}
│   │   ├── a1_no_kd/figures/
│   │   ├── a2_head_eval/figures/
│   │   ├── a3_linear_head/figures/
│   │   └── a4_random_bank/figures/
│   └── sensitivity/                # mirrors abalations/sensitivity/
│       ├── figures/                # memory_curve, retrieval_curve
│       ├── tables/                 # sensitivity.{md,tex}
│       ├── s1_budget500/figures/
│       ├── s2_budget4000/figures/
│       ├── s3_retr32/figures/
│       └── s4_retr128/figures/
└── cross_cutting/
    ├── figures/                    # master charts (5 figures)
    └── tables/                     # master_results, per_seed, per_task
```

Every figure is exported as `.png` (300 DPI) + `.pdf` (type-42 embedded fonts);
every table as `.md` + standalone compilable `.tex` (booktabs).

Figure inventory (38 figure sets x 2 formats = 76 files):

| Family | Figures |
|---|---|
| baseline (B1/B2/B3) | 28 per-experiment bespoke sets (curves, heatmaps, trajectory fans, stability slopes, forgetting-by-age, delta bars, accumulation traces, seed-consistency strip) |
| ablation/component | component comparison overlay; attribution deltas |
| ablation/sensitivity | memory curve; retrieval curve |
| cross_cutting | master accuracy comparison; accuracy-vs-forgetting scatter; all-runs per-task; forgetting heatmap; ranking lollipop |

Table inventory (6 tables = 12 files): baselines, component ablations
(per-seed matched deltas), sensitivity, master results, per-seed results,
per-task accuracies.

## Style

The palette, fonts, DPI and layout presets are the exact style of the
referenced framework (`studies/analysis/references/analysis1/common/style.py`):
colorblind-safe Wong & Tol + Apple neutrals, sans-serif (Aptos/Segoe UI),
single/double/full-width presets (4.5"/6.5"/13.2"), 300 DPI, PDF fonttype 42.

## Adding a new run (e.g. memory 8000)

1. Run the experiment — it lands under `experiment_output/.../<method>/<timestamp>/`.
2. Add one entry to `RUN_PATTERNS` in `src/common/constants.py` (loader auto-
   discovers the newest timestamp; memory/retrieval curves read x-values from
   the run config, so new budget points appear automatically).
3. Add it to the relevant group list (`SENSITIVITY_MEMORY_KEYS`, ...) and to
   `MASTER_ORDER`, `DISPLAY_NAMES`, `COLORS` as needed.
4. `python -m src.scripts.generate_all` — done.

## Notes

- Deltas are matched per-seed against the reference run (same 3-seed set),
  never mean-vs-mean.
- `RunResult.avg_acc` etc. are read from `results/final_results.json`; the
  evolution matrices come from `metrics/aggregated_accuracy_matrix.csv`.
- The analysis package is imported as `src.*`; run the scripts from
  `studies/analysis/` (each script bootstraps its own path).
