from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.replay_ablation import AblationConfig, _run_variant


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        return {
            "mean": float(values[0]),
            "std": 0.0,
            "min": float(values[0]),
            "max": float(values[0]),
        }
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _run_method(method: str, seeds: list[int], cfg: AblationConfig, output_dir: Path) -> dict:
    if method not in {"uniform_herding", "probe_blend_herding"}:
        raise ValueError(f"Unsupported confirmation method: {method}")

    allocation_mode = "uniform" if method == "uniform_herding" else "probe_blend"
    selection_policy = "herding"

    runs: list[dict] = []
    for seed in seeds:
        seed_cfg = AblationConfig(
            seed=seed,
            tasks=cfg.tasks,
            classes_per_task=cfg.classes_per_task,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            memory_total=cfg.memory_total,
            retrieval_budget=cfg.retrieval_budget,
            probe_split_size=cfg.probe_split_size,
            val_split_size=cfg.val_split_size,
            split_seed=cfg.split_seed,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
        )
        runs.append(
            _run_variant(
                variant_name=method,
                allocation_mode=allocation_mode,
                selection_policy=selection_policy,
                cfg=seed_cfg,
                output_dir=output_dir,
            )
        )

    raw_accs = [float(run["raw/test/avg_acc"]) for run in runs]
    nme_accs = [float(run["nme/test/avg_acc"]) for run in runs]
    probe_rs = [
        float(run["probe/spearman_r"])
        for run in runs
        if run.get("probe/spearman_r") is not None
    ]
    raw_forgetting = [float(run["raw/test/forgetting"]) for run in runs]
    raw_bwt = [float(run["raw/test/backward_transfer"]) for run in runs]

    return {
        "method": method,
        "seeds": seeds,
        "runs": runs,
        "summary": {
            "raw/test/avg_acc": _summarize(raw_accs),
            "nme/test/avg_acc": _summarize(nme_accs),
            "probe/spearman_r": _summarize(probe_rs),
            "raw/test/forgetting": _summarize(raw_forgetting),
            "raw/test/backward_transfer": _summarize(raw_bwt),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirmation sweep for uniform_herding vs probe_blend_herding.",
    )
    parser.add_argument(
        "--method",
        default="compare",
        choices=["compare", "uniform_herding", "probe_blend_herding"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 23])
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--memory-total", type=int, default=2000)
    parser.add_argument("--retrieval-budget", type=int, default=64)
    parser.add_argument("--probe-split-size", type=int, default=30)
    parser.add_argument("--val-split-size", type=int, default=20)
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis" / "outputs",
    )
    args = parser.parse_args()

    if args.accelerator == "cpu" and args.precision == "16-mixed":
        args.precision = "32-true"

    seed_tag = "-".join(str(seed) for seed in args.seeds)
    sweep_dir = args.output / f"confirmation_sweep_tasks{args.tasks}_{args.method}_seeds{seed_tag}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = AblationConfig(
        seed=args.seeds[0],
        tasks=args.tasks,
        classes_per_task=10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        memory_total=args.memory_total,
        retrieval_budget=args.retrieval_budget,
        probe_split_size=args.probe_split_size,
        val_split_size=args.val_split_size,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )

    methods = (
        ["uniform_herding", "probe_blend_herding"]
        if args.method == "compare"
        else [args.method]
    )

    results: dict[str, dict] = {}
    for method in methods:
        results[method] = _run_method(method, args.seeds, base_cfg, sweep_dir)

    out_path = sweep_dir / f"confirmation_sweep_tasks{args.tasks}_{args.method}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[analysis] wrote {out_path}", flush=True)
    print(json.dumps({m: results[m]["summary"] for m in methods}, indent=2), flush=True)


if __name__ == "__main__":
    main()
