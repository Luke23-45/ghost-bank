from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.probe_guided_audit import AuditConfig, _run_short_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare probe-guided, uniform replay, and frozen baseline.")
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--memory-total", type=int, default=2000)
    parser.add_argument("--retrieval-budget", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "outputs")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    args = parser.parse_args()

    if args.accelerator == "cpu" and args.precision == "16-mixed":
        args.precision = "32-true"

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = AuditConfig(
        seed=args.seed,
        tasks=args.tasks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        memory_total=args.memory_total,
        retrieval_budget=args.retrieval_budget,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )
    results = {}
    for method in ("probe_guided", "uniform_replay", "frozen_baseline"):
        results[method] = _run_short_audit(method, cfg)
    out_path = args.output / f"compare_seed{args.seed}_tasks{args.tasks}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[analysis] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
