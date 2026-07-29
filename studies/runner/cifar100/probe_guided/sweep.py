from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from studies.runner.cifar100.probe_guided.run import ProbeGuidedCIFAR100Runner


def main() -> None:
    runner = ProbeGuidedCIFAR100Runner(overrides=[
        "runner.seeds=[42, 1337, 2024]",
        "runner.epochs_per_task=70",
        "method=probe_guided",
        "method.gamma=0.5",
        "method.beta=1.0",
        "method.calibrate=false",
    ])
    metrics = runner.run()
    for m in metrics:
        avg_acc = m.get("test/avg_acc_mean", "N/A")
        forget = m.get("test/forgetting_mean", "N/A")
        spearman = m.get("probe/spearman_r_mean", "N/A")
        print(f"\nMethod: {m['method']}")
        print(f"  Avg Acc: {avg_acc}")
        print(f"  Forgetting: {forget}")
        print(f"  Spearman r: {spearman}")


if __name__ == "__main__":
    main()
