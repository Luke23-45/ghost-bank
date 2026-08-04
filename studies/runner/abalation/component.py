"""Component ablation family: a1..a4.

Each row removes or replaces exactly one headline design decision of the
locked reference (uniform herding + cosine-margin head + KD + herding
selection) to attribute the reference's performance:

- a1_no_kd        knowledge distillation on/off  (iCaRL mechanism)
- a2_head_eval    NME vs classifier-head evaluation (eval protocol)
- a3_linear_head  cosine-margin vs linear classifier geometry
- a4_random_bank  herding vs random exemplar selection (the core claim)

Second-order details (margin magnitude, LUCIR-style imprinting, KD
weight/temperature) are deliberately not ablated in the main study.
All rows run the full 3-seed protocol and report per-seed deltas vs
the locked reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.runner.abalation.shared.executor import run_family_cli  # noqa: E402
from studies.runner.abalation.shared.rows import AblationRow, UNIFORM_REF_BASE  # noqa: E402

FAMILY = "component"

ROWS: list[AblationRow] = [
    AblationRow(
        label="a1_no_kd",
        family=FAMILY,
        method="uniform_herding",
        overrides=["model.head=cosine_margin", "method.kd_weight=0.0"],
        hypothesis="KD off: distillation from the frozen old model removed.",
        expected_direction="decrease",
    ),
    AblationRow(
        label="a2_head_eval",
        family=FAMILY,
        method="uniform_herding",
        overrides=[*UNIFORM_REF_BASE, "method.predict_mode=head"],
        hypothesis="NME -> head logits for evaluation.",
        expected_direction="neutral",
    ),
    AblationRow(
        label="a3_linear_head",
        family=FAMILY,
        method="uniform_herding",
        overrides=["model.head=linear", "method.kd_weight=1.0"],
        hypothesis="Cosine-margin head replaced by a plain linear classifier.",
        expected_direction="decrease",
    ),
    AblationRow(
        label="a4_random_bank",
        family=FAMILY,
        method="uniform_herding",
        overrides=[*UNIFORM_REF_BASE, "bank.selection=random"],
        hypothesis="Herding exemplar selection replaced by seeded random selection.",
        expected_direction="decrease",
    ),
]


def main() -> int:
    return run_family_cli(FAMILY, ROWS)


if __name__ == "__main__":
    sys.exit(main())