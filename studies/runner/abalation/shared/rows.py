"""Row definitions for the ablation harness.

An :class:`AblationRow` fully describes one experiment run: which
method the CIFAR100 runner should select and which Hydra overrides move
it away from the locked reference config.  Overrides are declared in
exact Hydra ``key=value`` syntax (the same string format ``compose``
consumes), so a row is self-describing and ``protocol`` can validate the
resolved config against it before any GPU time is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_DIRECTION_ACCEPTED = ("increase", "decrease", "neutral")


@dataclass(frozen=True)
class AblationRow:
    """One ablation experiment.

    Attributes
    ----------
    label:
        Unique row label, e.g. ``a1_no_kd``; also the output directory
        name for the row (``<base>/<family>/<label>/``).
    family:
        Family this row belongs to (``reference``, ``component``,
        ``sensitivity`` or ``control``).
    method:
        Method config group to select, e.g. ``uniform_herding``. The
        runner attaches the matching replay bank automatically.
    overrides:
        Hydra override strings applied on top of the locked reference
        config.  Any key used here must exist in the reference schema,
        otherwise validation fails.
    hypothesis:
        One-line statement of what this row tests.
    expected_direction:
        ``increase``/``decrease``/``neutral``: how the headline metric
        (avg_acc) is expected to move relative to the reference.  Used
        for report flags (never a hard gate outside the reference row).
    """

    label: str
    family: str
    method: str
    overrides: list[str] = field(default_factory=list)
    hypothesis: str = ""
    expected_direction: str = "neutral"

    def __post_init__(self) -> None:
        if self.expected_direction not in _DIRECTION_ACCEPTED:
            raise ValueError(
                f"Row {self.label!r}: expected_direction must be one of "
                f"{_DIRECTION_ACCEPTED!r}, got {self.expected_direction!r}"
            )

    @property
    def override_keys(self) -> list[str]:
        """Dot-path keys of every declared override (typo guard input)."""
        keys: list[str] = []
        for override in self.overrides:
            key = override.split("=", 1)[0].strip()
            while key and key[0] in "+":
                key = key[1:]
            keys.append(key)
        return keys

    @property
    def run_label(self) -> str:
        return f"{self.family}/{self.label}"


# Overrides that lift the shipped config defaults back to the locked
# reference config.  The anchored run used a cosine-margin head and
# kd_weight=1.0 while the current YAML defaults are ``linear`` / ``0.0``,
# so every uniform-herding row must (re)declare them explicitly; the
# protocol lock turns any omission into a hard validation error.
UNIFORM_REF_BASE: list[str] = [
    "model.head=cosine_margin",
    "method.kd_weight=1.0",
]