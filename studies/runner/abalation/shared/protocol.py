"""Protocol/validation for the ablation harness.

The whole point of this module is to prevent silent misconfiguration: a
row that swaps the wrong knob (or a knob that is silently dropped from
the resolved config) must fail loudly *before* any GPU budget is spent.

Three independent checks run against every composed row config:

1. **Declared-key validity**: every Hydra override a row declares must
   name a real key from ``REFERENCE_SCHEMA`` (or an explicitly allowed
   runtime key).  This turns a typo such as ``method.kd_wght=1.0`` into
   an immediate error instead of a silently ignored override.
2. **Override values land**: each declared override must exist in the
   final resolved config with exactly the declared value.  This catches
   the classic Hydra struct bug where ``method.kd_weight`` is silently
   dropped because a config group was never activated.
3. **Reference lock**: every non-overridden, present key of
   ``REFERENCE_SCHEMA`` must equal the anchored value.  This catches
   silent drift of the underlying YAML configs (e.g. the anchored
   reference used ``model.head=cosine_margin`` while the shipped
   default is ``model.head=linear``).

``REFERENCE_SCHEMA`` is the exact resolved config of the published
anchor run
(``debug_run/final_run/output/cifar100/uniform_herding/20260803_143548``,
avg_acc 0.4499 +- 0.010, forgetting 0.1385 +- 0.0044, seeds 1993/2023/42).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Locked reference schema (from the published anchor run's resolved config).
# ---------------------------------------------------------------------------

REFERENCE_SCHEMA: dict[str, Any] = {
    "data.type": "cifar100",
    "data.root": "./data/cifar100",
    "data.seed": 13,
    "data.batch_size": 128,
    "data.num_workers": 4,
    "data.pin_memory": True,
    "data.persistent_workers": True,
    "data.prefetch_factor": 2,
    "data.num_tasks": 10,
    "data.classes_per_task": 10,
    "data.mean": [0.5071, 0.4867, 0.4408],
    "data.std": [0.2675, 0.2565, 0.2761],
    "data.probe_split_size": 30,
    "data.val_split_size": 20,
    "data.split_seed": 13,
    "data.memory_total": 2000,
    "data.probe_enabled": True,
    "model.type": "resnet",
    "model.depth": 18,
    "model.num_classes": 10,
    "model.base_filters": 64,
    "model.dropout": 0.0,
    "model.head": "cosine_margin",
    "model.head_scale": 30.0,
    "model.head_margin": 0.35,
    "model.imprint": True,
    "method.name": "uniform_herding",
    "method.retrieval_budget": 64,
    "method.warmup_steps": 0,
    "method.kd_weight": 1.0,
    "method.kd_temperature": 2.0,
    "method.predict_mode": "nme",
    "training.learning_rate": 0.1,
    "training.optimizer": "sgd",
    "training.momentum": 0.9,
    "training.weight_decay": 0.0005,
    "training.max_epochs": 70,
    "training.log_every_n_steps": 10,
    "training.gradient_clip_val": 1.0,
    "training.enable_progress_bar": True,
    "training.progress_refresh_rate": 1,
    "training.accelerator": "gpu",
    "training.devices": 1,
    "training.precision": "16-mixed",
    "training.logging.level": "warning",
    "output.save_checkpoint": False,
    "debug": False,
    "bank.name": "herding",
    "bank.floor": 1,
    "bank.exclude_classes": [],
    "bank.selection": "herding",
    "runner.experiment_name": "cifar100",
    "runner.epochs_per_task": 70,
    "runner.seeds": [1993, 2023, 42],
}

# Runtime/input-carrying keys the harness legitimately sets.  Excluded from
# the lock check, but every explicit override of them is still
# value-verified against the resolved config.
RUNTIME_KEYS = {
    "output.base_dir",
    "runner.methods",
    "runner.seeds",
    "data.root",
    "runner.experiment_name",
}

# Real config keys that are not part of the anchor schema but may be tuned
# by a row (e.g. the static bank per-class budget or an explicit bank seed).
EXTRA_ALLOWED_KEYS = {
    "bank.capacity_per_class",
    "bank.seed",
    "method.warmup_steps",
}

ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    set(REFERENCE_SCHEMA) | EXTRA_ALLOWED_KEYS | RUNTIME_KEYS
)

REFERENCE_SHA256 = hashlib.sha256(
    json.dumps(REFERENCE_SCHEMA, sort_keys=True).encode("utf-8")
).hexdigest()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def flatten_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Flatten a resolved Hydra config into a ``{dot.path: value}`` dict."""
    container = OmegaConf.to_container(cfg, resolve=True)
    flat: dict[str, Any] = {}

    def _walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{prefix}.{key}" if prefix else str(key))
        else:
            flat[prefix] = node

    _walk(container, "")
    return flat


def canonical_fingerprint(cfg: DictConfig) -> str:
    """Deterministic sha256 of the fully resolved config (provenance)."""
    flat = flatten_cfg(cfg)
    blob = json.dumps(flat, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def parse_override_value(raw: str) -> Any:
    """Parse a Hydra-style override value into a concrete Python value."""
    text = raw.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    if re.match(r"^-?\d+$", text):
        return int(text)
    if re.match(r"^-?\d*\.\d+(?:[eE][+-]?\d+)?$", lowered):
        return float(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        if not inner.strip():
            return []
        return [parse_override_value(part.strip()) for part in inner.split(",")]
    return text


def parse_overrides(override_strings: list[str]) -> list[tuple[str, Any]]:
    """Parse ``key=value`` strings into ``(dot_path, python_value)`` tuples.

    A leading ``+`` / ``++`` (Hydra "add" markers) is stripped: the dot
    path is what validation compares against the resolved config.
    """
    parsed: list[tuple[str, Any]] = []
    for override in override_strings:
        key, _, raw_value = override.partition("=")
        key = key.strip()
        while key and key[0] in "+":
            key = key[1:]
        parsed.append((key, parse_override_value(raw_value)))
    return parsed


def _dot_get(flat: dict[str, Any], key: str, _missing: Any = object()) -> Any:
    return flat.get(key, _missing)


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------


class ProtocolViolation(ValueError):
    """Raised when a resolved config violates the frozen protocol."""


def assert_declared_keys_valid(override_strings: list[str], label: str) -> None:
    """Fail fast (before any Hydra composition) on unknown override keys.

    Guard for the case where Hydra would otherwise raise an opaque
    struct/append error (or, worse, silently add a misspelled key) during
    composition.  The same check also runs inside
    :func:`validate_resolved_config` as layer 1.
    """
    for key, _ in parse_overrides(override_strings):
        if key not in ALLOWED_OVERRIDE_KEYS:
            raise ProtocolViolation(
                f"Row {label!r}: declared override key {key!r} is not in the "
                "reference schema (possible typo)."
            )


def validate_resolved_config(
    cfg: DictConfig,
    override_strings: list[str],
    seeds: list[int],
    declared_extra: list[tuple[str, Any]] | None = None,
) -> None:
    """Raise :class:`ProtocolViolation` if ``cfg`` diverges from the lock.

    Parameters
    ----------
    cfg:
        The row's resolved config, as compiled by the CIFAR runner.
    override_strings:
        The row's declared Hydra overrides (see ``AblationRow.overrides``).
    seeds:
        The requested seed list (validated as landing in ``runner.seeds``).
    declared_extra:
        Synthetic declarations used to value-verify intent that is not
        expressed as a raw Hydra override, e.g. ``("method.name", "icarl")``
        and ``("bank.name", "herding")``.  These keys are exempt from the
        lock comparison exactly like a declared override and are
        value-checked against the resolved config.
    """
    flat = flatten_cfg(cfg)

    # Combine the row's declared overrides, harness seed control and any
    # method/bank intent so every explicit input is value-verified.
    all_overrides = list(override_strings) + [f"runner.seeds={seeds!r}"]
    parsed = parse_overrides(all_overrides) + list(declared_extra or [])

    # --- Layer 1: declared keys must name a real, allowed schema key ------
    for key, _ in parsed:
        if key not in ALLOWED_OVERRIDE_KEYS:
            raise ProtocolViolation(
                f"Declared override key {key!r} is not in the reference "
                "schema (possible typo; the override would be silently "
                "ignored by Hydra). Allowed keys are the reference schema "
                "plus {sorted(EXTRA_ALLOWED_KEYS)!r}."
            )

    # --- Layer 2: every declared override must land with its value --------
    missing: list[str] = []
    mismatched: list[str] = []
    for key, expected in parsed:
        if key not in flat:
            missing.append(f"{key}={expected!r}")
            continue
        if flat[key] != expected:
            mismatched.append(f"{key}: resolved={flat[key]!r} declared={expected!r}")
    if missing:
        raise ProtocolViolation(
            f"Declared overrides never reached the resolved config: {missing!r}. "
            "This is the classic Hydra struct bug (e.g. method.kd_weight "
            "silently dropped when the config group was never activated)."
        )
    if mismatched:
        raise ProtocolViolation(
            f"Declared overrides did not match the resolved config: {mismatched!r}."
        )

    # --- Layer 3: lock keys that were NOT overridden must equal the anchor.
    overridden = {key for key, _ in parsed}
    for key, expected in REFERENCE_SCHEMA.items():
        if key in RUNTIME_KEYS or key in overridden:
            continue
        if key not in flat:
            # Presence-guarded: rows without a bank node (or a method
            # without KD) legitimately lack these keys.
            continue
        if flat[key] != expected:
            raise ProtocolViolation(
                f"Lock key {key!r} drifted from the anchored reference: "
                f"resolved={flat[key]!r} anchor={expected!r}. A row not "
                "explicitly overriding this key must reproduce the locked "
                "reference config exactly."
            )