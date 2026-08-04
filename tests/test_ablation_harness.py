"""Unit tests for the ablation harness (no GPU, no training).

Covers the three protocol layers (declared-key validity, override
landing, reference lock), row declarations across all families,
report math with synthetic metrics, and resume detection.  Hydra
composition tests exercise the real configs (fast, no data loading).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.runner.abalation.component import ROWS as COMPONENT_ROWS  # noqa: E402
from studies.runner.abalation.sensitivity import ROWS as SENSITIVITY_ROWS  # noqa: E402
from studies.runner.abalation.shared.executor import (  # noqa: E402
    RunOutcome,
    _iter_completed_run_dirs,
    _run_one_row,
    compose_row_cfg,
    parse_seeds,
    select_rows,
)
from studies.runner.abalation.shared.protocol import (  # noqa: E402
    ALLOWED_OVERRIDE_KEYS,
    REFERENCE_SCHEMA,
    ProtocolViolation,
    canonical_fingerprint,
    flatten_cfg,
    parse_override_value,
    validate_resolved_config,
)
from studies.runner.abalation.shared.report import (  # noqa: E402
    ACC_KEY,
    effect_label,
    mean_std,
    monotonic_violations,
    per_seed_map,
)
from studies.runner.abalation.shared.rows import (  # noqa: E402
    AblationRow,
    UNIFORM_REF_BASE,
)

ALL_ROWS = COMPONENT_ROWS + SENSITIVITY_ROWS
SEEDS = [1993, 2023, 42]


# ---------------------------------------------------------------------------
# Row declarations
# ---------------------------------------------------------------------------


class TestRowDeclarations:
    def test_all_labels_unique(self):
        labels = [row.label for row in ALL_ROWS]
        assert len(labels) == len(set(labels)), "row labels must be unique"

    def test_all_override_keys_are_allowed(self):
        for row in ALL_ROWS:
            for key in row.override_keys:
                assert key in ALLOWED_OVERRIDE_KEYS, (
                    f"{row.label}: {key} not allowed"
                )

    def test_family_counts(self):
        assert len(COMPONENT_ROWS) == 4
        assert len(SENSITIVITY_ROWS) == 4
        assert len(ALL_ROWS) == 8

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="expected_direction"):
            AblationRow("x", "f", "uniform_herding", expected_direction="sideways")

    def test_override_keys_strips_plus(self):
        row = AblationRow(
            "x", "f", "uniform_herding", ["+model.head=linear", "++bank.seed=3"]
        )
        assert row.override_keys == ["model.head", "bank.seed"]


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


class TestParseOverrideValue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("false", False),
            ("null", None),
            ("42", 42),
            ("-7", -7),
            ("0.5", 0.5),
            ("1.0", 1.0),
            ("[]", []),
            ("[1993, 2023, 42]", [1993, 2023, 42]),
            ("cosine_margin", "cosine_margin"),
            ("16-mixed", "16-mixed"),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_override_value(raw) == expected


class TestFingerprint:
    def test_flatten_and_fingerprint_stable(self):
        cfg = OmegaConf.create(
            {"a": {"b": 1}, "c": [1, 2], "d": "x"}
        )
        flat = flatten_cfg(cfg)
        assert flat == {"a.b": 1, "c": [1, 2], "d": "x"}
        assert canonical_fingerprint(cfg) == canonical_fingerprint(cfg)

    def test_schema_is_anchor_locked(self):
        assert REFERENCE_SCHEMA["model.head"] == "cosine_margin"
        assert REFERENCE_SCHEMA["method.kd_weight"] == 1.0
        assert REFERENCE_SCHEMA["runner.seeds"] == [1993, 2023, 42]


# ---------------------------------------------------------------------------
# Composition + validation against the real configs
# ---------------------------------------------------------------------------


def _compose(label: str, method: str, overrides: list[str]):
    row = AblationRow(label=label, family="test", method=method, overrides=overrides)
    return compose_row_cfg(row, SEEDS, Path(f"__test_cfg__/{label}"))


class TestComposeValidate:
    def test_locked_base_config_composes(self):
        cfg, run_name = _compose("ref", "uniform_herding", UNIFORM_REF_BASE)
        assert run_name == "uniform_herding"
        assert cfg.model.head == "cosine_margin"
        assert cfg.method.kd_weight == 1.0
        assert list(cfg.runner.seeds) == SEEDS

    def test_all_family_rows_compose(self):
        for row in ALL_ROWS:
            cfg, _ = compose_row_cfg(
                row, SEEDS, Path(f"__test_cfg__/{row.family}/{row.label}")
            )
            assert cfg.method.name == row.method

    def test_sensitivity_budget_lands(self):
        row = next(r for r in SENSITIVITY_ROWS if r.label == "s1_budget500")
        cfg, _ = compose_row_cfg(row, SEEDS, Path("__test_cfg__/s1"))
        assert cfg.data.memory_total == 500

    def test_typo_key_rejected(self):
        with pytest.raises(ProtocolViolation, match="not in the reference schema"):
            _compose("typo", "uniform_herding", ["method.kd_wght=1.0"])

    def test_lock_drift_rejected(self):
        cfg, _ = _compose("ref2", "uniform_herding", UNIFORM_REF_BASE)
        cfg.model.head = "linear"
        with pytest.raises(ProtocolViolation, match="model.head"):
            validate_resolved_config(
                cfg, UNIFORM_REF_BASE, SEEDS,
                declared_extra=[("method.name", "uniform_herding"), ("bank.name", "herding")],
            )

    def test_declared_value_mismatch_rejected(self):
        cfg, _ = _compose("ref3", "uniform_herding", UNIFORM_REF_BASE)
        # The row declares head=linear but the resolved config is cosine:
        # the harness must not silently accept a claimed-but-not-landed value.
        with pytest.raises(ProtocolViolation, match="did not match the resolved config"):
            validate_resolved_config(
                cfg, ["model.head=linear"], SEEDS,
                declared_extra=[("method.name", "uniform_herding"), ("bank.name", "herding")],
            )

    def test_method_name_mismatch_rejected(self):
        cfg, _ = _compose("ref4", "uniform_herding", UNIFORM_REF_BASE)
        with pytest.raises(ProtocolViolation):
            validate_resolved_config(
                cfg, UNIFORM_REF_BASE, SEEDS,
                declared_extra=[("method.name", "baseline"), ("bank.name", "herding")],
            )

    def test_seeds_override_lands(self):
        row = AblationRow("ref5", "test", "uniform_herding", overrides=UNIFORM_REF_BASE)
        cfg, _ = compose_row_cfg(row, [1, 2], Path("__test_cfg__/ref5"))
        assert list(cfg.runner.seeds) == [1, 2]
        validate_resolved_config(cfg, UNIFORM_REF_BASE, [1, 2], declared_extra=[
            ("method.name", "uniform_herding"), ("bank.name", "herding"),
        ])


# ---------------------------------------------------------------------------
# Resume / skip detection
# ---------------------------------------------------------------------------


class TestResume:
    def test_iter_completed_run_dirs(self, tmp_path):
        (tmp_path / "run1" / "results").mkdir(parents=True)
        (tmp_path / "run2" / "results").mkdir(parents=True)
        (tmp_path / "run1" / "results" / "final_results.json").write_text("{}")
        assert len(_iter_completed_run_dirs(tmp_path)) == 1

    def test_no_false_positive_without_results(self, tmp_path):
        (tmp_path / "run1" / "results").mkdir(parents=True)
        (tmp_path / "run1" / "results" / "other.json").write_text("{}")
        assert _iter_completed_run_dirs(tmp_path) == []

    def test_report_only_missing(self, tmp_path):
        row = COMPONENT_ROWS[0]
        outcome = _run_one_row(
            row, SEEDS, tmp_path, force=False, dry_run=True, report_only=True,
            verbose=False,
        )
        assert outcome.status == "MISSING"

    def test_dry_run_readiness(self, tmp_path):
        row = COMPONENT_ROWS[0]
        outcome = _run_one_row(
            row, SEEDS, tmp_path, force=False, dry_run=True, report_only=False,
            verbose=False,
        )
        assert outcome.status == "READY"
        assert outcome.resolved_sha256 is not None

    def test_skip_existing_run(self, tmp_path):
        row = COMPONENT_ROWS[0]
        row_root = tmp_path / row.family / row.label
        run_dir = row_root / "cifar100" / "uniform_herding" / "20260101_000000"
        (run_dir / "results").mkdir(parents=True)
        payload = {
            "aggregated": {"test/avg_acc_mean": 0.45, "test/avg_acc_std": 0.01},
            "per_seed_metrics": [],
        }
        (run_dir / "results" / "final_results.json").write_text(json.dumps(payload))
        outcome = _run_one_row(
            row, SEEDS, tmp_path, force=False, dry_run=False, report_only=False,
            verbose=False,
        )
        assert outcome.status == "SKIP"
        assert outcome.aggregated["test/avg_acc_mean"] == 0.45


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


class TestCliHelpers:
    def test_parse_seeds(self):
        assert parse_seeds("1993,2023,42") == [1993, 2023, 42]
        assert parse_seeds(" 5 , 6 ") == [5, 6]

    def test_select_rows_subset(self):
        selected = select_rows(COMPONENT_ROWS, "a1_no_kd,a3_linear_head")
        assert [r.label for r in selected] == ["a1_no_kd", "a3_linear_head"]

    def test_select_rows_unknown(self):
        with pytest.raises(SystemExit):
            select_rows(COMPONENT_ROWS, "zzz")


# ---------------------------------------------------------------------------
# Report math with synthetic metrics
# ---------------------------------------------------------------------------


class TestReportMath:
    def test_mean_std(self):
        mean, std = mean_std([1.0, 2.0, 3.0])
        assert mean == 2.0
        assert std == pytest.approx((2 / 3) ** 0.5)

    def test_per_seed_map(self):
        data = {"per_seed_metrics": [{"seed": 1993, "test/avg_acc": 0.4}]}
        assert per_seed_map(data) == {1993: {"seed": 1993, "test/avg_acc": 0.4}}

    def test_effect_label(self):
        assert effect_label(0.05, "increase", ACC_KEY) == "H+ (supports)"
        assert effect_label(-0.05, "increase", ACC_KEY) == "H- (contradicts)"
        assert effect_label(0.005, "increase", ACC_KEY) == "~ (no effect)"
        assert effect_label(-0.05, "neutral", ACC_KEY) == "down"

    def test_monotonic_violations(self):
        assert monotonic_violations([("a", 0.5), ("b", 0.4)], higher_better=True) == [
            "a -> b"
        ]
        assert monotonic_violations([("a", 0.5), ("b", 0.4)], higher_better=False) == []
        assert monotonic_violations([("a", None), ("b", 0.4)], higher_better=True) == []


def test_csv_header_body_width_match():
    from studies.runner.abalation.shared.report import build_csv

    per_seed = [
        {"seed": 1993, "test/avg_acc": 0.4510, "test/forgetting": 0.1337},
        {"seed": 2023, "test/avg_acc": 0.4616, "test/forgetting": 0.1377},
    ]
    row = AblationRow("a1", "component", "uniform_herding", expected_direction="decrease")
    outcome = RunOutcome(
        row, status="RUN", aggregated={"test/avg_acc_mean": 0.45},
        per_seed=per_seed,
    )
    csv = build_csv(
        "baselines",
        [outcome],
        {m["seed"]: m for m in per_seed},
    )
    header, body = csv.strip().splitlines()
    assert len(header.split(",")) == len(body.split(",")), (
        "CSV header/body width mismatch"
    )
    assert "a1" in body
    assert "effect" in header