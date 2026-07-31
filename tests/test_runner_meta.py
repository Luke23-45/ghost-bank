"""Tests for run metadata provenance collection."""

import datetime
import json

from omegaconf import OmegaConf

from studies.runner.cifar100.run import _build_run_meta


def _cfg(method="static_bank"):
    return OmegaConf.create({
        "runner": {"experiment_name": "cifar100"},
        "method": {"name": method},
        "data": {"num_tasks": 10},
    })


class TestBuildRunMeta:
    def test_basic_fields(self):
        started = datetime.datetime(2026, 8, 1, 12, 0, 0)
        meta = _build_run_meta(_cfg(), [13, 42], started)
        assert meta["experiment"] == "cifar100"
        assert meta["method"] == "static_bank"
        assert meta["seeds"] == [13, 42]
        assert meta["num_seeds"] == 2
        assert meta["num_tasks"] == 10
        assert meta["started_at"].startswith("2026-08-01T12:00:00")
        assert meta["git_commit"] is not None
        assert isinstance(meta["git_dirty"], bool)
        assert meta["python"].startswith("3.")
        assert "torch" in meta and "pytorch_lightning" in meta

    def test_json_serializable(self):
        meta = _build_run_meta(_cfg("baseline"), [13], datetime.datetime.now())
        json.dumps(meta)

    def test_wall_time_positive(self):
        started = datetime.datetime.now() - datetime.timedelta(seconds=5)
        meta = _build_run_meta(_cfg("baseline"), [13], started)
        assert meta["wall_time_s"] > 0
        assert meta["finished_at"] >= meta["started_at"]
