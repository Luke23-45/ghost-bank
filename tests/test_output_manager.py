"""Tests for the output state machine and OutputManager."""

import json
import os

import pytest

from studies.output.state_machine import OutputState, OutputStateMachine
from studies.output.manager import OutputManager


# -- OutputStateMachine -------------------------------------------------------

class TestOutputStateMachine:
    def test_initial_state(self):
        fsm = OutputStateMachine()
        assert fsm.state == OutputState.INITIALIZED

    def test_happy_path_transitions(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.CONFIG_SAVED)
        assert fsm.state == OutputState.CONFIG_SAVED

        fsm.transition(OutputState.METRICS_OPEN)
        assert fsm.state == OutputState.METRICS_OPEN

        fsm.transition(OutputState.RESULTS_WRITTEN)
        assert fsm.state == OutputState.RESULTS_WRITTEN

        fsm.transition(OutputState.ARTIFACTS_SAVED)
        assert fsm.state == OutputState.ARTIFACTS_SAVED

        fsm.transition(OutputState.COMPLETED)
        assert fsm.state == OutputState.COMPLETED

    def test_invalid_transition_raises(self):
        fsm = OutputStateMachine()
        with pytest.raises(RuntimeError, match="Cannot transition"):
            fsm.transition(OutputState.COMPLETED)

    def test_fail_from_initialized(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.FAILED)
        assert fsm.state == OutputState.FAILED

    def test_fail_from_config_saved(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.CONFIG_SAVED)
        fsm.transition(OutputState.FAILED)
        assert fsm.state == OutputState.FAILED

    def test_fail_from_metrics_open(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.CONFIG_SAVED)
        fsm.transition(OutputState.METRICS_OPEN)
        fsm.transition(OutputState.FAILED)
        assert fsm.state == OutputState.FAILED

    def test_fail_from_results_written(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.CONFIG_SAVED)
        fsm.transition(OutputState.METRICS_OPEN)
        fsm.transition(OutputState.RESULTS_WRITTEN)
        fsm.transition(OutputState.FAILED)
        assert fsm.state == OutputState.FAILED

    def test_complete_from_failed(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.FAILED)
        fsm.transition(OutputState.COMPLETED)
        assert fsm.state == OutputState.COMPLETED

    def test_cannot_fail_from_completed(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.CONFIG_SAVED)
        fsm.transition(OutputState.METRICS_OPEN)
        fsm.transition(OutputState.RESULTS_WRITTEN)
        fsm.transition(OutputState.COMPLETED)
        with pytest.raises(RuntimeError, match="Cannot transition"):
            fsm.transition(OutputState.FAILED)

    def test_cannot_double_complete(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.CONFIG_SAVED)
        fsm.transition(OutputState.METRICS_OPEN)
        fsm.transition(OutputState.RESULTS_WRITTEN)
        fsm.transition(OutputState.COMPLETED)
        with pytest.raises(RuntimeError, match="Cannot transition"):
            fsm.transition(OutputState.COMPLETED)

    def test_transition_to_failed_then_complete(self):
        fsm = OutputStateMachine()
        fsm.transition(OutputState.FAILED)
        fsm.transition(OutputState.COMPLETED)
        assert fsm.state == OutputState.COMPLETED


# -- OutputManager ------------------------------------------------------------

class TestOutputManager:
    def test_initialize_creates_directories(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        root = mgr.initialize()
        assert os.path.isdir(os.path.join(root, "configs"))
        assert os.path.isdir(os.path.join(root, "metrics"))
        assert os.path.isdir(os.path.join(root, "results"))
        assert os.path.isdir(os.path.join(root, "artifacts"))

    def test_initialize_returns_root_path(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        root = mgr.initialize()
        assert mgr.root == root
        assert "test_exp" in root

    def test_double_initialize_raises(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        with pytest.raises(RuntimeError, match="already in state"):
            mgr.initialize()

    def test_save_config_dict(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"key": "value", "num": 42})
        config_path = os.path.join(mgr.root, "configs", "resolved_config.yaml")
        assert os.path.isfile(config_path)
        with open(config_path) as f:
            content = json.load(f)
        assert content["key"] == "value"
        assert content["num"] == 42

    def test_save_config_string(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config("key: value\nnum: 42", name="custom.yaml")
        config_path = os.path.join(mgr.root, "configs", "custom.yaml")
        assert os.path.isfile(config_path)
        with open(config_path) as f:
            assert f.read() == "key: value\nnum: 42"

    def test_save_config_before_initialize_raises(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            mgr.save_config({"a": 1})

    def test_write_metrics_csv(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.write_metrics({"loss": 0.5, "acc": 0.9})
        metrics_path = os.path.join(mgr.root, "metrics", "train_metrics.csv")
        assert os.path.isfile(metrics_path)
        with open(metrics_path) as f:
            content = f.read()
        assert "loss" in content
        assert "0.5" in content

    def test_write_metrics_multiple_rows(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.write_metrics({"loss": 0.5, "epoch": 1})
        mgr.write_metrics({"loss": 0.3, "epoch": 2})
        with open(os.path.join(mgr.root, "metrics", "train_metrics.csv")) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 3

    def test_write_metrics_without_initialize_raises(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            mgr.write_metrics({"loss": 0.5})

    def test_finalize_writes_results(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.write_metrics({"loss": 0.5})
        mgr.finalize({"accuracy": 0.95})
        results_path = os.path.join(mgr.root, "results", "final_results.json")
        assert os.path.isfile(results_path)
        with open(results_path) as f:
            data = json.load(f)
        assert data["accuracy"] == 0.95

    def test_finalize_without_metrics(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.finalize({"accuracy": 0.95})
        results_path = os.path.join(mgr.root, "results", "final_results.json")
        assert os.path.isfile(results_path)

    def test_complete_after_finalize(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.finalize({"acc": 0.9})
        mgr.complete()

    def test_fail_handling(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.fail()
        mgr.complete()

    def test_complete_without_finalize_raises(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        with pytest.raises(RuntimeError):
            mgr.complete()

    def test_save_artifact_file_copy(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("hello")
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.finalize({"acc": 0.9})
        mgr.save_artifact("artifact.txt", str(src))
        artifact_path = os.path.join(mgr.root, "artifacts", "artifact.txt")
        assert os.path.isfile(artifact_path)
        with open(artifact_path) as f:
            assert f.read() == "hello"

    def test_save_artifact_string(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        mgr.finalize({"acc": 0.9})
        mgr.save_artifact("note.txt", "test content")
        artifact_path = os.path.join(mgr.root, "artifacts", "note.txt")
        assert os.path.isfile(artifact_path)

    def test_save_artifact_before_results_raises(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        mgr.initialize()
        mgr.save_config({"a": 1})
        with pytest.raises(RuntimeError):
            mgr.save_artifact("x.txt", "data")

    def test_lifecycle_happy_path(self, tmp_path):
        mgr = OutputManager(experiment="test_exp", base_dir=str(tmp_path))
        assert mgr.root is None
        root = mgr.initialize()
        assert mgr.root is not None
        mgr.save_config({"learning_rate": 0.01})
        mgr.write_metrics({"loss": 0.5})
        mgr.write_metrics({"loss": 0.4})
        mgr.finalize({"accuracy": 0.92, "f1": 0.88})
        mgr.save_artifact("model.pt", "pretend_weights")
        mgr.complete()
        assert os.path.isdir(root)
        assert os.path.isfile(os.path.join(root, "configs", "resolved_config.yaml"))
        assert os.path.isfile(os.path.join(root, "metrics", "train_metrics.csv"))
        assert os.path.isfile(os.path.join(root, "results", "final_results.json"))
        assert os.path.isfile(os.path.join(root, "artifacts", "model.pt"))


# -- MarkdownWriter ------------------------------------------------------------

class TestMarkdownWriter:
    def test_write_single_row(self, tmp_path):
        from studies.output.formatters.markdown_writer import MarkdownWriter
        writer = MarkdownWriter()
        path = str(tmp_path / "test.md")
        writer.write({"a": 1, "b": 2}, path)
        with open(path) as f:
            content = f.read()
        assert "| a | b |" in content
        assert "| 1 | 2 |" in content

    def test_write_multiple_rows(self, tmp_path):
        from studies.output.formatters.markdown_writer import MarkdownWriter
        writer = MarkdownWriter()
        path = str(tmp_path / "test.md")
        writer.write([{"a": 1, "b": 2}, {"a": 3, "b": 4}], path)
        with open(path) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 4

    def test_write_empty_list(self, tmp_path):
        from studies.output.formatters.markdown_writer import MarkdownWriter
        writer = MarkdownWriter()
        path = str(tmp_path / "test.md")
        writer.write([], path)
        assert not os.path.exists(path) or os.path.getsize(path) == 0

    def test_append_creates_header_on_first_call(self, tmp_path):
        from studies.output.formatters.markdown_writer import MarkdownWriter
        writer = MarkdownWriter()
        path = str(tmp_path / "test.md")
        writer.append({"a": 1}, path)
        with open(path) as f:
            content = f.read()
        assert "| a |" in content
        assert "| 1 |" in content

    def test_append_multiple_rows(self, tmp_path):
        from studies.output.formatters.markdown_writer import MarkdownWriter
        writer = MarkdownWriter()
        path = str(tmp_path / "test.md")
        writer.append({"a": 1}, path)
        writer.append({"a": 2}, path)
        with open(path) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 rows
