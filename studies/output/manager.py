from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

from studies.output.state_machine import OutputState, OutputStateMachine
from studies.output.writer import FORMAT_REGISTRY


class OutputManager:
    def __init__(self, experiment: str, base_dir: str = "output") -> None:
        self.experiment = experiment
        self.base_dir = os.path.abspath(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root: str | None = None
        self._fsm = OutputStateMachine()
        self._writers: dict[str, object] = {}

    def initialize(self) -> str:
        if self._fsm.state != OutputState.INITIALIZED:
            raise RuntimeError(
                f"Cannot initialize: already in state {self._fsm.state.name}"
            )
        self.root = os.path.join(self.base_dir, self.experiment, self.timestamp)
        os.makedirs(os.path.join(self.root, "configs"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "artifacts"), exist_ok=True)
        self._fsm.transition(OutputState.CONFIG_SAVED)
        return self.root

    def save_config(self, config: dict | str, name: str = "resolved_config.yaml") -> None:
        self._require_state(OutputState.CONFIG_SAVED)
        path = os.path.join(self.root, "configs", name)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(config, str):
                f.write(config)
            elif isinstance(config, dict):
                json.dump(config, f, indent=2, default=str)
            else:
                f.write(str(config))

    def write_metrics(self, data: dict, filename: str = "train_metrics.csv") -> None:
        self._require_state(OutputState.CONFIG_SAVED, OutputState.METRICS_OPEN)
        path = os.path.join(self.root, "metrics", filename)
        writer = self._get_writer(filename)
        writer.append(data, path)
        if self._fsm.state == OutputState.CONFIG_SAVED:
            self._fsm.transition(OutputState.METRICS_OPEN)

    def finalize(self, results: dict, filename: str = "final_results.json") -> None:
        self._require_state(
            OutputState.CONFIG_SAVED,
            OutputState.METRICS_OPEN,
            OutputState.RESULTS_WRITTEN,
        )
        path = os.path.join(self.root, "results", filename)
        writer = self._get_writer(filename)
        writer.write(results, path)
        if self._fsm.state != OutputState.RESULTS_WRITTEN:
            self._fsm.transition(OutputState.RESULTS_WRITTEN)

    def save_artifact(self, name: str, data_or_path: object) -> None:
        self._require_state(OutputState.RESULTS_WRITTEN, OutputState.ARTIFACTS_SAVED)
        path = os.path.join(self.root, "artifacts", name)
        if isinstance(data_or_path, str) and os.path.exists(data_or_path):
            shutil.copy2(data_or_path, path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(data_or_path))
        if self._fsm.state != OutputState.ARTIFACTS_SAVED:
            self._fsm.transition(OutputState.ARTIFACTS_SAVED)

    def complete(self) -> None:
        self._require_state(OutputState.RESULTS_WRITTEN, OutputState.ARTIFACTS_SAVED, OutputState.FAILED)
        self._fsm.transition(OutputState.COMPLETED)

    def fail(self) -> None:
        if self._fsm.state not in (OutputState.COMPLETED, OutputState.FAILED):
            self._fsm.transition(OutputState.FAILED)

    def _require_state(self, *states: OutputState) -> None:
        if self._fsm.state not in states:
            raise RuntimeError(
                f"Invalid state {self._fsm.state.name} for this operation. "
                f"Expected one of: {[s.name for s in states]}"
            )

    def _get_writer(self, filename: str) -> object:
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "json"
        if ext not in self._writers:
            writer_cls = FORMAT_REGISTRY.get(ext)
            if writer_cls is None:
                raise ValueError(f"No registered writer for extension '.{ext}'")
            self._writers[ext] = writer_cls()
        return self._writers[ext]
