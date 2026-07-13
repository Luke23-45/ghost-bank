from __future__ import annotations

from enum import Enum, auto


class OutputState(Enum):
    INITIALIZED = auto()
    CONFIG_SAVED = auto()
    METRICS_OPEN = auto()
    RESULTS_WRITTEN = auto()
    ARTIFACTS_SAVED = auto()
    COMPLETED = auto()
    FAILED = auto()


class OutputStateMachine:
    _TRANSITIONS: dict[OutputState, set[OutputState]] = {
        OutputState.INITIALIZED: {OutputState.CONFIG_SAVED, OutputState.FAILED},
        OutputState.CONFIG_SAVED: {OutputState.METRICS_OPEN, OutputState.RESULTS_WRITTEN, OutputState.FAILED},
        OutputState.METRICS_OPEN: {OutputState.RESULTS_WRITTEN, OutputState.FAILED},
        OutputState.RESULTS_WRITTEN: {OutputState.ARTIFACTS_SAVED, OutputState.COMPLETED, OutputState.FAILED},
        OutputState.ARTIFACTS_SAVED: {OutputState.COMPLETED, OutputState.FAILED},
        OutputState.FAILED: {OutputState.COMPLETED},
        OutputState.COMPLETED: set(),
    }

    def __init__(self) -> None:
        self.state: OutputState = OutputState.INITIALIZED

    def transition(self, target: OutputState) -> None:
        allowed = self._TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise RuntimeError(
                f"Cannot transition from {self.state.name} to {target.name}. "
                f"Allowed targets: {[s.name for s in allowed]}"
            )
        self.state = target
