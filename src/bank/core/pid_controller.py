from __future__ import annotations

import math


class PIDController:
    """PID feedback controller for per-class replay debt.

    Computes debt per class as:
        debt_c(t) = max(0, weight_c * (K_p * L_c(t) + K_i * I_c(t) + K_d * D_c(t)))

    where:
        L_c(t) — smoothed per-class loss (proportional term)
        I_c(t) — EMA of smoothed loss (integral term)
        D_c(t) — first difference of smoothed loss (derivative term)
        weight_c — per-class multiplier (higher → more allocation to that class)

    The ``class_weights`` argument accepts a list of per-class multipliers.
    When set to e.g. ``[1.0 / sqrt(freq_c / max_freq)]``, minority classes
    receive amplified debt, compensating for their under-representation in
    the loss signal.

    Only updates internal state for classes present in the current batch.
    Absent classes keep their previous state.
    """

    def __init__(
        self,
        num_classes: int,
        K_p: float = 1.0,
        K_i: float = 0.1,
        K_d: float = 0.5,
        decay: float = 0.99,
        smooth: float = 0.9,
        temperature: float = 1.0,
        class_weights: list[float] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.decay = decay
        self.smooth = smooth
        self.temperature = temperature
        self.class_weights = class_weights or [1.0] * num_classes

        self._integral: list[float] = [0.0] * num_classes
        self._prev_loss: list[float] = [0.0] * num_classes
        self._smoothed_loss: list[float] = [0.0] * num_classes
        self._last_debt: list[float] = [0.0] * num_classes

    def update(self, per_class_loss: list[float | None]) -> list[float]:
        """Ingest per-class loss, update PID state, return per-class debt.

        Entries in ``per_class_loss`` may be ``None`` to indicate the
        class was absent from the batch — its internal state is left
        unchanged.
        """
        raw_debt = []
        for c in range(self.num_classes):
            L = per_class_loss[c]

            if L is not None:
                self._smoothed_loss[c] = (
                    self.smooth * self._smoothed_loss[c] + (1.0 - self.smooth) * L
                )
                self._integral[c] = (
                    self.decay * self._integral[c] + (1.0 - self.decay) * self._smoothed_loss[c]
                )

            p = self.K_p * self._smoothed_loss[c]
            i = self.K_i * self._integral[c]
            d = self.K_d * (self._smoothed_loss[c] - self._prev_loss[c])

            val = self.class_weights[c] * (p + i + d)
            raw_debt.append(min(10000.0, max(0.0, val)))

            if L is not None:
                self._prev_loss[c] = self._smoothed_loss[c]

        self._last_debt = raw_debt
        return raw_debt

    @property
    def last_debt(self) -> list[float]:
        return list(self._last_debt)

    def reset(self) -> None:
        self._integral = [0.0] * self.num_classes
        self._prev_loss = [0.0] * self.num_classes
        self._smoothed_loss = [0.0] * self.num_classes
        self._last_debt = [0.0] * self.num_classes

    def state_dict(self) -> dict:
        return {
            "integral": list(self._integral),
            "prev_loss": list(self._prev_loss),
            "smoothed_loss": list(self._smoothed_loss),
        }

    def load_state_dict(self, state: dict) -> None:
        self._integral = list(state["integral"])
        self._prev_loss = list(state["prev_loss"])
        self._smoothed_loss = list(state["smoothed_loss"])
