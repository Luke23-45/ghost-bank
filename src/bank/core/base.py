from __future__ import annotations

from abc import ABC, abstractmethod

import torch


def _to_int(value: int | torch.Tensor) -> int:
    return value.item() if isinstance(value, torch.Tensor) else int(value)


class AbstractGhostBank(ABC):
    """Interface for ghost bank implementations.

    Subclasses manage per-class memory pools and retrieval strategies.
    """

    @abstractmethod
    def store(self, examples: list) -> None:
        """Store a list of (features, label) examples in the bank."""
        ...

    @abstractmethod
    def query(self, budget: int, **kwargs) -> list:
        """Retrieve up to ``budget`` examples from the bank."""
        ...

    def freeze(self) -> None:
        """Prevent further storage.  Query still works."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable storage after a freeze."""
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return getattr(self, "_frozen", False)

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore serialized state from a checkpoint."""
        ...
