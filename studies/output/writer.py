from __future__ import annotations

from abc import ABC, abstractmethod


class BaseWriter(ABC):
    @abstractmethod
    def write(self, data: dict | list[dict], path: str) -> None:
        ...

    @abstractmethod
    def append(self, data: dict, path: str) -> None:
        ...


FORMAT_REGISTRY: dict[str, type[BaseWriter]] = {}


def register_format(ext: str, writer_cls: type[BaseWriter]) -> None:
    FORMAT_REGISTRY[ext] = writer_cls
