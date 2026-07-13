from __future__ import annotations

import csv
import os

from studies.output.writer import BaseWriter


class CSVWriter(BaseWriter):
    def __init__(self) -> None:
        self._known_fields: set[str] = set()

    def write(self, data: dict | list[dict], path: str) -> None:
        rows = data if isinstance(data, list) else [data]
        if not rows:
            return
        self._known_fields.update(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._known_fields))
            writer.writeheader()
            writer.writerows(rows)

    def append(self, data: dict, path: str) -> None:
        self._known_fields.update(data.keys())
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._known_fields))
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
