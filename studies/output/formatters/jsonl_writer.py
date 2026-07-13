from __future__ import annotations

import json

from studies.output.writer import BaseWriter


class JSONLWriter(BaseWriter):
    def write(self, data: dict | list[dict], path: str) -> None:
        rows = data if isinstance(data, list) else [data]
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

    def append(self, data: dict, path: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
