from __future__ import annotations

import json
import os

from studies.output.writer import BaseWriter


class JSONWriter(BaseWriter):
    def write(self, data: dict | list[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def append(self, data: dict, path: str) -> None:
        existing: list[dict] = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                existing = content if isinstance(content, list) else [content]
        existing.append(data)
        self.write(existing, path)
