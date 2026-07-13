from __future__ import annotations

import os

from studies.output.writer import BaseWriter


class MarkdownWriter(BaseWriter):
    def write(self, data: dict | list[dict], path: str) -> None:
        rows = data if isinstance(data, list) else [data]
        if not rows:
            return
        headers = list(rows[0].keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join("---" for _ in headers) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")

    def append(self, data: dict, path: str) -> None:
        rows = [data]
        headers = list(data.keys())
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join("---" for _ in headers) + "|\n")
            f.write("| " + " | ".join(str(data.get(h, "")) for h in headers) + " |\n")
