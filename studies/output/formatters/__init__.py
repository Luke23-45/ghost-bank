from __future__ import annotations

from studies.output.formatters.csv_writer import CSVWriter
from studies.output.formatters.jsonl_writer import JSONLWriter
from studies.output.formatters.json_writer import JSONWriter
from studies.output.formatters.markdown_writer import MarkdownWriter
from studies.output.writer import register_format

register_format("csv", CSVWriter)
register_format("jsonl", JSONLWriter)
register_format("json", JSONWriter)
register_format("md", MarkdownWriter)

__all__ = [
    "CSVWriter",
    "JSONLWriter",
    "JSONWriter",
    "MarkdownWriter",
]
