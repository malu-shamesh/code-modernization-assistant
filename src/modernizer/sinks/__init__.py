"""Result persistence: JSONL files, BigQuery, fan-out and a no-op sink."""

from __future__ import annotations

from .bigquery import BigQuerySink
from .composite import CompositeSink, NullSink
from .factory import sink_from_spec
from .jsonl import JsonlSink
from .rows import call_row, module_row

__all__ = [
    "BigQuerySink",
    "CompositeSink",
    "JsonlSink",
    "NullSink",
    "call_row",
    "module_row",
    "sink_from_spec",
]
