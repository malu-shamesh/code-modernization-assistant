from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ..models import LLMCallRecord, ModuleResult
from .rows import call_row, module_row

MODULES_FILE = "modules.jsonl"
CALLS_FILE = "llm_calls.jsonl"


class JsonlSink:
    """Appends one JSON object per line to ``modules.jsonl`` and ``llm_calls.jsonl``.

    Safe to share between threads; every record is written and flushed under a lock so lines
    never interleave. Existing files are appended to, so a directory can hold several runs.
    """

    def __init__(self, directory: str | os.PathLike[str], *, include_source: bool = False) -> None:
        self.directory = Path(directory)
        self.include_source = include_source
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._modules = open(self.directory / MODULES_FILE, "a", encoding="utf-8", newline="\n")
        self._calls = open(self.directory / CALLS_FILE, "a", encoding="utf-8", newline="\n")

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        row = module_row(run_id, result, include_source=self.include_source)
        self._append(self._modules, row)

    def write_llm_call(self, record: LLMCallRecord) -> None:
        self._append(self._calls, call_row(record))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._modules.close()
            self._calls.close()

    def _append(self, handle: Any, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("JsonlSink is closed")
            handle.write(line)
            handle.flush()
