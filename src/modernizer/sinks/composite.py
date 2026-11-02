from __future__ import annotations

import logging

from ..models import LLMCallRecord, ModuleResult
from ..protocols import ResultSink

log = logging.getLogger(__name__)


class NullSink:
    """Discards everything."""

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        return None

    def write_llm_call(self, record: LLMCallRecord) -> None:
        return None

    def close(self) -> None:
        return None


class CompositeSink:
    """Fans every record out to several sinks.

    A failing sink is logged and skipped so it cannot starve the others or abort the run.
    """

    def __init__(self, *sinks: ResultSink) -> None:
        self.sinks: tuple[ResultSink, ...] = sinks

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        for sink in self.sinks:
            try:
                sink.write_module(run_id, result)
            except Exception:
                log.exception("%s.write_module failed", type(sink).__name__)

    def write_llm_call(self, record: LLMCallRecord) -> None:
        for sink in self.sinks:
            try:
                sink.write_llm_call(record)
            except Exception:
                log.exception("%s.write_llm_call failed", type(sink).__name__)

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                log.exception("%s.close failed", type(sink).__name__)
