from __future__ import annotations

import logging
import threading
from typing import Any

from ..models import LLMCallRecord, ModuleResult
from .rows import call_row, module_row

try:
    from google.cloud import bigquery
except ImportError:  # optional dependency, see the "gcp" extra
    bigquery = None

log = logging.getLogger(__name__)

_MAX_LOGGED_ERRORS = 3


class BigQuerySink:
    """Streams module results and LLM call records into BigQuery with ``insert_rows_json``.

    Rows are buffered per table and sent in batches of ``batch_size`` and on ``close()``. The
    sink is safe to share between threads. Failures never propagate out of ``write_*`` or
    ``close``: rejected rows, and whole batches lost to transport errors, are logged at ERROR
    and kept in ``failed_rows`` as ``{"table", "row", "errors"}`` dicts.
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        *,
        modules_table: str = "module_results",
        calls_table: str = "llm_calls",
        client: object | None = None,
        batch_size: int = 50,
        location: str | None = None,
        include_source: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._owns_client = client is None
        if client is None:
            if bigquery is None:
                raise ImportError(
                    "BigQuerySink needs google-cloud-bigquery: "
                    "pip install 'code-modernization-assistant[gcp]'"
                )
            client = bigquery.Client(project=project, location=location)
        self._client: Any = client
        self.modules_table_id = f"{project}.{dataset}.{modules_table}"
        self.calls_table_id = f"{project}.{dataset}.{calls_table}"
        self.batch_size = batch_size
        self.include_source = include_source
        self.failed_rows: list[dict[str, Any]] = []
        self._buffers: dict[str, list[dict[str, Any]]] = {
            self.modules_table_id: [],
            self.calls_table_id: [],
        }
        self._lock = threading.Lock()  # guards buffers, failed_rows and the closed flag
        self._send_lock = threading.Lock()  # serialises API calls so close() waits for them
        self._closed = False

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        row = module_row(run_id, result, include_source=self.include_source)
        self._enqueue(self.modules_table_id, row)

    def write_llm_call(self, record: LLMCallRecord) -> None:
        self._enqueue(self.calls_table_id, call_row(record))

    def flush(self) -> None:
        for table_id in self._buffers:
            self._flush_table(table_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.flush()
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                log.exception("closing the BigQuery client failed")

    def _enqueue(self, table_id: str, row: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("BigQuerySink is closed")
            buffer = self._buffers[table_id]
            buffer.append(row)
            full = len(buffer) >= self.batch_size
        if full:
            self._flush_table(table_id)

    def _flush_table(self, table_id: str) -> None:
        with self._send_lock:
            with self._lock:
                rows = self._buffers[table_id]
                self._buffers[table_id] = []
            if rows:
                self._send(table_id, rows)

    def _send(self, table_id: str, rows: list[dict[str, Any]]) -> None:
        failures: list[dict[str, Any]] = []
        try:
            # skip_invalid_rows keeps one bad row from rejecting the whole request.
            errors = self._client.insert_rows_json(table_id, rows, skip_invalid_rows=True)
        except Exception as exc:
            log.exception(
                "BigQuery insert into %s failed; keeping %d rows in failed_rows",
                table_id,
                len(rows),
            )
            failures = [
                {"table": table_id, "row": row, "errors": [{"message": repr(exc)}]} for row in rows
            ]
        else:
            for err in errors or ():
                index = err.get("index")
                row = rows[index] if isinstance(index, int) and 0 <= index < len(rows) else None
                failures.append({"table": table_id, "row": row, "errors": err.get("errors", [])})
            if failures:
                log.error(
                    "BigQuery rejected %d of %d rows for %s; first errors: %s",
                    len(failures),
                    len(rows),
                    table_id,
                    [f["errors"] for f in failures[:_MAX_LOGGED_ERRORS]],
                )
        if failures:
            with self._lock:
                self.failed_rows.extend(failures)
