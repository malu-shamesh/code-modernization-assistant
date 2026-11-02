from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from modernizer.protocols import ResultSink
from modernizer.sinks import BigQuerySink
from tests.sinks.conftest import make_call, make_result

MODULES = "proj.ds.module_results"
CALLS = "proj.ds.llm_calls"


class FakeClient:
    def __init__(self, errors: list[dict] | None = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, list[dict], dict[str, Any]]] = []
        self.errors = errors or []
        self.raises = raises
        self.closed = False
        self._lock = threading.Lock()

    def insert_rows_json(self, table: str, json_rows: list[dict], **kwargs: Any) -> list[dict]:
        with self._lock:
            self.calls.append((table, list(json_rows), kwargs))
        if self.raises:
            raise self.raises
        return self.errors

    def close(self) -> None:
        self.closed = True


def test_is_a_result_sink() -> None:
    assert isinstance(BigQuerySink("proj", "ds", client=FakeClient()), ResultSink)


def test_batches_at_batch_size_per_table() -> None:
    client = FakeClient()
    sink = BigQuerySink("proj", "ds", client=client, batch_size=3)
    for i in range(2):
        sink.write_llm_call(make_call(i))
    sink.write_module("r", make_result())
    assert client.calls == []
    sink.write_llm_call(make_call(2))
    assert [(t, len(r)) for t, r, _ in client.calls] == [(CALLS, 3)]
    assert client.calls[0][1][2]["input_tokens"] == 12


def test_flush_on_close_sends_remaining_rows_to_both_tables() -> None:
    client = FakeClient()
    sink = BigQuerySink("proj", "ds", client=client, batch_size=100)
    sink.write_module("r", make_result())
    sink.write_llm_call(make_call())
    sink.close()
    assert sorted((t, len(r)) for t, r, _ in client.calls) == [(CALLS, 1), (MODULES, 1)]
    assert sink.failed_rows == []


def test_custom_table_names_and_empty_close_makes_no_calls() -> None:
    client = FakeClient()
    sink = BigQuerySink("p", "d", modules_table="m", calls_table="c", client=client, batch_size=1)
    sink.write_module("r", make_result())
    sink.write_llm_call(make_call())
    assert [t for t, _, _ in client.calls] == ["p.d.m", "p.d.c"]
    client.calls.clear()
    sink.close()
    assert client.calls == []


def test_skip_invalid_rows_requested() -> None:
    client = FakeClient()
    sink = BigQuerySink("proj", "ds", client=client, batch_size=1)
    sink.write_llm_call(make_call())
    assert client.calls[0][2] == {"skip_invalid_rows": True}


def test_row_errors_are_surfaced_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    errors = [{"index": 1, "errors": [{"reason": "invalid", "message": "bad field"}]}]
    client = FakeClient(errors=errors)
    sink = BigQuerySink("proj", "ds", client=client, batch_size=2)
    with caplog.at_level(logging.ERROR, logger="modernizer.sinks.bigquery"):
        sink.write_llm_call(make_call(0))
        sink.write_llm_call(make_call(1))
    assert len(sink.failed_rows) == 1
    failed = sink.failed_rows[0]
    assert failed["table"] == CALLS
    assert failed["row"]["input_tokens"] == 11
    assert failed["errors"][0]["message"] == "bad field"
    assert all(r.levelno == logging.ERROR for r in caplog.records)
    assert "rejected 1 of 2" in caplog.text
    assert "bad field" in caplog.text


def test_out_of_range_error_index_is_tolerated() -> None:
    client = FakeClient(errors=[{"index": 99, "errors": [{"message": "x"}]}])
    sink = BigQuerySink("proj", "ds", client=client, batch_size=1)
    sink.write_llm_call(make_call())
    assert sink.failed_rows[0]["row"] is None


def test_flush_exception_does_not_crash_and_rows_are_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient(raises=ConnectionError("network down"))
    sink = BigQuerySink("proj", "ds", client=client, batch_size=2)
    with caplog.at_level(logging.ERROR, logger="modernizer.sinks.bigquery"):
        sink.write_llm_call(make_call(0))
        sink.write_llm_call(make_call(1))
        sink.write_module("r", make_result())
        sink.close()
    assert len(sink.failed_rows) == 3
    assert "network down" in sink.failed_rows[0]["errors"][0]["message"]
    assert {f["table"] for f in sink.failed_rows} == {CALLS, MODULES}
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_close_is_idempotent_and_write_after_close_raises() -> None:
    client = FakeClient()
    sink = BigQuerySink("proj", "ds", client=client)
    sink.write_llm_call(make_call())
    sink.close()
    sink.close()
    assert len(client.calls) == 1
    with pytest.raises(RuntimeError):
        sink.write_llm_call(make_call())


def test_injected_client_is_not_closed() -> None:
    client = FakeClient()
    BigQuerySink("proj", "ds", client=client).close()
    assert client.closed is False


def test_include_source_flag() -> None:
    client = FakeClient()
    sink = BigQuerySink("proj", "ds", client=client, batch_size=1, include_source=True)
    sink.write_module("r", make_result())
    assert client.calls[0][1][0]["converted_source"] == "print('hi')\n"


def test_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        BigQuerySink("proj", "ds", client=FakeClient(), batch_size=0)


def test_concurrent_writes_lose_no_rows() -> None:
    client = FakeClient()
    sink = BigQuerySink("proj", "ds", client=client, batch_size=7)
    n_threads, per_thread = 8, 40

    def work(t: int) -> None:
        for i in range(per_thread):
            sink.write_llm_call(make_call(t * per_thread + i))

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    sink.close()
    sent = [row["input_tokens"] for _, rows, _ in client.calls for row in rows]
    assert sorted(sent) == sorted(10 + i for i in range(n_threads * per_thread))
    assert all(len(rows) <= 7 for _, rows, _ in client.calls)
