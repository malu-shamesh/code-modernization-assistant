from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from modernizer.protocols import ResultSink
from modernizer.sinks import JsonlSink
from tests.sinks.conftest import make_call, make_result


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_is_a_result_sink(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path)
    assert isinstance(sink, ResultSink)
    sink.close()


def test_creates_nested_directory_and_flushes_each_write(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    sink = JsonlSink(target)
    sink.write_module("run-1", make_result())
    sink.write_llm_call(make_call())
    # readable before close() because every write is flushed
    assert read_lines(target / "modules.jsonl")[0]["path"] == "pkg/a.py"
    assert read_lines(target / "llm_calls.jsonl")[0]["module"] == "pkg/m0.py"
    sink.close()


def test_source_excluded_by_default_and_included_on_request(tmp_path: Path) -> None:
    plain = JsonlSink(tmp_path / "plain")
    plain.write_module("r", make_result())
    plain.close()
    full = JsonlSink(tmp_path / "full", include_source=True)
    full.write_module("r", make_result())
    full.close()
    assert "converted_source" not in read_lines(tmp_path / "plain" / "modules.jsonl")[0]
    assert read_lines(tmp_path / "full" / "modules.jsonl")[0]["converted_source"] == "print('hi')\n"


def test_appends_across_instances_and_keeps_unicode(tmp_path: Path) -> None:
    for _ in range(2):
        sink = JsonlSink(tmp_path)
        sink.write_module("r", make_result("pkg/café.py"))
        sink.close()
    rows = read_lines(tmp_path / "modules.jsonl")
    assert [r["path"] for r in rows] == ["pkg/café.py"] * 2


def test_concurrent_writes_produce_valid_lines(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path, include_source=True)
    n_threads, per_thread = 8, 50
    barrier = threading.Barrier(n_threads)

    def work(t: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            sink.write_module(f"run-{t}", make_result(f"pkg/{t}_{i}.py"))
            sink.write_llm_call(make_call(t * per_thread + i))

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    sink.close()

    modules = read_lines(tmp_path / "modules.jsonl")
    calls = read_lines(tmp_path / "llm_calls.jsonl")
    assert len(modules) == len(calls) == n_threads * per_thread
    assert len({m["path"] for m in modules}) == n_threads * per_thread
    assert len({c["input_tokens"] for c in calls}) == n_threads * per_thread


def test_close_is_idempotent_and_write_after_close_raises(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path)
    sink.close()
    sink.close()
    with pytest.raises(RuntimeError):
        sink.write_module("r", make_result())
    with pytest.raises(RuntimeError):
        sink.write_llm_call(make_call())
