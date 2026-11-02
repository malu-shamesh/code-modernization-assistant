from __future__ import annotations

from pathlib import Path

import pytest

from modernizer.sinks import (
    BigQuerySink,
    CompositeSink,
    JsonlSink,
    NullSink,
    sink_from_spec,
)
from modernizer.sinks import bigquery as bq_module
from tests.sinks.conftest import make_call, make_result


class StubClient:
    def __init__(self, project: str | None = None, location: str | None = None) -> None:
        self.rows = 0

    def insert_rows_json(self, table: str, rows: list[dict], **kwargs: object) -> list[dict]:
        self.rows += len(rows)
        return []

    def close(self) -> None:
        return None


class StubBigQueryModule:
    Client = StubClient


@pytest.fixture(autouse=True)
def stub_bigquery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bq_module, "bigquery", StubBigQueryModule)


@pytest.mark.parametrize("spec", [None, "none", "NONE", "", " , ", "none,none"])
def test_null_specs(spec: str | None) -> None:
    assert isinstance(sink_from_spec(spec), NullSink)


def test_jsonl_spec(tmp_path: Path) -> None:
    sink = sink_from_spec(f"jsonl:{tmp_path / 'out'}")
    assert isinstance(sink, JsonlSink)
    assert sink.directory == tmp_path / "out"
    sink.close()


def test_jsonl_defaults_to_run_dir(tmp_path: Path) -> None:
    sink = sink_from_spec("jsonl", run_dir=str(tmp_path / "run"))
    assert isinstance(sink, JsonlSink)
    assert sink.directory == tmp_path / "run"
    sink.close()
    with pytest.raises(ValueError):
        sink_from_spec("jsonl")


def test_bigquery_spec() -> None:
    sink = sink_from_spec("bigquery:my-proj.my_ds")
    assert isinstance(sink, BigQuerySink)
    assert sink.modules_table_id == "my-proj.my_ds.module_results"


def test_bigquery_spec_with_domain_scoped_project() -> None:
    sink = sink_from_spec("bigquery:example.com:proj.ds")
    assert isinstance(sink, BigQuerySink)
    assert sink.calls_table_id == "example.com:proj.ds.llm_calls"


@pytest.mark.parametrize(
    "spec", ["bigquery:", "bigquery:onlyproject", "bigquery:.ds", "bigquery:p."]
)
def test_bad_bigquery_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        sink_from_spec(spec)


@pytest.mark.parametrize("spec", ["s3:bucket", "file.jsonl", "jsonl-x:foo", "none:x"])
def test_unknown_scheme(spec: str) -> None:
    with pytest.raises(ValueError, match="unknown sink spec"):
        sink_from_spec(spec)


def test_combination_builds_composite_and_fans_out(tmp_path: Path) -> None:
    sink = sink_from_spec(f"jsonl:{tmp_path}, bigquery:p.d , none")
    assert isinstance(sink, CompositeSink)
    jsonl, bigquery = sink.sinks
    assert isinstance(jsonl, JsonlSink) and isinstance(bigquery, BigQuerySink)
    sink.write_module("r", make_result())
    sink.write_llm_call(make_call())
    sink.close()
    assert (tmp_path / "modules.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert bigquery._client.rows == 2


def test_failed_combination_closes_already_built_sinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sink_from_spec(f"jsonl:{tmp_path},bogus:x")
    # open handles would make this fail on Windows
    (tmp_path / "modules.jsonl").unlink()
    (tmp_path / "llm_calls.jsonl").unlink()


def test_composite_isolates_failing_sink(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def write_module(self, run_id: str, result: object) -> None:
            raise OSError("disk full")

        def write_llm_call(self, record: object) -> None:
            raise OSError("disk full")

        def close(self) -> None:
            raise OSError("disk full")

    good = JsonlSink(tmp_path)
    composite = CompositeSink(Broken(), good)  # type: ignore[arg-type]
    composite.write_module("r", make_result())
    composite.write_llm_call(make_call())
    composite.close()
    assert (tmp_path / "modules.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert "disk full" in caplog.text


def test_null_sink_accepts_everything() -> None:
    sink = NullSink()
    sink.write_module("r", make_result())
    sink.write_llm_call(make_call())
    sink.close()
