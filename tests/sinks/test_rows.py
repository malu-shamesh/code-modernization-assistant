from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from modernizer.models import TestRunResult
from modernizer.sinks import call_row, module_row
from tests.sinks.conftest import make_call, make_result


def test_module_row_values() -> None:
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    row = module_row("run-1", make_result(escalated=True), timestamp=ts)
    assert row["run_id"] == "run-1"
    assert row["timestamp"] == "2026-01-02T03:04:05+00:00"
    assert row["routed_tier"] == "open"
    assert row["final_tier"] == "frontier"
    assert row["escalated"] is True
    assert row["routing_policy"] == "complexity"
    assert row["routing_model"] == "open-model"
    assert row["final_model"] == "frontier-model"
    assert row["n_attempts"] == 2
    assert (row["input_tokens"], row["output_tokens"]) == (300, 130)
    assert (row["tests_total"], row["tests_passed"], row["tests_failed"]) == (4, 3, 1)
    assert row["tests_errors"] == 0
    assert row["tests_timed_out"] is False
    assert row["validation_issue_codes"] == ["legacy_construct_remaining", "api_changed"]
    assert row["error"] is None
    json.dumps(row, allow_nan=False)


def test_module_row_default_timestamp_is_utc_iso() -> None:
    parsed = datetime.fromisoformat(module_row("r", make_result())["timestamp"])
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_source_only_with_flag() -> None:
    result = make_result()
    assert "converted_source" not in module_row("r", result)
    with_source = module_row("r", result, include_source=True)
    assert with_source["converted_source"] == result.converted_source
    assert "generated_tests" not in with_source


def test_module_row_without_attempts_or_tests() -> None:
    result = make_result(with_test=False)
    result.attempts = []
    result.usage_by_model = {}
    result.error = "boom"
    row = module_row("r", result)
    assert row["final_model"] is None
    assert row["n_attempts"] == 0
    assert row["validation_issue_codes"] == []
    assert row["tests_total"] is None and row["tests_timed_out"] is None
    assert (row["input_tokens"], row["output_tokens"]) == (0, 0)
    assert row["error"] == "boom"


def test_module_row_falls_back_to_last_attempt_test() -> None:
    result = make_result()
    result.test = None
    assert module_row("r", result)["tests_total"] == 4


def test_module_row_does_not_borrow_tests_from_earlier_attempts() -> None:
    result = make_result()
    result.test = None
    result.attempts[-1].test = None
    result.attempts[0].test = TestRunResult(passed=1, total=1)
    assert module_row("r", result)["tests_total"] is None


def test_lone_surrogates_are_replaced_so_rows_stay_encodable() -> None:
    result = make_result("pkg/bad\udcff.py")
    result.error = "oops \ud800"
    record = make_call()
    record.module = "m\udcff"
    rows = [module_row("r", result, include_source=True), call_row(record)]
    for row in rows:
        json.dumps(row, ensure_ascii=False).encode("utf-8")
    assert rows[0]["path"].startswith("pkg/bad")
    assert rows[1]["module"].startswith("m")


def test_non_finite_floats_become_null() -> None:
    result = make_result()
    result.cost_usd = math.inf
    result.routing.difficulty = math.nan
    row = module_row("r", result)
    assert row["cost_usd"] is None and row["difficulty"] is None
    json.dumps(row, allow_nan=False)


def test_call_row() -> None:
    row = call_row(make_call(2))
    assert row["module"] == "pkg/m2.py"
    assert row["input_tokens"] == 12
    assert row["tier"] == "open"
    json.dumps(row, allow_nan=False)
