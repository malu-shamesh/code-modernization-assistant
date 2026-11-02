from __future__ import annotations

import json
from pathlib import Path

import pytest

from modernizer.sinks import call_row, module_row
from tests.sinks.conftest import make_call, make_result

DEPLOY = Path(__file__).resolve().parents[2] / "deploy" / "bigquery"

PY_TYPES = {
    "STRING": str,
    "TIMESTAMP": str,
    "BOOLEAN": bool,
    "INTEGER": int,
    "FLOAT": float,
}


def load_schema(name: str) -> dict[str, dict]:
    fields = json.loads((DEPLOY / name).read_text(encoding="utf-8"))
    return {f["name"]: f for f in fields}


def assert_value_fits(field: dict, value: object) -> None:
    name, mode, bq_type = field["name"], field["mode"], field["type"]
    if mode == "REPEATED":
        assert isinstance(value, list), name
        assert all(isinstance(v, PY_TYPES[bq_type]) for v in value), name
        return
    if value is None:
        assert mode == "NULLABLE", f"{name} is None but mode is {mode}"
        return
    expected = PY_TYPES[bq_type]
    if expected is float:
        assert isinstance(value, (int, float)) and not isinstance(value, bool), name
    elif expected is int:
        assert isinstance(value, int) and not isinstance(value, bool), name
    else:
        assert isinstance(value, expected), name


def check(schema: dict[str, dict], row: dict, optional: set[str]) -> None:
    """Row keys plus the optional (absent) columns must equal the schema's column names."""
    assert set(row) | optional == set(schema)
    assert not (set(row) & optional)
    for name, value in row.items():
        assert_value_fits(schema[name], value)
    for name in optional:
        assert schema[name]["mode"] == "NULLABLE"


def test_module_results_schema_matches_row() -> None:
    schema = load_schema("schema_module_results.json")
    check(schema, module_row("run-1", make_result()), {"converted_source"})
    check(schema, module_row("run-1", make_result(), include_source=True), set())


def test_module_results_schema_matches_row_without_tests_or_attempts() -> None:
    schema = load_schema("schema_module_results.json")
    result = make_result(with_test=False)
    result.attempts = []
    check(schema, module_row("run-1", result), {"converted_source"})


def test_llm_calls_schema_matches_row() -> None:
    check(load_schema("schema_llm_calls.json"), call_row(make_call()), set())


@pytest.mark.parametrize("name", ["schema_module_results.json", "schema_llm_calls.json"])
def test_schema_fields_are_well_formed(name: str) -> None:
    for field in load_schema(name).values():
        assert field["type"] in PY_TYPES
        assert field["mode"] in {"REQUIRED", "NULLABLE", "REPEATED"}


def test_timestamp_is_a_required_field_for_partitioning() -> None:
    for name in ("schema_module_results.json", "schema_llm_calls.json"):
        field = load_schema(name)["timestamp"]
        assert (field["type"], field["mode"]) == ("TIMESTAMP", "REQUIRED")


def test_views_use_placeholders_and_cover_required_views() -> None:
    sql = (DEPLOY / "views.sql").read_text(encoding="utf-8")
    for view in (
        "v_cost_by_tier",
        "v_pass_rate_by_tier",
        "v_routing_summary",
        "v_cost_per_passing_module",
    ):
        assert f"CREATE OR REPLACE VIEW `${{PROJECT}}.${{DATASET}}.{view}`" in sql


def test_view_columns_exist_in_schemas() -> None:
    modules = set(load_schema("schema_module_results.json"))
    calls = set(load_schema("schema_llm_calls.json"))
    sql = (DEPLOY / "views.sql").read_text(encoding="utf-8")
    llm_part, module_part = sql.split("v_pass_rate_by_tier", 1)
    for col in ("model", "tier", "input_tokens", "output_tokens", "cost_usd", "timestamp"):
        assert col in calls and col in llm_part
    for col in ("run_id", "routing_policy", "routed_tier", "final_tier", "success", "escalated"):
        assert col in modules and col in module_part
