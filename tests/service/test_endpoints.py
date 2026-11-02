from __future__ import annotations

import logging
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from modernizer.llm import LLMError, ModelConfig
from modernizer.pipeline import Pipeline, PipelineConfig
from modernizer.routing import DifficultyWeights
from modernizer.service import app as app_module
from modernizer.service import create_app
from modernizer.settings import Settings

from .fakes import (
    ACCUM_LEGACY,
    EQUIV_LEGACY,
    GCD_LEGACY,
    GOOD,
    HALF_GOOD,
    HALF_LEGACY,
    HALF_TEST_COUNT,
    HALF_TESTS,
    LEGACY,
    TEST_COUNT,
    Harness,
    port_for,
)

SMALL_SET = [
    ("sumsq.f", LEGACY),
    ("half.f", HALF_LEGACY),
    ("accum.f", ACCUM_LEGACY),
    ("gcd.f", GCD_LEGACY),
]


def body(*files: tuple[str, str], **options: object) -> dict[str, object]:
    return {"files": [{"path": p, "source": s} for p, s in files], **options}


def test_healthz_is_always_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_non_secret_config(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    defaults = Settings()
    assert data["status"] == "ready"
    assert data["provider"] == defaults.models.provider
    assert data["frontier_model"] == defaults.models.frontier_model
    assert data["open_model"] == defaults.models.open_model
    assert data["open_backend"] == defaults.models.open_backend == "endpoint"
    assert data["retrieval"] == defaults.retrieval
    assert data["policy"] == defaults.policy
    assert data["threshold"] == defaults.threshold
    assert data["open_fraction"] is None


def test_readyz_names_the_backend_but_never_ids_urls_or_projects(
    make_client: Callable[..., TestClient],
) -> None:
    secrets = ("1234567890123", "https://gemma.internal.example/v1", "my-secret-project")

    def readyz(**models: object) -> tuple[dict, str]:
        settings = Settings(
            models=ModelConfig(project="my-secret-project", **models),  # type: ignore[arg-type]
            pipeline=PipelineConfig(target_open_fraction=0.7),
        )
        response = make_client(settings).get("/readyz")
        assert response.status_code == 200
        return response.json(), response.text

    data, text = readyz(open_backend="endpoint", open_endpoint_id=secrets[0])
    assert data["open_backend"] == "endpoint" and data["open_fraction"] == 0.7
    data, text_maas = readyz(open_backend="maas")
    assert data["open_backend"] == "maas"
    data, text_url = readyz(open_backend="endpoint", open_base_url=secrets[1])
    assert data["open_backend"] == "openai", "a base URL turns the open tier into an openai server"
    for leaked in (text, text_maas, text_url):
        for secret in secrets:
            assert secret not in leaked


def test_not_ready_before_startup_finished(harness: Harness) -> None:
    client = TestClient(create_app(Settings(), pipeline_factory=harness))  # no lifespan
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
    assert client.post("/v1/analyze", json=body(("a.f", LEGACY))).status_code == 503


def test_modernize_runs_generated_tests_in_the_real_sandbox(
    client: TestClient, harness: Harness
) -> None:
    response = client.post("/v1/modernize", json=body(("pkg/sumsq.f", LEGACY), include_tests=True))
    assert response.status_code == 200
    data = response.json()

    assert data["run_id"] == harness.pipelines[0].run_id
    (result,) = data["results"]
    assert result["path"] == "pkg/sumsq.f"
    assert result["success"] is True
    assert result["converted_source"] == GOOD
    assert result["routed_tier"] == result["final_tier"] == "open"
    assert result["escalated"] is False
    assert result["attempts"] == 1
    assert result["validation_issues"] == []
    assert result["error"] is None
    assert result["tests"] == {"passed": TEST_COUNT, "failed": 0, "errors": 0, "total": TEST_COUNT}
    assert result["input_tokens"] > 0 and result["output_tokens"] > 0
    assert 0.0 <= result["difficulty"] <= 1.0
    assert "def test_isumsq" in result["generated_tests"]

    summary = data["summary"]
    assert summary["modules"] == 1 and summary["succeeded"] == 1
    assert summary["input_tokens"] == result["input_tokens"]
    assert summary["total_tokens"] == summary["input_tokens"] + summary["output_tokens"]
    assert harness.sink.run_ids == [data["run_id"]]


def test_the_sandbox_provides_the_fortran_runtime(client: TestClient, harness: Harness) -> None:
    """Converted code imports ``f77rt``; the image ships it and the sandbox must offer it."""
    harness.open_script = [HALF_GOOD]
    harness.tests = HALF_TESTS
    response = client.post("/v1/modernize", json=body(("half.f", HALF_LEGACY)))
    (result,) = response.json()["results"]
    assert result["success"] is True, result
    assert result["tests"]["passed"] == HALF_TEST_COUNT
    assert result["tests"]["failed"] == result["tests"]["errors"] == 0


def test_tests_are_skipped_and_not_returned_unless_asked(client: TestClient) -> None:
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    (result,) = response.json()["results"]
    assert result["success"] is True
    assert result["tests"] is None
    assert "generated_tests" not in result


@pytest.mark.parametrize("path", ["SUMSQ.F", "legacy/sumsq.for", "a/b.F77", "mod.Ftn"])
def test_fortran_extensions_are_accepted_in_any_case(client: TestClient, path: str) -> None:
    response = client.post("/v1/modernize", json=body((path, LEGACY), run_tests=False))
    assert response.status_code == 200
    assert response.json()["results"][0]["path"] == path


def test_escalation_is_visible_in_the_response(client: TestClient, harness: Harness) -> None:
    harness.open_script = [LEGACY, LEGACY, LEGACY]
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    data = response.json()
    (result,) = data["results"]
    assert result["success"] is True
    assert result["escalated"] is True
    assert result["routed_tier"] == "open"
    assert result["final_tier"] == "frontier"
    assert result["attempts"] == 4
    assert data["summary"]["escalated"] == 1
    assert data["summary"]["final_open"] == 0


def test_failed_conversion_reports_validation_issues(client: TestClient, harness: Harness) -> None:
    harness.open_script = harness.frontier_script = [LEGACY] * 3  # Fortran is not Python
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    (result,) = response.json()["results"]
    assert result["success"] is False
    assert result["error"].startswith("validation failed")
    assert result["validation_issues"]
    issue = result["validation_issues"][0]
    assert set(issue) == {"code", "message", "line"}
    assert issue["code"] == "syntax_error"


def test_a_model_that_breaks_the_calling_convention_is_reported(
    client: TestClient, harness: Harness
) -> None:
    harness.open_script = harness.frontier_script = ["def other(n):\n    return n\n"] * 3
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    (result,) = response.json()["results"]
    assert result["success"] is False
    assert {i["code"] for i in result["validation_issues"]} == {"api_changed"}


def test_model_failures_give_a_generic_error_without_the_exception_text(
    client: TestClient, harness: Harness
) -> None:
    harness.open_error = LLMError("quota exceeded for project my-secret-project\nINFO forged line")
    harness.frontier_error = LLMError("backend 10.0.0.7 refused")
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["success"] is False
    assert result["error"] == "model call failed on the frontier tier (LLMError)"
    for leaked in ("my-secret-project", "forged", "10.0.0.7"):
        assert leaked not in response.text


def test_a_crash_inside_one_module_is_reported_generically(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("password=hunter2\nWARNING forged log line")

    monkeypatch.setattr(Pipeline, "_convert", explode)
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["success"] is False
    assert result["error"] == "unexpected error (RuntimeError), see the server log"
    assert "hunter2" not in response.text and "forged" not in response.text


def test_request_overrides_apply_to_that_request_only(client: TestClient, harness: Harness) -> None:
    harness.frontier_script = [GOOD]
    first = client.post(
        "/v1/modernize",
        json=body(("sumsq.f", LEGACY), policy="always_frontier", run_tests=False),
    )
    assert first.json()["results"][0]["routed_tier"] == "frontier"

    second = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    assert second.json()["results"][0]["routed_tier"] == "open"
    assert first.json()["run_id"] != second.json()["run_id"]


def test_threshold_override_changes_the_complexity_routing(
    client: TestClient, harness: Harness
) -> None:
    harness.policy = "complexity"
    strict = client.post(
        "/v1/modernize", json=body(("sumsq.f", LEGACY), threshold=0.0, run_tests=False)
    )
    lenient = client.post(
        "/v1/modernize", json=body(("sumsq.f", LEGACY), threshold=1.0, run_tests=False)
    )
    assert strict.json()["results"][0]["routed_tier"] == "frontier"
    assert lenient.json()["results"][0]["routed_tier"] == "open"


def test_overrides_keep_every_other_router_setting(client: TestClient, harness: Harness) -> None:
    """A policy or threshold override must not reset weights or the forced-frontier rule."""
    weights = DifficultyWeights()
    harness.policy = "complexity"
    harness.router_options = {
        "force_frontier_on_features": False,
        "force_frontier_features": frozenset({"goto_assigned"}),
        "weights": weights,
    }
    harness.frontier_script = harness.open_script = [GOOD]
    client.post(
        "/v1/modernize",
        json=body(("sumsq.f", LEGACY), policy="always_open", threshold=0.5, run_tests=False),
    )
    client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), threshold=0.5, run_tests=False))
    for pipeline, policy in zip(harness.pipelines, ("always_open", "complexity"), strict=True):
        router = pipeline.router
        assert router.policy == policy and router.threshold == 0.5
        assert router.force_frontier_on_features is False
        assert router.force_frontier_features == frozenset({"goto_assigned"})
        assert router.weights is weights
        assert (router.frontier_model, router.open_model) == ("big", "small")


def test_forced_frontier_rule_survives_a_policy_override(
    client: TestClient, harness: Harness
) -> None:
    harness.policy = "always_open"
    harness.router_options = {"force_frontier_on_features": True}
    harness.frontier_script = harness.open_script = [GOOD]
    response = client.post(
        "/v1/modernize",
        json=body(("overly.f", EQUIV_LEGACY), policy="complexity", threshold=1.0, run_tests=False),
    )
    (result,) = response.json()["results"]
    assert result["routed_tier"] == "frontier", "EQUIVALENCE is forced to the frontier model"


def convert_by_unit(prompt: str) -> str:
    return port_for(prompt)


def test_open_fraction_calibrates_the_batch_and_reports_the_plan(
    client: TestClient, harness: Harness
) -> None:
    harness.policy = "complexity"
    harness.open_script = harness.frontier_script = convert_by_unit
    response = client.post(
        "/v1/modernize", json=body(*SMALL_SET, open_fraction=0.5, run_tests=False)
    )
    assert response.status_code == 200
    data = response.json()
    assert [r["success"] for r in data["results"]] == [True] * 4
    tiers = [r["routed_tier"] for r in data["results"]]
    assert tiers == ["open", "open", "frontier", "frontier"], "the two easiest modules go open"
    summary = data["summary"]
    assert summary["planned_open_share"] == 0.5
    assert summary["open_fraction"] == 0.5, "the realised share is reported next to the plan"
    assert summary["routed_open"] == 2


def test_open_fraction_overrides_the_threshold(client: TestClient, harness: Harness) -> None:
    harness.policy = "complexity"
    harness.open_script = harness.frontier_script = convert_by_unit
    response = client.post(
        "/v1/modernize",
        json=body(*SMALL_SET, open_fraction=0.25, threshold=1.0, run_tests=False),
    )
    assert response.json()["summary"]["routed_open"] == 1
    assert response.json()["summary"]["planned_open_share"] == 0.25


def test_open_fraction_edges(client: TestClient, harness: Harness) -> None:
    harness.policy = "complexity"
    harness.open_script = harness.frontier_script = convert_by_unit
    for fraction, expected in ((0.0, 0), (1.0, 4)):
        summary = client.post(
            "/v1/modernize", json=body(*SMALL_SET, open_fraction=fraction, run_tests=False)
        ).json()["summary"]
        assert summary["routed_open"] == expected
        assert summary["planned_open_share"] == expected / 4


def test_without_open_fraction_no_plan_is_reported(client: TestClient, harness: Harness) -> None:
    harness.policy = "complexity"
    harness.open_script = harness.frontier_script = convert_by_unit
    summary = client.post("/v1/modernize", json=body(*SMALL_SET, run_tests=False)).json()["summary"]
    assert "planned_open_share" in summary and summary["planned_open_share"] is None
    assert summary["routed_open"] == 3, "the default threshold alone decides without a target"


def test_the_configured_open_fraction_applies_unless_the_request_sets_one(
    client: TestClient, harness: Harness
) -> None:
    harness.policy = "complexity"
    harness.config = PipelineConfig(target_open_fraction=0.25, run_tests=False)
    harness.open_script = harness.frontier_script = convert_by_unit
    default = client.post("/v1/modernize", json=body(*SMALL_SET, run_tests=False)).json()
    assert default["summary"]["planned_open_share"] == 0.25
    other = client.post(
        "/v1/modernize", json=body(*SMALL_SET, open_fraction=0.75, run_tests=False)
    ).json()
    assert other["summary"]["planned_open_share"] == 0.75
    assert harness.config.target_open_fraction == 0.25, "a request never edits the shared config"


def test_open_fraction_is_ignored_by_the_fixed_policies(
    client: TestClient, harness: Harness
) -> None:
    harness.open_script = harness.frontier_script = convert_by_unit
    summary = client.post(
        "/v1/modernize",
        json=body(*SMALL_SET, policy="always_open", open_fraction=0.25, run_tests=False),
    ).json()["summary"]
    assert summary["routed_open"] == 4
    assert summary["planned_open_share"] is None


def test_several_files_keep_their_order(client: TestClient, harness: Harness) -> None:
    harness.open_script = [GOOD]
    response = client.post(
        "/v1/modernize",
        json=body(("b.f", LEGACY), ("a.f", LEGACY), run_tests=False),
    )
    assert [r["path"] for r in response.json()["results"]] == ["b.f", "a.f"]
    assert response.json()["summary"]["modules"] == 2
    assert response.json()["summary"]["succeeded"] == 2


def test_analyze_reports_features_difficulty_and_routing(
    client: TestClient, harness: Harness
) -> None:
    response = client.post("/v1/analyze", json=body(("pkg/sumsq.f", LEGACY)))
    assert response.status_code == 200
    (file,) = response.json()["files"]

    assert file["path"] == "pkg/sumsq.f"
    assert file["error"] is None
    loops = [f for f in file["features"] if f["id"] == "do_labeled"]
    assert loops and loops[0]["line"] == 5 and "DO 10" in loops[0]["snippet"]
    assert file["metrics"]["sloc"] > 0
    assert 0.0 < file["difficulty"]["score"] < 1.0
    assert file["difficulty"]["reasons"]
    routing = file["routing"]
    assert routing["difficulty"] == pytest.approx(file["difficulty"]["score"])
    assert routing["policy"] == Settings().policy
    assert routing["threshold"] == Settings().threshold
    assert routing["tier"] in {"open", "frontier"}
    assert routing["model"] in {Settings().models.frontier_model, Settings().models.open_model}
    assert harness.pipelines == [], "analysis must not build a pipeline or call a model"


def test_analyze_routes_a_goto_module_above_the_default_threshold(client: TestClient) -> None:
    response = client.post(
        "/v1/analyze", json=body(("sumsq.f", LEGACY), ("gcd.f", GCD_LEGACY), ("e.f", EQUIV_LEGACY))
    )
    easy, gcd, equiv = response.json()["files"]
    assert easy["routing"]["tier"] == "open"
    assert gcd["routing"]["tier"] == "frontier"
    assert any(f["id"] == "goto_unconditional" for f in gcd["features"])
    assert equiv["routing"]["tier"] == "frontier"
    assert "forced to frontier" in equiv["routing"]["reasons"][0]


def test_analyze_uses_the_configured_policy(make_client) -> None:
    settings = Settings(policy="always_frontier")
    response = make_client(settings).post("/v1/analyze", json=body(("a.f", LEGACY)))
    assert response.json()["files"][0]["routing"]["tier"] == "frontier"


def test_analyze_failure_of_one_file_is_generic_and_does_not_stop_the_others(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = app_module.analyze

    def flaky(path: str, source: str):
        if path == "bad.f":
            raise RuntimeError("parser exploded: token=abc123")
        return real(path, source)

    monkeypatch.setattr(app_module, "analyze", flaky)
    with caplog.at_level(logging.ERROR):
        response = client.post("/v1/analyze", json=body(("bad.f", LEGACY), ("ok.f", LEGACY)))
    assert response.status_code == 200
    bad, ok = response.json()["files"]
    assert (bad["path"], bad["error"], bad["features"]) == ("bad.f", "analysis failed", [])
    assert ok["error"] is None and ok["features"]
    assert "abc123" not in response.text
    assert any("abc123" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)


def test_telemetry_and_sink_are_flushed_after_each_request(
    client: TestClient, harness: Harness
) -> None:
    client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    assert harness.flushes == 1
    assert harness.sink.flushes == 1
    harness.open_script = [GOOD]
    client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    assert harness.flushes == 2


def test_failing_flush_does_not_fail_the_request(client: TestClient, harness: Harness) -> None:
    harness.flush_error = RuntimeError("exporter down")
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY), run_tests=False))
    assert response.status_code == 200
    assert response.json()["results"][0]["success"] is True
    assert harness.flushes == 1


def test_a_failing_test_generation_does_not_leak_the_backend_text(
    client: TestClient, harness: Harness
) -> None:
    def refuse(prompt: str) -> str:
        raise LLMError("quota exceeded for project my-secret-project\nINFO forged line")

    harness.tests = refuse
    response = client.post("/v1/modernize", json=body(("sumsq.f", LEGACY)))
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["success"] is False
    assert result["error"] == "test generation failed"
    for leaked in ("my-secret-project", "forged"):
        assert leaked not in response.text
