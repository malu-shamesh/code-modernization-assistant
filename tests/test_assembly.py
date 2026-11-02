from __future__ import annotations

import pytest

from modernizer.factory import build_pipeline, build_retriever, build_router
from modernizer.fileio import collect_sources, read_source, write_output
from modernizer.llm import ModelSet, ScriptedLLM, build_model_set
from modernizer.models import ModuleResult, RoutingDecision, Tier, TokenUsage
from modernizer.parsing import analyze
from modernizer.report import format_summary, summarize
from modernizer.settings import Settings


def test_settings_defaults_and_env_overrides():
    defaults = Settings.from_env({})
    assert defaults.policy == "complexity"
    assert defaults.retrieval == "local"
    assert defaults.pipeline.max_repairs == 2
    assert defaults.pipeline.target_open_fraction is None

    env = {
        "MODERNIZER_POLICY": "ALWAYS_OPEN",
        "MODERNIZER_THRESHOLD": "0.5",
        "MODERNIZER_MAX_REPAIRS": "1",
        "MODERNIZER_WORKERS": "8",
        "MODERNIZER_RETRIEVAL": "vertex",
        "MODERNIZER_VS_INDEX": "idx",
        "MODERNIZER_PROVIDER": "mock",
        "MODERNIZER_OPEN_FRACTION": "0.7",
    }
    settings = Settings.from_env(env)
    assert settings.policy == "always_open"
    assert settings.threshold == 0.5
    assert settings.pipeline.max_repairs == 1
    assert settings.pipeline.max_workers == 8
    assert settings.retrieval == "vertex"
    assert settings.vector_search.index_name == "idx"
    assert settings.models.provider == "mock"
    assert settings.pipeline.target_open_fraction == 0.7


def test_local_retriever_returns_context_for_detected_constructs():
    retriever = build_retriever(Settings())
    source = (
        "      SUBROUTINE S(I, X)\n"
        "      INTEGER I\n"
        "      REAL X\n"
        "      GO TO (10, 20), I\n"
        "   10 X = 1.0\n"
        "   20 WRITE (6, 100) X\n"
        "  100 FORMAT (1X, F8.3)\n"
        "      RETURN\n"
        "      END\n"
    )
    module = analyze("s.f", source)
    snippets = retriever.retrieve(module, k=4)
    assert snippets
    wanted = {"goto_computed", "format_statement"}
    assert wanted & {f for s in snippets for f in s.feature_ids}


def test_vertex_retriever_needs_all_index_settings():
    settings = Settings.from_env({"MODERNIZER_RETRIEVAL": "vertex"})
    with pytest.raises(ValueError, match="MODERNIZER_VS_INDEX"):
        build_retriever(settings)


def test_router_uses_model_names_and_settings():
    models = ModelSet(
        frontier=ScriptedLLM(["x"], name="big"), open=ScriptedLLM(["x"], name="small")
    )
    router = build_router(Settings.from_env({"MODERNIZER_THRESHOLD": "0.2"}), models)
    assert router.frontier_model == "big"
    assert router.open_model == "small"
    assert router.threshold == 0.2


def test_build_pipeline_with_mock_provider():
    settings = Settings.from_env({"MODERNIZER_PROVIDER": "mock"})
    pipeline = build_pipeline(settings, models=build_model_set(settings.models))
    assert pipeline.run_id
    pipeline.telemetry.shutdown()


def test_collect_sources_walks_dirs_and_skips_noise(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.f").write_text("      END\n", encoding="utf-8")
    (tmp_path / "b.for").write_bytes(b"C caf\xe9\n      END\n")
    (tmp_path / "C.F77").write_text("      END\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "skip.f").write_text("      END\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "script.py").write_text("x = 1\n", encoding="utf-8")

    sources = collect_sources([tmp_path])
    assert set(sources) == {"pkg/a.f", "b.for", "C.F77"}
    assert "caf\xe9" in sources["b.for"], "latin-1 fallback"

    single = collect_sources([tmp_path / "b.for"])
    assert list(single) == ["b.for"]


def test_read_source_strips_bom(tmp_path):
    path = tmp_path / "m.f"
    path.write_bytes(b"\xef\xbb\xbf      END\n")
    assert read_source(path) == "      END\n"


def test_write_output_refuses_to_escape(tmp_path):
    out = tmp_path / "out"
    target = write_output(out, "pkg/a.py", "x = 1\n")
    assert target.read_text(encoding="utf-8") == "x = 1\n"
    with pytest.raises(ValueError):
        write_output(out, "../evil.py", "x = 1\n")


def _result(path: str, tier: Tier, final: Tier, success: bool, tokens: dict[str, int]):
    model = "small" if tier == Tier.OPEN else "big"
    return ModuleResult(
        path=path,
        success=success,
        converted_source="",
        generated_tests="",
        routing=RoutingDecision(tier, model, 0.1, 0.35, "complexity"),
        final_tier=final,
        escalated=tier != final,
        usage_by_model={m: TokenUsage(n, n // 2) for m, n in tokens.items()},
        cost_usd=0.5,
    )


def test_summary_counts_tiers_tokens_and_cost():
    results = [
        _result("a.f", Tier.OPEN, Tier.OPEN, True, {"small": 100, "big": 40}),
        _result("b.f", Tier.OPEN, Tier.FRONTIER, True, {"small": 100, "big": 200}),
        _result("c.f", Tier.FRONTIER, Tier.FRONTIER, False, {"big": 300}),
    ]
    summary = summarize(results)
    assert summary.modules == 3
    assert summary.succeeded == 2
    assert summary.routed_open == 2
    assert summary.escalated == 1
    assert summary.input_tokens == 740
    assert summary.output_tokens == 370
    assert summary.tokens_open == 300
    assert summary.tokens_frontier == 810
    assert summary.cost_usd == 1.5
    assert "66.7%" in format_summary(summary)
    assert summary.to_dict()["total_tokens"] == 1110


def test_summary_of_nothing():
    assert summarize([]).pass_rate == 0.0


def test_settings_reject_non_numeric_values_with_the_variable_name():
    with pytest.raises(ValueError, match="MODERNIZER_OPEN_FRACTION"):
        Settings.from_env({"MODERNIZER_OPEN_FRACTION": "seventy"})
    with pytest.raises(ValueError, match="MODERNIZER_THRESHOLD"):
        Settings.from_env({"MODERNIZER_THRESHOLD": "high"})
