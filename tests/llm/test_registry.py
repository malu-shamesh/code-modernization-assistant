from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from modernizer.llm import ModelConfig, ModelSet, OpenAICompatibleClient, build_model_set, registry
from modernizer.models import Tier

ENV_VARS = (
    "MODERNIZER_PROVIDER",
    "GOOGLE_CLOUD_PROJECT",
    "MODERNIZER_FRONTIER_MODEL",
    "MODERNIZER_OPEN_MODEL",
    "MODERNIZER_FRONTIER_LOCATION",
    "MODERNIZER_OPEN_LOCATION",
    "MODERNIZER_OPEN_BACKEND",
    "MODERNIZER_OPEN_ENDPOINT_ID",
    "MODERNIZER_OPEN_BASE_URL",
    "MODERNIZER_OPEN_FOLD_SYSTEM",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    cfg = ModelConfig()
    assert cfg.provider == "vertex"
    assert cfg.project is None
    assert cfg.frontier_model == "gemini-2.5-pro"
    assert cfg.open_model == "gemma-3-12b-it"
    assert cfg.open_location == "us-south1"
    assert cfg.open_backend == "endpoint"
    assert cfg.open_endpoint_id is None
    assert cfg.open_base_url is None
    assert cfg.open_fold_system is True


def test_from_env_with_nothing_set_gives_defaults():
    assert ModelConfig.from_env() == ModelConfig()


def test_from_env_reads_every_variable(monkeypatch):
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    monkeypatch.setenv("MODERNIZER_FRONTIER_MODEL", "big")
    monkeypatch.setenv("MODERNIZER_OPEN_MODEL", "small")
    monkeypatch.setenv("MODERNIZER_FRONTIER_LOCATION", "europe-west4")
    monkeypatch.setenv("MODERNIZER_OPEN_LOCATION", "us-east5")
    monkeypatch.setenv("MODERNIZER_OPEN_BACKEND", " MaaS ")
    monkeypatch.setenv("MODERNIZER_OPEN_ENDPOINT_ID", "12345")
    monkeypatch.setenv("MODERNIZER_OPEN_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("MODERNIZER_OPEN_FOLD_SYSTEM", "false")
    assert ModelConfig.from_env() == ModelConfig(
        provider="mock",
        project="proj-1",
        frontier_model="big",
        open_model="small",
        frontier_location="europe-west4",
        open_location="us-east5",
        open_backend="maas",
        open_endpoint_id="12345",
        open_base_url="http://vllm:8000/v1",
        open_fold_system=False,
    )


def test_empty_variables_are_treated_as_unset(monkeypatch):
    monkeypatch.setenv("MODERNIZER_OPEN_MODEL", "  ")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    cfg = ModelConfig.from_env()
    assert cfg.open_model == ModelConfig().open_model
    assert cfg.project is None


def test_provider_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("MODERNIZER_PROVIDER", " Mock ")
    assert ModelConfig.from_env().provider == "mock"


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("MODERNIZER_PROVIDER", "azure")
    with pytest.raises(ValueError, match="unknown provider"):
        ModelConfig.from_env()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("TRUE", True), ("yes", True), ("on", True), ("0", False), ("No", False)],
)
def test_fold_system_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("MODERNIZER_OPEN_FOLD_SYSTEM", raw)
    assert ModelConfig.from_env().open_fold_system is expected


def test_fold_system_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("MODERNIZER_OPEN_FOLD_SYSTEM", "maybe")
    with pytest.raises(ValueError, match="MODERNIZER_OPEN_FOLD_SYSTEM"):
        ModelConfig.from_env()


def test_unknown_open_backend_rejected(monkeypatch):
    monkeypatch.setenv("MODERNIZER_OPEN_BACKEND", "sagemaker")
    with pytest.raises(ValueError, match="unknown open backend"):
        ModelConfig.from_env()


def test_mock_provider_builds_offline_stub_clients():
    models = build_model_set(ModelConfig(provider="mock"))
    assert models.name_for(Tier.FRONTIER) == "mock-frontier"
    assert models.name_for(Tier.OPEN) == "mock-open"
    assert models.for_tier(Tier.FRONTIER) is models.frontier
    assert models.for_tier(Tier.OPEN) is models.open
    for tier in Tier:
        resp = models.for_tier(tier).generate("anything")
        assert resp.text == registry.MOCK_STUB_RESPONSE
        assert resp.model == models.name_for(tier)


class FakeGemini:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.name = model
        self.kwargs = kwargs


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setattr(registry, "GeminiClient", FakeGemini)


def fake_open_client(calls: dict[str, Any], key: str):
    def factory(model: str, **kwargs: Any) -> object:
        calls[key] = (model, kwargs)
        return type("FakeOpen", (), {"name": model})()

    return factory


def test_maas_backend_wires_clients(monkeypatch, fake_gemini):
    calls: dict[str, Any] = {}
    monkeypatch.setattr(registry, "vertex_maas_client", fake_open_client(calls, "maas"))
    cfg = ModelConfig(
        project="p",
        frontier_model="f",
        open_model="o/x",
        open_location="us-east5",
        open_backend="maas",
        open_fold_system=False,
    )
    models = build_model_set(cfg)
    assert isinstance(models, ModelSet)
    assert models.frontier.kwargs == {"project": "p", "location": "global"}
    assert calls["maas"] == (
        "o/x",
        {"project": "p", "location": "us-east5", "fold_system": False},
    )
    assert models.name_for(Tier.FRONTIER) == "f"
    assert models.name_for(Tier.OPEN) == "o/x"


def test_endpoint_backend_wires_client(monkeypatch, fake_gemini):
    calls: dict[str, Any] = {}
    monkeypatch.setattr(registry, "vertex_endpoint_client", fake_open_client(calls, "endpoint"))
    models = build_model_set(ModelConfig(project="p", open_endpoint_id="777"))
    assert calls["endpoint"] == (
        "gemma-3-12b-it",
        {"endpoint_id": "777", "project": "p", "location": "us-south1", "fold_system": True},
    )
    assert models.name_for(Tier.OPEN) == "gemma-3-12b-it"


def test_endpoint_backend_builds_real_client_with_expected_url(fake_gemini):
    models = build_model_set(ModelConfig(project="p", open_endpoint_id="777"))
    assert isinstance(models.open, OpenAICompatibleClient)
    assert models.open.base_url == (
        "https://us-south1-aiplatform.googleapis.com/v1beta1/projects/p"
        "/locations/us-south1/endpoints/777"
    )


def test_base_url_overrides_backend_and_folds_system(fake_gemini):
    cfg = ModelConfig(open_backend="maas", open_base_url="http://vllm.internal:8000/v1/")
    models = build_model_set(cfg)
    assert isinstance(models.open, OpenAICompatibleClient)
    assert models.open.base_url == "http://vllm.internal:8000/v1"
    assert models.open._token_provider is None
    assert models.open._fold_system is True


def test_google_hosted_base_url_gets_a_token_provider(fake_gemini):
    cfg = ModelConfig(project="p", open_base_url="https://x.prediction.vertexai.goog/v1")
    assert build_model_set(cfg).open._token_provider is not None


def test_look_alike_host_gets_no_token_provider(fake_gemini):
    cfg = ModelConfig(open_base_url="https://evilgoogleapis.com/v1")
    assert build_model_set(cfg).open._token_provider is None


def test_base_url_client_sends_folded_request(fake_gemini):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    models = build_model_set(ModelConfig(open_base_url="http://vllm:8000/v1"))
    models.open._http = httpx.Client(transport=httpx.MockTransport(handler))
    assert models.open.generate("PROMPT", system="SYS").text == "ok"
    assert str(seen[0].url) == "http://vllm:8000/v1/chat/completions"
    assert json.loads(seen[0].content)["messages"] == [{"role": "user", "content": "SYS\n\nPROMPT"}]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ModelConfig(project="p"), "open_endpoint_id"),
        (ModelConfig(project="p", open_endpoint_id="  "), "open_endpoint_id"),
        (ModelConfig(open_endpoint_id="1"), "Google Cloud project"),
        (ModelConfig(project="p", open_backend="openai"), "open_base_url"),
    ],
)
def test_misconfiguration_raises_clear_value_error(fake_gemini, config, message):
    with pytest.raises(ValueError, match=message):
        build_model_set(config)


def test_mock_provider_ignores_open_backend_settings():
    models = build_model_set(ModelConfig(provider="mock", open_backend="openai"))
    assert models.name_for(Tier.OPEN) == "mock-open"


def test_blank_base_url_is_treated_as_unset(fake_gemini):
    with pytest.raises(ValueError, match="open_base_url"):
        build_model_set(ModelConfig(open_backend="openai", open_base_url="   "))
