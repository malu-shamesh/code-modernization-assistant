from __future__ import annotations

import threading
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from modernizer.llm import (
    GeminiClient,
    LLMBlockedError,
    LLMEmptyResponseError,
    LLMRequestError,
    LLMRetryExhaustedError,
    RetryPolicy,
    vertex_gemini,
)
from modernizer.protocols import LLMClient


def make_response(
    text: str | None = "hello",
    *,
    prompt_tokens: int | None = 11,
    output_tokens: int | None = 7,
    thought_tokens: int | None = None,
    finish: types.FinishReason | None = types.FinishReason.STOP,
    parts: list[types.Part] | None = None,
    no_candidates: bool = False,
    block_reason: types.BlockedReason | None = None,
    usage: bool = True,
) -> types.GenerateContentResponse:
    if parts is None:
        parts = [types.Part(text=text)] if text is not None else []
    candidates = (
        []
        if no_candidates
        else [
            types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)
        ]
    )
    return types.GenerateContentResponse(
        candidates=candidates,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thought_tokens,
        )
        if usage
        else None,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
        if block_reason
        else None,
    )


class FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        with self._lock:
            self.requests.append({"model": model, "contents": contents, "config": config})
            outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeGenAI:
    def __init__(self, outcomes: list[Any]) -> None:
        self.models = FakeModels(outcomes)


def client_for(*outcomes: Any, sleeps: list[float] | None = None) -> tuple[GeminiClient, FakeGenAI]:
    fake = FakeGenAI(list(outcomes))
    policy = RetryPolicy(
        max_attempts=3, jitter=False, sleep=(sleeps if sleeps is not None else []).append
    )
    return GeminiClient("gemini-test", client=fake, retry=policy), fake


def api_error(cls: type[genai_errors.APIError], code: int) -> genai_errors.APIError:
    return cls(code, {"error": {"code": code, "message": "boom", "status": "STATUS"}})


def test_implements_protocol():
    client, _ = client_for(make_response())
    assert isinstance(client, LLMClient)
    assert client.name == "gemini-test"


def test_generate_returns_text_and_usage_and_sends_config():
    client, fake = client_for(make_response("converted", prompt_tokens=100, output_tokens=40))
    resp = client.generate("PROMPT", system="SYS", temperature=0.3, max_output_tokens=512)
    assert resp.text == "converted"
    assert resp.model == "gemini-test"
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (100, 40)
    assert resp.latency_s >= 0
    request = fake.models.requests[0]
    assert request["model"] == "gemini-test"
    assert request["contents"] == "PROMPT"
    config = request["config"]
    assert config.system_instruction == "SYS"
    assert config.temperature == 0.3
    assert config.max_output_tokens == 512


def test_thinking_tokens_are_billed_as_output():
    client, _ = client_for(make_response(output_tokens=40, thought_tokens=60))
    assert client.generate("p").usage.output_tokens == 100


def test_missing_token_counts_default_to_zero():
    client, _ = client_for(make_response(prompt_tokens=None, output_tokens=None))
    usage = client.generate("p").usage
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)


def test_missing_usage_metadata_records_zero_tokens():
    client, _ = client_for(make_response(usage=False))
    assert client.generate("p").usage.total == 0


def test_thought_parts_are_excluded_from_text_and_parts_joined():
    parts = [
        types.Part(text="reasoning...", thought=True),
        types.Part(text="def f():\n"),
        types.Part(text="    return 1\n"),
    ]
    client, _ = client_for(make_response(parts=parts))
    assert client.generate("p").text == "def f():\n    return 1\n"


def test_prompt_block_raises_blocked_error():
    client, _ = client_for(
        make_response(no_candidates=True, block_reason=types.BlockedReason.SAFETY)
    )
    with pytest.raises(LLMBlockedError) as info:
        client.generate("p")
    assert info.value.reason == "SAFETY"


def test_safety_finish_without_text_raises_blocked_error():
    client, _ = client_for(make_response(text=None, finish=types.FinishReason.SAFETY))
    with pytest.raises(LLMBlockedError) as info:
        client.generate("p")
    assert info.value.reason == "SAFETY"


@pytest.mark.parametrize(
    "finish", [types.FinishReason.OTHER, types.FinishReason.MALFORMED_FUNCTION_CALL]
)
def test_non_safety_abnormal_finish_without_text_is_empty_not_blocked(finish):
    client, _ = client_for(make_response(text=None, finish=finish))
    with pytest.raises(LLMEmptyResponseError, match=finish.name):
        client.generate("p")


def test_no_candidates_raises_empty_response_error():
    client, _ = client_for(make_response(no_candidates=True))
    with pytest.raises(LLMEmptyResponseError):
        client.generate("p")


def test_whitespace_only_text_raises_empty_response_error():
    client, _ = client_for(make_response(text="  \n"))
    with pytest.raises(LLMEmptyResponseError):
        client.generate("p")


def test_max_tokens_with_no_visible_text_explains_thinking():
    client, _ = client_for(make_response(text=None, finish=types.FinishReason.MAX_TOKENS))
    with pytest.raises(LLMEmptyResponseError, match="thinking"):
        client.generate("p")


def test_truncated_text_is_returned():
    client, _ = client_for(make_response("partial", finish=types.FinishReason.MAX_TOKENS))
    assert client.generate("p").text == "partial"


def test_transient_error_then_success():
    sleeps: list[float] = []
    client, fake = client_for(
        api_error(genai_errors.ServerError, 503),
        api_error(genai_errors.ClientError, 429),
        make_response("ok"),
        sleeps=sleeps,
    )
    assert client.generate("p").text == "ok"
    assert len(fake.models.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_non_transient_error_fails_immediately():
    client, fake = client_for(api_error(genai_errors.ClientError, 403), make_response())
    with pytest.raises(LLMRequestError) as info:
        client.generate("p")
    assert info.value.status_code == 403
    assert len(fake.models.requests) == 1


def test_exhausted_retries_raise_typed_error():
    err = api_error(genai_errors.ServerError, 503)
    client, fake = client_for(err, err, err)
    with pytest.raises(LLMRetryExhaustedError) as info:
        client.generate("p")
    assert info.value.attempts == 3
    assert len(fake.models.requests) == 3


def test_default_construction_uses_vertex_mode_and_timeout(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(vertex_gemini.genai, "Client", fake_client)
    GeminiClient("gemini-test", project="proj", location="europe-west4", timeout_s=30)
    assert captured["vertexai"] is True
    assert captured["project"] == "proj"
    assert captured["location"] == "europe-west4"
    assert captured["http_options"].timeout == 30_000
