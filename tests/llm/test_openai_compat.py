from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from modernizer.llm import (
    GoogleTokenProvider,
    LLMBlockedError,
    LLMEmptyResponseError,
    LLMError,
    LLMRequestError,
    LLMRetryExhaustedError,
    OpenAICompatibleClient,
    RetryPolicy,
    vertex_endpoint_client,
    vertex_maas_client,
)
from modernizer.llm.openai_compat import (
    is_google_host,
    vertex_endpoint_base_url,
    vertex_maas_base_url,
)
from modernizer.protocols import LLMClient

OK_BODY = {
    "choices": [
        {"message": {"role": "assistant", "content": "print('hi')"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26},
}


class Recorder:
    """httpx.MockTransport handler that replays queued responses and records requests."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.queue = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(
    *responses: httpx.Response | Exception,
    sleeps: list[float] | None = None,
    token_provider=None,
    max_attempts: int = 3,
) -> tuple[OpenAICompatibleClient, Recorder]:
    recorder = Recorder(*responses)
    http = httpx.Client(transport=httpx.MockTransport(recorder))
    sink = sleeps if sleeps is not None else []
    client = OpenAICompatibleClient(
        "vendor/model-x",
        base_url="https://example.test/v1/",
        token_provider=token_provider,
        http=http,
        retry=RetryPolicy(max_attempts=max_attempts, jitter=False, sleep=sink.append),
    )
    return client, recorder


def json_response(body: Any, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers)


def test_implements_protocol():
    client, _ = make_client()
    assert isinstance(client, LLMClient)
    assert client.name == "vendor/model-x"


def test_request_shape_and_parsing():
    client, rec = make_client(json_response(OK_BODY), token_provider=lambda: "tok-1")
    resp = client.generate("PROMPT", system="SYS", temperature=0.2, max_output_tokens=256)
    assert resp.text == "print('hi')"
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (21, 5)
    assert resp.model == "vendor/model-x"

    request = rec.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer tok-1"
    body = json.loads(request.content)
    assert body["model"] == "vendor/model-x"
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "PROMPT"},
    ]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 256


def test_no_system_message_and_no_auth_header_without_provider():
    client, rec = make_client(json_response(OK_BODY))
    client.generate("PROMPT")
    assert json.loads(rec.requests[0].content)["messages"] == [
        {"role": "user", "content": "PROMPT"}
    ]
    assert "authorization" not in rec.requests[0].headers


def test_token_provider_called_per_attempt():
    tokens = iter(["a", "b"])
    client, rec = make_client(
        json_response({"error": "x"}, 503),
        json_response(OK_BODY),
        token_provider=lambda: next(tokens),
    )
    client.generate("p")
    assert [r.headers["authorization"] for r in rec.requests] == ["Bearer a", "Bearer b"]


def test_transient_status_then_success_with_backoff():
    sleeps: list[float] = []
    client, rec = make_client(
        json_response({"error": "busy"}, 429),
        httpx.ConnectError("refused"),
        json_response(OK_BODY),
        sleeps=sleeps,
    )
    assert client.generate("p").text == "print('hi')"
    assert len(rec.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_retry_after_header_is_honoured():
    sleeps: list[float] = []
    client, _ = make_client(
        json_response({"error": "busy"}, 429, **{"Retry-After": "7"}),
        json_response(OK_BODY),
        sleeps=sleeps,
    )
    client.generate("p")
    assert sleeps == [7.0]


@pytest.mark.parametrize("status", [400, 401, 403])
def test_non_transient_status_fails_immediately(status):
    client, rec = make_client(json_response({"error": "nope"}, status), json_response(OK_BODY))
    with pytest.raises(LLMRequestError) as info:
        client.generate("p")
    assert info.value.status_code == status
    assert len(rec.requests) == 1


def test_exhausted_retries():
    client, rec = make_client(*[json_response({"error": "x"}, 503) for _ in range(3)])
    with pytest.raises(LLMRetryExhaustedError) as info:
        client.generate("p")
    assert info.value.attempts == 3
    assert info.value.status_code == 503
    assert len(rec.requests) == 3


def test_timeout_is_retried():
    client, rec = make_client(httpx.ReadTimeout("slow"), json_response(OK_BODY))
    assert client.generate("p").text
    assert len(rec.requests) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"usage": {}},
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]},
    ],
)
def test_empty_output_raises_without_retry(body):
    client, rec = make_client(json_response(body), json_response(OK_BODY))
    with pytest.raises(LLMEmptyResponseError):
        client.generate("p")
    assert len(rec.requests) == 1


def test_content_filter_raises_blocked():
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
    client, _ = make_client(json_response(body))
    with pytest.raises(LLMBlockedError):
        client.generate("p")


def test_invalid_json_body_raises_llm_error():
    client, rec = make_client(httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(LLMError, match="not valid JSON"):
        client.generate("p")
    assert len(rec.requests) == 1


def test_missing_usage_records_zero_tokens():
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    client, _ = make_client(json_response(body))
    assert client.generate("p").usage.total == 0


def test_content_part_list_is_flattened():
    body = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}]
                },
                "finish_reason": "stop",
            }
        ]
    }
    client, _ = make_client(json_response(body))
    assert client.generate("p").text == "abcd"


class FakeCredentials:
    def __init__(self, project: str | None = None) -> None:
        self.valid = False
        self.token: str | None = None
        self.refreshes = 0
        self.project_id = project

    def refresh(self, request: object) -> None:
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"
        self.valid = True


def test_google_token_provider_refreshes_only_when_invalid():
    creds = FakeCredentials()
    provider = GoogleTokenProvider(creds, project="p")
    assert provider() == "token-1"
    assert provider() == "token-1"
    assert creds.refreshes == 1
    creds.valid = False
    assert provider() == "token-2"


def test_google_token_provider_rejects_empty_token():
    class NoToken(FakeCredentials):
        def refresh(self, request: object) -> None:
            self.valid = True

    with pytest.raises(LLMError):
        GoogleTokenProvider(NoToken(), project="p")()


def test_vertex_base_url_shape():
    assert vertex_maas_base_url("my-proj", "us-south1") == (
        "https://us-south1-aiplatform.googleapis.com/v1/projects/my-proj"
        "/locations/us-south1/endpoints/openapi"
    )
    assert vertex_maas_base_url("p", "global").startswith("https://aiplatform.googleapis.com/v1/")


def test_vertex_maas_client_wires_url_model_and_bearer_token():
    recorder = Recorder(json_response(OK_BODY))
    http = httpx.Client(transport=httpx.MockTransport(recorder))
    client = vertex_maas_client(
        "qwen/qwen3-coder-480b-a35b-instruct-maas",
        project="proj",
        location="us-south1",
        credentials=FakeCredentials(),
        http=http,
    )
    assert client.generate("p").text
    request = recorder.requests[0]
    assert str(request.url) == (
        "https://us-south1-aiplatform.googleapis.com/v1/projects/proj/locations/us-south1"
        "/endpoints/openapi/chat/completions"
    )
    assert request.headers["authorization"] == "Bearer token-1"
    assert json.loads(request.content)["model"] == "qwen/qwen3-coder-480b-a35b-instruct-maas"


def test_vertex_maas_client_project_from_env_and_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    client = vertex_maas_client("m", credentials=FakeCredentials())
    assert "/projects/env-proj/" in client.base_url
    client.close()

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    with pytest.raises(LLMError, match="no Google Cloud project"):
        vertex_maas_client("m", credentials=FakeCredentials())


def make_plain_client(recorder: Recorder, **kwargs: Any) -> OpenAICompatibleClient:
    http = httpx.Client(transport=httpx.MockTransport(recorder))
    return OpenAICompatibleClient("m", base_url="https://example.test/v1", http=http, **kwargs)


def test_fold_system_prepends_system_to_user_message():
    recorder = Recorder(json_response(OK_BODY))
    make_plain_client(recorder, fold_system=True).generate("PROMPT", system="SYS")
    assert json.loads(recorder.requests[0].content)["messages"] == [
        {"role": "user", "content": "SYS\n\nPROMPT"}
    ]


def test_fold_system_without_system_prompt_is_a_no_op():
    recorder = Recorder(json_response(OK_BODY))
    make_plain_client(recorder, fold_system=True).generate("PROMPT")
    assert json.loads(recorder.requests[0].content)["messages"] == [
        {"role": "user", "content": "PROMPT"}
    ]


def test_chat_path_is_overridable():
    recorder = Recorder(json_response(OK_BODY))
    make_plain_client(recorder, chat_path="openai/chat/completions").generate("p")
    assert str(recorder.requests[0].url) == "https://example.test/v1/openai/chat/completions"


def test_vertex_endpoint_base_url_shape():
    assert vertex_endpoint_base_url("my-proj", "us-south1", "42") == (
        "https://us-south1-aiplatform.googleapis.com/v1beta1/projects/my-proj"
        "/locations/us-south1/endpoints/42"
    )
    assert vertex_endpoint_base_url("p", "global", "1", api_version="v1").startswith(
        "https://aiplatform.googleapis.com/v1/projects/p/"
    )


def test_vertex_endpoint_client_url_and_body():
    recorder = Recorder(json_response(OK_BODY))
    http = httpx.Client(transport=httpx.MockTransport(recorder))
    client = vertex_endpoint_client(
        "gemma-3-12b-it",
        endpoint_id="42",
        project="proj",
        location="us-central1",
        credentials=FakeCredentials(),
        http=http,
        fold_system=True,
    )
    client.generate("PROMPT", system="SYS")
    request = recorder.requests[0]
    assert str(request.url) == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/proj/locations/us-central1"
        "/endpoints/42/chat/completions"
    )
    assert request.headers["authorization"] == "Bearer token-1"
    body = json.loads(request.content)
    assert body["model"] == "gemma-3-12b-it"
    assert body["messages"] == [{"role": "user", "content": "SYS\n\nPROMPT"}]


def test_vertex_endpoint_client_overrides_and_validation():
    client = vertex_endpoint_client(
        "m",
        endpoint_id=" 9 ",
        project="p",
        api_version="v1",
        chat_path="/custom",
        credentials=FakeCredentials(),
    )
    assert client.base_url.endswith("/v1/projects/p/locations/us-central1/endpoints/9")
    assert client._chat_path == "/custom"
    client.close()
    with pytest.raises(ValueError, match="endpoint_id"):
        vertex_endpoint_client("m", endpoint_id=" ", project="p", credentials=FakeCredentials())


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://us-central1-aiplatform.googleapis.com/v1", True),
        ("https://abc.us-central1-123.prediction.vertexai.goog/v1", True),
        ("http://localhost:8000/v1", False),
        ("https://notgoogleapis.com/v1", False),
        ("https://googleapis.com.evil.test/v1", False),
    ],
)
def test_is_google_host(url, expected):
    assert is_google_host(url) is expected
