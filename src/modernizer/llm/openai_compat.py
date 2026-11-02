"""Client for OpenAI-compatible ``/chat/completions`` endpoints, and the Vertex AI wiring for it.

``vertex_maas_client`` targets open models in Vertex AI Model Garden that are served as
managed APIs ("model as a service") through Vertex's OpenAI-compatible endpoint:

    https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi

with ``POST {base_url}/chat/completions`` and a Google OAuth bearer token.

Verified locally (google-cloud-aiplatform 2.7 sources): the REST route
``/v1beta1/{endpoint=projects/*/locations/*/endpoints/*}/chat/completions`` exists, so the
``.../endpoints/{endpoint}/chat/completions`` shape is right.

NOT verifiable offline, taken from Google's published documentation and to be checked
against the current docs before relying on it:
  - the literal endpoint id ``openapi`` for managed open models;
  - the ``v1`` (rather than ``v1beta1``) API version for this route;
  - the ``publisher/model-id`` naming of the ``model`` field (for example
    ``qwen/qwen3-coder-480b-a35b-instruct-maas``) and which models are offered in which region;
  - the host for ``location="global"`` (assumed ``aiplatform.googleapis.com``, no region prefix,
    matching how ``google-genai`` treats global).

``vertex_endpoint_client`` targets a model you deploy yourself from Model Garden (for example
Gemma 3 12B on a dedicated accelerator endpoint) and talks to that endpoint's
OpenAI-compatible route:

    https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/{endpoint_id}/chat/completions

UNVERIFIED against live documentation: the ``v1beta1`` API version, whether a dedicated
(private or dedicated-DNS) endpoint must be called on its own hostname instead of the regional
one, and what the ``model`` field must contain for a self-deployed server. Both the API version
and the trailing path are parameters, and a fully explicit URL can be given as
``MODERNIZER_OPEN_BASE_URL`` (see ``registry``), which also serves vLLM, Ollama or GKE deployments.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from ..models import LLMResponse, TokenUsage
from .errors import LLMBlockedError, LLMEmptyResponseError, LLMError, LLMRequestError
from .retry import RetryPolicy, call_with_retry

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    import google.auth
    from google.auth.transport import requests as google_auth_requests
except ImportError:
    google = None  # type: ignore[assignment]
    google_auth_requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_CHAT_PATH = "/chat/completions"
DEFAULT_ENDPOINT_API_VERSION = "v1beta1"
_GOOGLE_HOST_SUFFIXES = ("googleapis.com", "vertexai.goog")
_ERROR_BODY_CHARS = 500


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None  # HTTP-date form is not worth parsing for a back-off hint
    return seconds if seconds >= 0 else None


class OpenAICompatibleClient:
    """``LLMClient`` for any server speaking the OpenAI chat-completions protocol.

    ``token_provider`` is called before every request so short-lived tokens stay fresh.
    ``http`` may be a pre-built ``httpx.Client`` (tests use ``httpx.MockTransport``); a client
    created here is owned and closed by ``close()``.

    ``fold_system`` suits models whose chat template has no system role (Gemma): the system
    prompt is prepended to the single user message instead of being sent as its own message.
    ``chat_path`` is appended to ``base_url`` and can be changed for servers that deviate.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        token_provider: Callable[[], str] | None = None,
        http: httpx.Client | None = None,
        retry: RetryPolicy | None = None,
        timeout_s: float = 120.0,
        fold_system: bool = False,
        chat_path: str = DEFAULT_CHAT_PATH,
    ) -> None:
        if http is None:
            if httpx is None:
                raise ImportError(
                    "OpenAICompatibleClient needs httpx; install it with "
                    "'pip install code-modernization-assistant[gcp]'"
                )
            http = httpx.Client()
            self._owns_http = True
        else:
            self._owns_http = False
        self.name = model
        self.base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._http = http
        self._retry = retry or RetryPolicy()
        self._timeout_s = timeout_s
        self._fold_system = fold_system
        self._chat_path = "/" + chat_path.lstrip("/")

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system and self._fold_system:
            prompt = f"{system}\n\n{prompt}"
        elif system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        started = time.perf_counter()
        body = call_with_retry(
            lambda: self._post(payload), self._retry, describe=f"chat completion on {self.name}"
        )
        return LLMResponse(
            text=self._extract_text(body),
            usage=self._extract_usage(body),
            model=self.name,
            latency_s=time.perf_counter() - started,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {self._token_provider()}"
        response = self._http.post(
            f"{self.base_url}{self._chat_path}",
            json=payload,
            headers=headers,
            timeout=self._timeout_s,
        )
        if response.status_code >= 400:
            snippet = response.text[:_ERROR_BODY_CHARS]
            raise LLMRequestError(
                f"{self.name}: HTTP {response.status_code}: {snippet}",
                status_code=response.status_code,
                retry_after_s=_parse_retry_after(response.headers.get("retry-after")),
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.name}: response was not valid JSON") from exc
        if not isinstance(body, dict):
            raise LLMError(f"{self.name}: unexpected response payload type {type(body).__name__}")
        return body

    def _extract_text(self, body: Mapping[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMEmptyResponseError(f"{self.name}: response contained no choices")
        choice = choices[0]
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        finish = choice.get("finish_reason")
        if not isinstance(content, str) or not content.strip():
            if finish == "content_filter":
                raise LLMBlockedError(
                    f"{self.name}: response blocked (content_filter)", reason=finish
                )
            raise LLMEmptyResponseError(f"{self.name}: empty response, finish_reason={finish}")
        if finish == "length":
            logger.warning("%s: output truncated at max_tokens", self.name)
        return content

    def _extract_usage(self, body: Mapping[str, Any]) -> TokenUsage:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            logger.warning("%s: response carried no usage; recording zero tokens", self.name)
            return TokenUsage()
        return TokenUsage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


class GoogleTokenProvider:
    """Thread-safe OAuth access tokens from Application Default Credentials.

    ``credentials`` may be injected (tests, service-account impersonation). Credentials are
    loaded and refreshed lazily, under a lock because refreshing mutates shared state.
    """

    def __init__(self, credentials: Any | None = None, project: str | None = None) -> None:
        self._credentials = credentials
        self._project = project
        self._lock = threading.Lock()

    def _ensure_credentials(self) -> Any:
        if self._credentials is None:
            if google_auth_requests is None:
                raise ImportError(
                    "Vertex AI authentication needs google-auth; install it with "
                    "'pip install code-modernization-assistant[gcp]'"
                )
            self._credentials, default_project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
            self._project = self._project or default_project
        return self._credentials

    @property
    def project(self) -> str | None:
        with self._lock:
            if self._project is None:
                credentials = self._ensure_credentials()
                self._project = self._project or getattr(credentials, "project_id", None)
            return self._project

    def __call__(self) -> str:
        with self._lock:
            credentials = self._ensure_credentials()
            if not credentials.valid:
                credentials.refresh(google_auth_requests.Request())
            token = credentials.token
        if not token:
            raise LLMError("Google credentials produced no access token")
        return str(token)


def _vertex_host(location: str) -> str:
    return (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )


def is_google_host(url: str) -> bool:
    """True when ``url`` points at a Google API host, where an ADC bearer token is appropriate."""
    host = (urlsplit(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in _GOOGLE_HOST_SUFFIXES)


def vertex_maas_base_url(project: str, location: str) -> str:
    return (
        f"https://{_vertex_host(location)}/v1/projects/{project}"
        f"/locations/{location}/endpoints/openapi"
    )


def vertex_endpoint_base_url(
    project: str,
    location: str,
    endpoint_id: str,
    api_version: str = DEFAULT_ENDPOINT_API_VERSION,
) -> str:
    """Base URL of a deployed Vertex AI endpoint; the client appends ``/chat/completions``."""
    return (
        f"https://{_vertex_host(location)}/{api_version}/projects/{project}"
        f"/locations/{location}/endpoints/{endpoint_id}"
    )


def _resolve_project(provider: GoogleTokenProvider) -> str:
    resolved = provider.project
    if not resolved:
        raise LLMError(
            "no Google Cloud project: pass project=, set GOOGLE_CLOUD_PROJECT, "
            "or use credentials that carry one"
        )
    return resolved


def vertex_maas_client(
    model: str,
    *,
    project: str | None = None,
    location: str = "us-central1",
    credentials: Any | None = None,
    http: httpx.Client | None = None,
    retry: RetryPolicy | None = None,
    timeout_s: float = 120.0,
    fold_system: bool = False,
) -> OpenAICompatibleClient:
    """Client for a Model Garden open model served through Vertex's OpenAI-compatible API.

    The project comes from ``project``, then ``GOOGLE_CLOUD_PROJECT``, then the default
    credentials. See the module docstring for what could not be verified offline.
    """
    provider = GoogleTokenProvider(
        credentials, project=project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    return OpenAICompatibleClient(
        model,
        base_url=vertex_maas_base_url(_resolve_project(provider), location),
        token_provider=provider,
        http=http,
        retry=retry,
        timeout_s=timeout_s,
        fold_system=fold_system,
    )


def vertex_endpoint_client(
    model: str,
    *,
    endpoint_id: str,
    project: str | None = None,
    location: str = "us-central1",
    api_version: str = DEFAULT_ENDPOINT_API_VERSION,
    chat_path: str = DEFAULT_CHAT_PATH,
    credentials: Any | None = None,
    http: httpx.Client | None = None,
    retry: RetryPolicy | None = None,
    timeout_s: float = 120.0,
    fold_system: bool = False,
) -> OpenAICompatibleClient:
    """Client for a self-deployed Model Garden model on a dedicated Vertex AI endpoint.

    Billing for such a deployment is per accelerator hour, not per token (see ``pricing``).
    See the module docstring for the URL shape and what could not be verified offline.
    """
    if not endpoint_id.strip():
        raise ValueError("endpoint_id must not be empty")
    provider = GoogleTokenProvider(
        credentials, project=project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    return OpenAICompatibleClient(
        model,
        base_url=vertex_endpoint_base_url(
            _resolve_project(provider), location, endpoint_id.strip(), api_version
        ),
        token_provider=provider,
        http=http,
        retry=retry,
        timeout_s=timeout_s,
        fold_system=fold_system,
        chat_path=chat_path,
    )
