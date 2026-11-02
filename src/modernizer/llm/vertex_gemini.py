"""Gemini on Vertex AI through the ``google-genai`` SDK.

The client is built with ``genai.Client(vertexai=True, project=..., location=...)`` (the SDK
also reads ``GOOGLE_CLOUD_PROJECT`` when ``project`` is None). The ``location`` default
"global" makes the SDK use ``https://aiplatform.googleapis.com/``.

Gemini 2.5 models "think" before answering. Thinking tokens are billed as output tokens, so
they are added to ``output_tokens``; they also count against ``max_output_tokens``, so a low
limit can leave no room for the visible answer (surfaced as ``LLMEmptyResponseError``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from ..models import LLMResponse, TokenUsage
from .errors import LLMBlockedError, LLMEmptyResponseError
from .retry import RetryPolicy, call_with_retry

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BLOCKING_FINISH = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


class GeminiClient:
    """``LLMClient`` for Gemini models served by Vertex AI.

    ``client`` may be any object exposing ``models.generate_content(model=, contents=, config=)``
    (used by tests). ``timeout_s`` is the per-request timeout passed to the SDK.
    """

    def __init__(
        self,
        model: str,
        *,
        project: str | None = None,
        location: str = "global",
        client: object | None = None,
        retry: RetryPolicy | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if client is None:
            if genai is None:
                raise ImportError(
                    "GeminiClient needs google-genai; install it with "
                    "'pip install code-modernization-assistant[gcp]'"
                )
            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)),
            )
        self.name = model
        self._client: Any = client
        self._retry = retry or RetryPolicy()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        config = self._build_config(system, temperature, max_output_tokens)
        started = time.perf_counter()
        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self.name, contents=prompt, config=config
            ),
            self._retry,
            describe=f"Gemini call to {self.name}",
        )
        text = self._extract_text(response)
        return LLMResponse(
            text=text,
            usage=self._extract_usage(response),
            model=self.name,
            latency_s=time.perf_counter() - started,
        )

    def _build_config(self, system: str | None, temperature: float, max_output_tokens: int) -> Any:
        if genai_types is None:
            return {
                "system_instruction": system,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        return genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _extract_text(self, response: Any) -> str:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason is not None:
            reason = _enum_name(block_reason)
            raise LLMBlockedError(f"{self.name}: prompt blocked ({reason})", reason=reason)

        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise LLMEmptyResponseError(f"{self.name}: response contained no candidates")
        candidate = candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        finish_name = _enum_name(finish) if finish is not None else None

        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text = "".join(
            part.text for part in parts if isinstance(part.text, str) and not part.thought
        )
        if not text.strip():
            if finish_name in _BLOCKING_FINISH:
                raise LLMBlockedError(
                    f"{self.name}: response blocked ({finish_name})", reason=finish_name
                )
            hint = " (thinking may have consumed max_output_tokens)" if finish_name else ""
            raise LLMEmptyResponseError(
                f"{self.name}: empty response, finish_reason={finish_name}{hint}"
            )
        if finish_name == "MAX_TOKENS":
            logger.warning("%s: output truncated at max_output_tokens", self.name)
        return text

    def _extract_usage(self, response: Any) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            logger.warning(
                "%s: response carried no usage_metadata; recording zero tokens", self.name
            )
            return TokenUsage()
        output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        return TokenUsage(input_tokens=meta.prompt_token_count or 0, output_tokens=output)
