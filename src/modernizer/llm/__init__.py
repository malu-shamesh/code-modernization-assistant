"""Model-client layer: provider clients, pricing, retry and the frontier/open model pair."""

from __future__ import annotations

from .errors import (
    LLMBlockedError,
    LLMEmptyResponseError,
    LLMError,
    LLMRequestError,
    LLMRetryExhaustedError,
    ScriptExhaustedError,
)
from .mock import ScriptedLLM
from .openai_compat import (
    GoogleTokenProvider,
    OpenAICompatibleClient,
    vertex_endpoint_client,
    vertex_maas_client,
)
from .pricing import ModelPrice, PricingTable, amortised_price_per_mtok, cost_usd
from .registry import ModelConfig, ModelSet, build_model_set
from .retry import RetryPolicy, call_with_retry, is_transient
from .vertex_gemini import GeminiClient

__all__ = [
    "GeminiClient",
    "GoogleTokenProvider",
    "LLMBlockedError",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMRequestError",
    "LLMRetryExhaustedError",
    "ModelConfig",
    "ModelPrice",
    "ModelSet",
    "OpenAICompatibleClient",
    "PricingTable",
    "RetryPolicy",
    "ScriptExhaustedError",
    "ScriptedLLM",
    "amortised_price_per_mtok",
    "build_model_set",
    "call_with_retry",
    "cost_usd",
    "is_transient",
    "vertex_endpoint_client",
    "vertex_maas_client",
]
