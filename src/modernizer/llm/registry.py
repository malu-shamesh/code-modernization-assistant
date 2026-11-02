"""Model configuration and construction of the frontier/open model pair.

The default model ids below are configuration defaults, not claims about availability. Model
ids, regions and access change over time: check them against the current Vertex AI model
catalogue (Model Garden) and override through the environment variables read by
``ModelConfig.from_env``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from ..models import Tier
from ..protocols import LLMClient
from .mock import ScriptedLLM
from .openai_compat import (
    GoogleTokenProvider,
    OpenAICompatibleClient,
    is_google_host,
    vertex_endpoint_client,
    vertex_maas_client,
)
from .vertex_gemini import GeminiClient

Provider = Literal["vertex", "mock"]
OpenBackend = Literal["endpoint", "maas", "openai"]

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

MOCK_STUB_RESPONSE = "```python\n# stub output from the mock provider\npass\n```"


def _parse_bool(name: str, value: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{name}={value!r} is not a boolean; use true/false, 1/0, yes/no or on/off")


@dataclass(frozen=True)
class ModelConfig:
    """Which models to call.

    The open tier defaults to Gemma 3 12B (instruction tuned) on a Vertex AI Model Garden
    endpoint you deploy yourself (``open_backend="endpoint"``, needs ``open_endpoint_id``).
    Other backends: ``"maas"`` (a Model Garden managed API) and ``"openai"`` (any
    OpenAI-compatible server at ``open_base_url``, such as vLLM, Ollama or a GKE deployment).
    A non-empty ``open_base_url`` takes precedence over ``open_backend``.
    ``open_fold_system`` prepends the system prompt to the user message, because Gemma chat
    templates have no system role.
    """

    provider: Provider = "vertex"
    project: str | None = None
    frontier_model: str = "gemini-2.5-pro"
    open_model: str = "gemma-3-12b-it"
    frontier_location: str = "global"
    open_location: str = "us-south1"
    open_backend: OpenBackend = "endpoint"
    open_endpoint_id: str | None = None
    open_base_url: str | None = None
    open_fold_system: bool = True

    def __post_init__(self) -> None:
        if self.provider not in get_args(Provider):
            raise ValueError(
                f"unknown provider {self.provider!r}; expected one of {get_args(Provider)}"
            )
        if self.open_backend not in get_args(OpenBackend):
            raise ValueError(
                f"unknown open backend {self.open_backend!r}; "
                f"expected one of {get_args(OpenBackend)}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelConfig:
        """Build a config from ``MODERNIZER_*`` and ``GOOGLE_CLOUD_PROJECT`` variables.

        Unset or empty variables fall back to the dataclass defaults.
        """
        source = os.environ if env is None else env

        def read(name: str) -> str | None:
            value = source.get(name, "").strip()
            return value or None

        defaults = cls()
        fold = read("MODERNIZER_OPEN_FOLD_SYSTEM")
        return cls(
            provider=(read("MODERNIZER_PROVIDER") or defaults.provider).lower(),  # type: ignore[arg-type]
            project=read("GOOGLE_CLOUD_PROJECT"),
            frontier_model=read("MODERNIZER_FRONTIER_MODEL") or defaults.frontier_model,
            open_model=read("MODERNIZER_OPEN_MODEL") or defaults.open_model,
            frontier_location=read("MODERNIZER_FRONTIER_LOCATION") or defaults.frontier_location,
            open_location=read("MODERNIZER_OPEN_LOCATION") or defaults.open_location,
            open_backend=(read("MODERNIZER_OPEN_BACKEND") or defaults.open_backend).lower(),  # type: ignore[arg-type]
            open_endpoint_id=read("MODERNIZER_OPEN_ENDPOINT_ID"),
            open_base_url=read("MODERNIZER_OPEN_BASE_URL"),
            open_fold_system=(
                defaults.open_fold_system
                if fold is None
                else _parse_bool("MODERNIZER_OPEN_FOLD_SYSTEM", fold)
            ),
        )


@dataclass(frozen=True)
class ModelSet:
    frontier: LLMClient
    open: LLMClient

    def for_tier(self, tier: Tier) -> LLMClient:
        if tier == Tier.FRONTIER:
            return self.frontier
        if tier == Tier.OPEN:
            return self.open
        raise ValueError(f"unknown tier {tier!r}")

    def name_for(self, tier: Tier) -> str:
        return self.for_tier(tier).name


def _build_open_client(config: ModelConfig) -> LLMClient:
    """The open-tier client; raises ``ValueError`` for an inconsistent configuration."""
    base_url = (config.open_base_url or "").strip()
    if base_url:
        # Google-hosted URLs get an ADC bearer token; self-hosted servers are called as given.
        provider = GoogleTokenProvider(project=config.project) if is_google_host(base_url) else None
        return OpenAICompatibleClient(
            config.open_model,
            base_url=base_url,
            token_provider=provider,
            fold_system=config.open_fold_system,
        )
    if config.open_backend == "openai":
        raise ValueError("open_backend 'openai' needs open_base_url (MODERNIZER_OPEN_BASE_URL)")
    if config.open_backend == "maas":
        return vertex_maas_client(
            config.open_model,
            project=config.project,
            location=config.open_location,
            fold_system=config.open_fold_system,
        )
    if not (config.open_endpoint_id or "").strip():
        raise ValueError(
            "open_backend 'endpoint' needs open_endpoint_id (MODERNIZER_OPEN_ENDPOINT_ID); "
            "deploy the model from Model Garden first, or choose another open_backend"
        )
    if not config.project:
        raise ValueError(
            "open_backend 'endpoint' needs a Google Cloud project (GOOGLE_CLOUD_PROJECT)"
        )
    return vertex_endpoint_client(
        config.open_model,
        endpoint_id=config.open_endpoint_id,
        project=config.project,
        location=config.open_location,
        fold_system=config.open_fold_system,
    )


def build_model_set(config: ModelConfig) -> ModelSet:
    """Create the two clients described by ``config``.

    Provider "mock" returns offline ``ScriptedLLM`` clients that always answer with a fixed stub.
    """
    if config.provider == "mock":
        return ModelSet(
            frontier=ScriptedLLM(lambda _prompt: MOCK_STUB_RESPONSE, name="mock-frontier"),
            open=ScriptedLLM(lambda _prompt: MOCK_STUB_RESPONSE, name="mock-open"),
        )
    open_client = _build_open_client(config)
    return ModelSet(
        frontier=GeminiClient(
            config.frontier_model, project=config.project, location=config.frontier_location
        ),
        open=open_client,
    )
