"""Runtime settings, read from ``MODERNIZER_*`` environment variables (CLI flags override)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from .llm import ModelConfig
from .pipeline import PipelineConfig
from .routing import RoutingPolicyName

RetrievalBackend = Literal["local", "vertex"]


def _number(name: str, raw: str | float, cast: type[float] | type[int]):
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


@dataclass(frozen=True)
class VectorSearchSettings:
    index_name: str | None = None
    endpoint_name: str | None = None
    deployed_index_id: str | None = None
    location: str = "us-central1"
    embedding_model: str = "text-embedding-005"
    embedding_dim: int = 768


@dataclass(frozen=True)
class Settings:
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalBackend = "local"
    vector_search: VectorSearchSettings = field(default_factory=VectorSearchSettings)
    telemetry: str = "none"
    otlp_endpoint: str | None = None
    sink: str | None = None
    policy: RoutingPolicyName = "complexity"
    threshold: float = 0.35
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env

        def read(name: str) -> str | None:
            value = source.get(name, "").strip()
            return value or None

        defaults = cls()
        vs_defaults = VectorSearchSettings()
        pipeline = defaults.pipeline
        if (repairs := read("MODERNIZER_MAX_REPAIRS")) is not None:
            pipeline = replace(
                pipeline, max_repairs=_number("MODERNIZER_MAX_REPAIRS", repairs, int)
            )
        if (workers := read("MODERNIZER_WORKERS")) is not None:
            pipeline = replace(pipeline, max_workers=_number("MODERNIZER_WORKERS", workers, int))
        if (fraction := read("MODERNIZER_OPEN_FRACTION")) is not None:
            pipeline = replace(
                pipeline, target_open_fraction=_number("MODERNIZER_OPEN_FRACTION", fraction, float)
            )
        if (timeout := read("MODERNIZER_TEST_TIMEOUT_S")) is not None:
            pipeline = replace(
                pipeline,
                test_timeout_s=_number("MODERNIZER_TEST_TIMEOUT_S", timeout, float),
            )
        return cls(
            models=ModelConfig.from_env(source),
            retrieval=(read("MODERNIZER_RETRIEVAL") or defaults.retrieval).lower(),  # type: ignore[arg-type]
            vector_search=VectorSearchSettings(
                index_name=read("MODERNIZER_VS_INDEX"),
                endpoint_name=read("MODERNIZER_VS_ENDPOINT"),
                deployed_index_id=read("MODERNIZER_VS_DEPLOYED_ID"),
                location=read("MODERNIZER_VS_LOCATION") or vs_defaults.location,
                embedding_model=read("MODERNIZER_EMBED_MODEL") or vs_defaults.embedding_model,
                embedding_dim=int(read("MODERNIZER_EMBED_DIM") or vs_defaults.embedding_dim),
            ),
            telemetry=(read("MODERNIZER_TELEMETRY") or defaults.telemetry).lower(),
            otlp_endpoint=read("MODERNIZER_OTLP_ENDPOINT"),
            sink=read("MODERNIZER_SINK"),
            policy=(read("MODERNIZER_POLICY") or defaults.policy).lower(),  # type: ignore[arg-type]
            threshold=_number(
                "MODERNIZER_THRESHOLD", read("MODERNIZER_THRESHOLD") or defaults.threshold, float
            ),
            pipeline=pipeline,
        )
