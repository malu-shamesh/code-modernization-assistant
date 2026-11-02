"""Per-model token pricing.

Prices come from ``pricing.yaml`` (packaged default) or any YAML file with the same shape; the
environment variable ``MODERNIZER_PRICING_FILE`` points the default table at such a file. The
packaged numbers are illustrative placeholders; see the header of that file.

Models you host yourself (for example Gemma on a dedicated accelerator endpoint) are billed per
accelerator hour, not per token. ``amortised_price_per_mtok`` converts that hourly cost and a
measured throughput into an effective per-million-token rate for the table.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

from ..models import TokenUsage

logger = logging.getLogger(__name__)

_PER_MILLION = 1_000_000
_SECONDS_PER_HOUR = 3600
PRICING_FILE_ENV = "MODERNIZER_PRICING_FILE"


def amortised_price_per_mtok(
    node_hour_usd: float, tokens_per_second: float, utilisation: float = 1.0
) -> float:
    """Effective USD per one million tokens for a self-hosted, hour-billed deployment.

    ``node_hour_usd`` is what the serving node (accelerator plus host) costs per hour,
    ``tokens_per_second`` the sustained throughput you measured on it for your real traffic mix,
    and ``utilisation`` the fraction of billed time the node actually serves requests (an idle
    node still bills). The result is
    ``node_hour_usd / (tokens_per_second * 3600 * utilisation) * 1e6``.

    Input and output tokens share one rate here; if you measured them separately, call this once
    per direction. Garbage in, garbage out: use measured throughput, never a datasheet peak.
    """
    for name, value in (
        ("node_hour_usd", node_hour_usd),
        ("tokens_per_second", tokens_per_second),
        ("utilisation", utilisation),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if node_hour_usd < 0:
        raise ValueError(f"node_hour_usd must not be negative, got {node_hour_usd}")
    if tokens_per_second <= 0:
        raise ValueError(f"tokens_per_second must be positive, got {tokens_per_second}")
    if not 0 < utilisation <= 1:
        raise ValueError(f"utilisation must be in (0, 1], got {utilisation}")
    return node_hour_usd / (tokens_per_second * _SECONDS_PER_HOUR * utilisation) * _PER_MILLION


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: TokenUsage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / _PER_MILLION


def _normalise(model: str) -> str:
    """Lower-case and drop a resource-name prefix such as ``publishers/google/models/``."""
    key = model.strip().lower()
    marker = "/models/"
    if marker in key:
        key = key.rsplit(marker, 1)[1]
    return key


class PricingTable:
    """Longest-prefix price lookup. Unknown models cost 0.0 and log one warning per model."""

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices: dict[str, ModelPrice] = {}
        for name, price in prices.items():
            key = _normalise(name)
            if not key:
                raise ValueError("pricing table contains an empty model name")
            if key in self._prices:
                raise ValueError(f"duplicate pricing entry for model {key!r}")
            self._prices[key] = price
        self._warned: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PricingTable:
        return cls._from_text(Path(path).read_text(encoding="utf-8"), str(path))

    @classmethod
    def default(cls) -> PricingTable:
        """The table named by ``MODERNIZER_PRICING_FILE``, else the packaged one.

        Instances are shared per source and file modification time, so warnings are emitted once
        per process and an edited file is re-read.
        """
        override = os.environ.get(PRICING_FILE_ENV, "").strip()
        if not override:
            return _default_table(None, 0)
        return _default_table(override, Path(override).stat().st_mtime_ns)

    @classmethod
    def _from_text(cls, text: str, origin: str) -> PricingTable:
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping) or not isinstance(data.get("models"), Mapping):
            raise ValueError(f"{origin}: expected a top-level 'models' mapping")
        prices: dict[str, ModelPrice] = {}
        for name, entry in data["models"].items():
            if not isinstance(entry, Mapping):
                raise ValueError(f"{origin}: entry for {name!r} must be a mapping")
            try:
                price = ModelPrice(
                    input_per_mtok=float(entry["input_per_mtok"]),
                    output_per_mtok=float(entry["output_per_mtok"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{origin}: entry for {name!r} needs numeric input_per_mtok and output_per_mtok"
                ) from exc
            if price.input_per_mtok < 0 or price.output_per_mtok < 0:
                raise ValueError(f"{origin}: negative price for {name!r}")
            prices[str(name)] = price
        return cls(prices)

    def lookup(self, model: str) -> ModelPrice | None:
        key = _normalise(model)
        best: str | None = None
        for prefix in self._prices:
            if key.startswith(prefix) and (best is None or len(prefix) > len(best)):
                best = prefix
        return self._prices[best] if best is not None else None

    def cost_usd(self, model: str, usage: TokenUsage) -> float:
        price = self.lookup(model)
        if price is None:
            with self._lock:
                first = model not in self._warned
                self._warned.add(model)
            if first:
                logger.warning("no pricing entry for model %r; costing it at 0.0 USD", model)
            return 0.0
        return price.cost(usage)

    def __contains__(self, model: object) -> bool:
        return isinstance(model, str) and self.lookup(model) is not None


@lru_cache(maxsize=8)
def _default_table(override: str | None, mtime_ns: int) -> PricingTable:
    if override is not None:
        return PricingTable.from_yaml(override)
    text = resources.files(__package__).joinpath("pricing.yaml").read_text(encoding="utf-8")
    return PricingTable._from_text(text, "pricing.yaml")


def cost_usd(model: str, usage: TokenUsage) -> float:
    """Cost of ``usage`` on ``model`` according to ``PricingTable.default()``."""
    return PricingTable.default().cost_usd(model, usage)
