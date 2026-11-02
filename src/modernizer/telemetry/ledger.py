"""In-process, per-module accounting of tokens and cost."""

from __future__ import annotations

import threading
from collections import defaultdict

from ..models import Tier, TokenUsage


def tier_value(tier: Tier | str) -> str:
    return tier.value if isinstance(tier, Tier) else str(tier)


class UsageLedger:
    """Thread-safe accumulator of token usage and cost, keyed by module, model and tier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: dict[str, dict[str, TokenUsage]] = defaultdict(dict)
        self._cost: dict[str, float] = defaultdict(float)
        self._by_tier: dict[str, TokenUsage] = {}
        self._cost_by_tier: dict[str, float] = defaultdict(float)

    def record(
        self,
        module: str,
        model: str,
        tier: Tier | str,
        stage: str,
        usage: TokenUsage,
        cost_usd: float,
    ) -> None:
        tier_key = tier_value(tier)
        with self._lock:
            per_model = self._usage[module]
            per_model[model] = per_model.get(model, TokenUsage()) + usage
            self._cost[module] += cost_usd
            self._by_tier[tier_key] = self._by_tier.get(tier_key, TokenUsage()) + usage
            self._cost_by_tier[tier_key] += cost_usd

    def usage_by_model(self, module: str) -> dict[str, TokenUsage]:
        with self._lock:
            per_model = self._usage.get(module, {})
            return {m: TokenUsage(u.input_tokens, u.output_tokens) for m, u in per_model.items()}

    def cost(self, module: str) -> float:
        with self._lock:
            return self._cost.get(module, 0.0)

    def totals(self) -> tuple[dict[str, TokenUsage], float]:
        """Usage per model and total cost across every module."""
        with self._lock:
            merged: dict[str, TokenUsage] = {}
            for per_model in self._usage.values():
                for model, usage in per_model.items():
                    merged[model] = merged.get(model, TokenUsage()) + usage
            return merged, sum(self._cost.values())

    def by_tier(self) -> dict[str, TokenUsage]:
        with self._lock:
            return {
                t: TokenUsage(u.input_tokens, u.output_tokens) for t, u in self._by_tier.items()
            }

    def cost_by_tier(self) -> dict[str, float]:
        with self._lock:
            return dict(self._cost_by_tier)

    def modules(self) -> list[str]:
        with self._lock:
            return list(self._usage)
