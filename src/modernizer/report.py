"""Aggregate statistics over a set of module results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .models import ModuleResult, Tier


@dataclass(frozen=True)
class RunSummary:
    modules: int
    succeeded: int
    pass_rate: float
    routed_open: int
    open_fraction: float
    escalated: int
    final_open: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tokens_open: int
    tokens_frontier: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


def summarize(results: Sequence[ModuleResult]) -> RunSummary:
    n = len(results)
    succeeded = sum(r.success for r in results)
    routed_open = sum(r.routing.tier == Tier.OPEN for r in results)
    input_tokens = output_tokens = tokens_open = tokens_frontier = 0
    open_models = {r.routing.model for r in results if r.routing.tier == Tier.OPEN}
    for r in results:
        for model, usage in r.usage_by_model.items():
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            if model in open_models:
                tokens_open += usage.total
            else:
                tokens_frontier += usage.total
    return RunSummary(
        modules=n,
        succeeded=succeeded,
        pass_rate=succeeded / n if n else 0.0,
        routed_open=routed_open,
        open_fraction=routed_open / n if n else 0.0,
        escalated=sum(r.escalated for r in results),
        final_open=sum(r.final_tier == Tier.OPEN and r.success for r in results),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=sum(r.cost_usd for r in results),
        tokens_open=tokens_open,
        tokens_frontier=tokens_frontier,
    )


def format_summary(summary: RunSummary) -> str:
    return "\n".join(
        [
            f"modules:            {summary.modules}",
            f"passed:             {summary.succeeded} ({summary.pass_rate:.1%})",
            f"routed to open:     {summary.routed_open} ({summary.open_fraction:.1%})",
            f"escalated:          {summary.escalated}",
            f"tokens in / out:    {summary.input_tokens} / {summary.output_tokens}",
            f"estimated cost USD: {summary.cost_usd:.4f}",
        ]
    )
