"""Routing policy: pick the frontier or the open model per module."""

from __future__ import annotations

import logging
import math
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from typing import Literal, get_args

from modernizer.features import FEATURES
from modernizer.models import ParsedModule, RoutingDecision, Tier
from modernizer.routing.difficulty import (
    DEFAULT_WEIGHTS,
    DifficultyWeights,
    difficulty_score,
    explain_difficulty,
)

logger = logging.getLogger(__name__)

RoutingPolicyName = Literal["always_frontier", "always_open", "complexity"]

# Constructs where small models fail systematically: they alias or re-target storage and control
# flow in ways that a local rewrite cannot see (EQUIVALENCE overlays storage, assigned GO TO and
# ENTRY jump into the middle of a unit, alternate RETURN changes the call protocol, and passing an
# array element where an array is expected relies on sequence association). Modules containing
# any of them go to the frontier model regardless of their difficulty score. The set is
# configurable through the Router constructor parameter ``force_frontier_features``.
FORCE_FRONTIER_FEATURES: frozenset[str] = frozenset(
    {
        "equivalence",
        "goto_assigned",
        "entry_statement",
        "alternate_return",
        "sequence_association",
    }
)
ESCALATION_REASON = "escalated after failures"


def _validated_features(features: Iterable[str]) -> frozenset[str]:
    result = frozenset(features)
    unknown = sorted(result - FEATURES.keys())
    if unknown:
        raise ValueError(f"unknown feature ids in force_frontier_features: {', '.join(unknown)}")
    return result


class Router:
    """Immutable after construction, so one instance can be shared across threads.

    ``always_frontier`` and ``always_open`` are the baselines: they ignore difficulty and the
    forced-frontier rule so that cost and pass rate can be compared against ``complexity``.

    ``force_frontier_on_features`` switches the forced-frontier rule on or off;
    ``force_frontier_features`` is the set of feature ids that triggers it (default
    ``FORCE_FRONTIER_FEATURES``).
    """

    def __init__(
        self,
        frontier_model: str,
        open_model: str,
        *,
        policy: RoutingPolicyName = "complexity",
        threshold: float = 0.35,
        force_frontier_on_features: bool = True,
        force_frontier_features: frozenset[str] | None = None,
        weights: DifficultyWeights = DEFAULT_WEIGHTS,
    ) -> None:
        if policy not in get_args(RoutingPolicyName):
            raise ValueError(f"unknown routing policy {policy!r}")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be a finite number")
        self.frontier_model = frontier_model
        self.open_model = open_model
        self.policy: RoutingPolicyName = policy
        self.threshold = threshold
        self.force_frontier_on_features = force_frontier_on_features
        self.force_frontier_features = _validated_features(
            FORCE_FRONTIER_FEATURES if force_frontier_features is None else force_frontier_features
        )
        self.weights = weights

    def route(self, module: ParsedModule) -> RoutingDecision:
        difficulty = difficulty_score(module, self.weights)

        if self.policy == "always_frontier":
            return self._decision(
                Tier.FRONTIER, difficulty, None, ["policy always_frontier: difficulty ignored"]
            )
        if self.policy == "always_open":
            return self._decision(
                Tier.OPEN, difficulty, None, ["policy always_open: difficulty ignored"]
            )

        forced = self._forced_features(module)
        if forced:
            reasons = [
                "forced to frontier: module contains " + ", ".join(forced),
                f"difficulty {difficulty:.2f}",
            ]
            return self._decision(Tier.FRONTIER, difficulty, self.threshold, reasons)

        if difficulty <= self.threshold:
            head = f"difficulty {difficulty:.2f} <= threshold {self.threshold:.2f}: open model"
            tier = Tier.OPEN
        else:
            head = f"difficulty {difficulty:.2f} > threshold {self.threshold:.2f}: frontier model"
            tier = Tier.FRONTIER
        reasons = [head, *explain_difficulty(module, self.weights)]
        return self._decision(tier, difficulty, self.threshold, reasons)

    def escalate(self, decision: RoutingDecision) -> RoutingDecision | None:
        """Frontier decision to retry an OPEN module with, or None if already on the frontier."""
        if decision.tier is not Tier.OPEN:
            return None
        return RoutingDecision(
            tier=Tier.FRONTIER,
            model=self.frontier_model,
            difficulty=decision.difficulty,
            threshold=decision.threshold,
            policy=decision.policy,
            reasons=[*decision.reasons, ESCALATION_REASON],
        )

    def with_threshold(self, threshold: float) -> Router:
        """A new Router identical to this one except for the threshold."""
        return Router(
            self.frontier_model,
            self.open_model,
            policy=self.policy,
            threshold=threshold,
            force_frontier_on_features=self.force_frontier_on_features,
            force_frontier_features=self.force_frontier_features,
            weights=self.weights,
        )

    def calibrated_for_batch(
        self, modules: Sequence[ParsedModule], target_open_fraction: float
    ) -> Router:
        """A new Router whose threshold sends about ``target_open_fraction`` of ``modules`` OPEN.

        The share is measured over exactly this batch. Modules forced to the frontier by the
        feature rule count as frontier and are left out of the eligible pool. With
        ``k = floor(target * len(modules) + 0.5)`` (round half up) the ``k`` easiest eligible
        modules go open. Modules with equal scores are always routed alike, so when a tie
        straddles the cut the achievable open count closest to ``k`` is used, and on an equal
        distance the smaller count (fewer open modules). If fewer than ``k`` modules are
        eligible, every eligible module goes open and a warning is logged. Only the
        ``complexity`` policy has a threshold; other policies return ``self`` unchanged. An empty
        batch leaves the threshold unchanged.
        """
        if not 0.0 <= target_open_fraction <= 1.0:
            raise ValueError("target_open_fraction must be in [0, 1]")
        if self.policy != "complexity" or not modules:
            return self

        eligible = sorted(
            difficulty_score(m, self.weights) for m in modules if not self._forced_features(m)
        )
        k = math.floor(target_open_fraction * len(modules) + 0.5)
        if not eligible:
            if k > 0:
                logger.warning(
                    "cannot reach open target %.2f: all %d modules are forced to the frontier",
                    target_open_fraction,
                    len(modules),
                )
            return self

        if len(eligible) < k:
            logger.warning(
                "cannot reach open target %.2f: only %d of %d modules are eligible for the "
                "open model, routing all of them open",
                target_open_fraction,
                len(eligible),
                len(modules),
            )
            return self.with_threshold(eligible[-1])

        best_threshold = math.nextafter(eligible[0], -math.inf)
        best_gap = k
        for t in dict.fromkeys(eligible):
            gap = abs(bisect_right(eligible, t) - k)
            if gap < best_gap:
                best_threshold, best_gap = t, gap
        return self.with_threshold(best_threshold)

    def _forced_features(self, module: ParsedModule) -> list[str]:
        if not self.force_frontier_on_features:
            return []
        return sorted(
            {h.feature_id for h in module.hits if h.feature_id in self.force_frontier_features}
        )

    def _decision(
        self, tier: Tier, difficulty: float, threshold: float | None, reasons: list[str]
    ) -> RoutingDecision:
        model = self.frontier_model if tier is Tier.FRONTIER else self.open_model
        return RoutingDecision(
            tier=tier,
            model=model,
            difficulty=difficulty,
            threshold=threshold,
            policy=self.policy,
            reasons=reasons,
        )


def batch_open_share(router: Router, modules: Sequence[ParsedModule]) -> float:
    """Fraction of ``modules`` that ``router`` routes OPEN (0.0 for an empty batch)."""
    if not modules:
        return 0.0
    return sum(1 for m in modules if router.route(m).tier is Tier.OPEN) / len(modules)
