"""Routing policies under evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..routing import RoutingPolicyName


@dataclass(frozen=True)
class PolicySpec:
    """One arm of the experiment.

    ``threshold`` only matters for the "complexity" policy; None means "use the threshold from
    the settings". ``target_open_fraction`` (also complexity only) asks the pipeline to calibrate
    the threshold on the evaluated batch so that about that share of the modules is routed to
    the open model; it replaces ``threshold``, which then only documents the starting point.
    It is a policy parameter, not a result. ``escalate`` allows the open tier to hand a failed
    module to the frontier.
    """

    name: str
    policy: RoutingPolicyName
    threshold: float | None = None
    escalate: bool = True
    target_open_fraction: float | None = None

    def __post_init__(self) -> None:
        target = self.target_open_fraction
        if target is None:
            return
        if self.policy != "complexity":
            raise ValueError(
                f"policy {self.name!r}: target_open_fraction only applies to the complexity policy"
            )
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError(f"policy {self.name!r}: target_open_fraction must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "policy": self.policy,
            "threshold": self.threshold,
            "escalate": self.escalate,
            "target_open_fraction": self.target_open_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PolicySpec:
        threshold = data["threshold"]
        target = data.get("target_open_fraction")
        return cls(
            name=str(data["name"]),
            policy=data["policy"],  # type: ignore[arg-type]
            threshold=None if threshold is None else float(threshold),  # type: ignore[arg-type]
            escalate=bool(data["escalate"]),
            target_open_fraction=None if target is None else float(target),  # type: ignore[arg-type]
        )


def default_policies(
    threshold: float, target_open_fraction: float | None = None
) -> list[PolicySpec]:
    """The baseline pair plus the cost-aware policy.

    Neither baseline escalates: ``always_frontier`` has nowhere to escalate to, and
    ``always_open`` without escalation measures the open model's own quality (which is also what
    threshold calibration needs). ``complexity`` escalates failures, as in production. With
    ``target_open_fraction`` the complexity arm calibrates its threshold on the evaluated batch
    (see ``PipelineConfig.target_open_fraction``) instead of using the fixed ``threshold``.
    """
    return [
        PolicySpec("always_frontier", "always_frontier", None, False),
        PolicySpec("always_open", "always_open", None, False),
        PolicySpec("complexity", "complexity", threshold, True, target_open_fraction),
    ]
