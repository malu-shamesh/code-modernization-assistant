"""Choosing the complexity threshold from an always-open run."""

from __future__ import annotations

from collections.abc import Sequence

from ..routing import calibrate_from_outcomes
from .corpus import BenchmarkCase
from .results import PolicyResult


def calibrate_open_threshold(
    cases: Sequence[BenchmarkCase], run_result_open: PolicyResult, *, min_pass_rate: float = 0.9
) -> float:
    """Largest difficulty threshold at which the open model's modules still pass often enough.

    ``run_result_open`` must come from an ``always_open`` policy without escalation, so every
    outcome is the open model's own. The outcome used is the pipeline's GENERATED-test success,
    which also exists in production, and deliberately not the held-out tests: tuning the
    threshold on the evaluation metric would leak the answer key into the policy.

    Only the modules in ``cases`` are used (pass the calibration half of ``split_calibration``).

    Caveats: if the result is then reported on the same modules, the savings are in-sample and
    optimistic; evaluate on the other half. The rule picks the most favourable cut among many on
    a small sample (see ``routing.calibrate_from_outcomes``). The router's forced-frontier rule
    for risky constructs still applies on top of the returned threshold. A return value of 0.0
    means no cut met the target.
    """
    spec = run_result_open.spec
    if spec.policy != "always_open" or spec.escalate:
        raise ValueError(
            "calibration needs an always_open run without escalation, so that every outcome "
            "is the open model's own"
        )
    if not run_result_open.generated_tests_run:
        raise ValueError(
            "calibration needs generated-test outcomes; that run skipped tests, so its "
            "'success' only means static validation"
        )
    if not cases:
        raise ValueError("no calibration cases")
    rows = {row.id: row for row in run_result_open.rows}
    missing = sorted(c.id for c in cases if c.id not in rows)
    if missing:
        raise ValueError(f"the open run has no result for: {', '.join(missing)}")
    selected = [rows[c.id] for c in cases]
    return calibrate_from_outcomes(
        [row.difficulty_score for row in selected],
        [row.generated_success for row in selected],
        min_pass_rate,
    )
