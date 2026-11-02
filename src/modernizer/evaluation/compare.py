"""Paired comparison of two policies on the same modules, with a deliberately cautious verdict.

The verdict never says "same pass rate". Equal observed rates on a few dozen modules are
compatible with real differences of many points, and a non-significant test is absence of
evidence, not evidence of equivalence. The wording always carries n, the intervals and the
discordant counts, and says when the evidence is limited.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .results import Comparison, ModuleRow, PassComparison, PolicyResult
from .stats import paired_counts, reduction_pct, sign_test_pvalue, wilson_interval

ALPHA = 0.05
MIN_ADEQUATE_N = 30
DRY_RUN_PREFIX = "DRY RUN (oracle replaying the answer key, not a model): "


def _pass_comparison(
    metric: str,
    baseline_rows: Sequence[ModuleRow],
    policy_rows: Sequence[ModuleRow],
    flag: Callable[[ModuleRow], bool],
) -> PassComparison:
    base = [flag(r) for r in baseline_rows]
    other = [flag(r) for r in policy_rows]
    only_base, only_policy = paired_counts(base, other)
    n = len(base)
    base_pass, policy_pass = sum(base), sum(other)
    return PassComparison(
        metric=metric,
        n=n,
        baseline_pass=base_pass,
        policy_pass=policy_pass,
        baseline_rate=base_pass / n,
        policy_rate=policy_pass / n,
        delta=(policy_pass - base_pass) / n,
        baseline_interval=wilson_interval(base_pass, n),
        policy_interval=wilson_interval(policy_pass, n),
        only_baseline_passes=only_base,
        only_policy_passes=only_policy,
        p_value=sign_test_pvalue(only_base, only_policy),
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _interval(interval: tuple[float, float]) -> str:
    return f"[{_pct(interval[0])}, {_pct(interval[1])}]"


def _pass_sentence(label: str, c: PassComparison, baseline: str, policy: str) -> str:
    return (
        f"{label}: {policy} {c.policy_pass}/{c.n} ({_pct(c.policy_rate)}, 95% Wilson "
        f"{_interval(c.policy_interval)}) vs {baseline} {c.baseline_pass}/{c.n} "
        f"({_pct(c.baseline_rate)}, {_interval(c.baseline_interval)}); discordant modules: "
        f"{c.only_baseline_passes} only {baseline} passes, {c.only_policy_passes} only "
        f"{policy} passes; exact sign test p = {c.p_value:.3f}."
    )


def _conclusion(c: PassComparison, label: str, policy: str, baseline: str) -> str:
    if c.p_value < ALPHA and c.delta != 0:
        direction = "lower" if c.delta < 0 else "higher"
        return (
            f"A {label} pass-rate difference was detected ({policy} {direction}, p = "
            f"{c.p_value:.3f}); this is one uncorrected comparison on one corpus."
        )
    width = max(
        c.baseline_interval[1] - c.baseline_interval[0],
        c.policy_interval[1] - c.policy_interval[0],
    )
    return (
        f"No detectable {label} pass-rate difference at n = {c.n} (p = {c.p_value:.3f}). "
        f"This is absence of evidence, not proof of equal quality: the wider 95% interval "
        f"spans {width * 100:.0f} percentage points."
    )


def _limits(n: int, comparisons: Sequence[PassComparison]) -> str | None:
    reasons: list[str] = []
    if n < MIN_ADEQUATE_N:
        reasons.append(f"n = {n} is below {MIN_ADEQUATE_N}")
    discordant = sum(c.discordant for c in comparisons)
    if discordant:
        reasons.append("some modules have different outcomes under the two policies")
    if not reasons:
        return None
    return "Evidence is limited: " + " and ".join(reasons) + "."


def _change(label: str, pct: float | None) -> str:
    if pct is None:
        return f"{label}: not computable (baseline is zero)"
    direction = "lower" if pct >= 0 else "higher"
    return f"{label}: {abs(pct):.1f}% {direction}"


def build_verdict(
    baseline: str,
    policy: str,
    n: int,
    *,
    generated: PassComparison,
    reference: PassComparison | None,
    reductions: dict[str, float | None],
    dry_run: bool = False,
    generated_tests_ran: bool = True,
) -> str:
    # Without test runs the "generated" outcome is static validation only; say so in the text.
    label, kind = (
        ("Generated tests", "generated-test")
        if generated_tests_ran
        else ("Static validation only (no tests were run)", "static-validation")
    )
    parts = [f"{policy} vs {baseline} on {n} modules."]
    if reference is not None:
        parts.append(_pass_sentence("Held-out tests", reference, baseline, policy))
        parts.append(_conclusion(reference, "held-out-test", policy, baseline))
    parts.append(_pass_sentence(label, generated, baseline, policy))
    if reference is None:
        parts.append(_conclusion(generated, kind, policy, baseline))
        parts.append("Held-out tests were not run, so the independent check is missing.")
    measured = [c for c in (reference, generated) if c is not None]
    limits = _limits(n, measured)
    if limits:
        parts.append(limits)
    if any(c.policy_pass < c.baseline_pass for c in measured):
        parts.append(
            f"{policy} passed fewer modules than {baseline}, so the reductions below were "
            "not obtained at equal quality."
        )
    parts.append(
        "Versus the baseline, "
        + "; ".join(
            [
                _change("tokens, all calls", reductions["tokens"]),
                _change("tokens excluding test generation", reductions["tokens_excl"]),
                _change("cost, all calls", reductions["cost"]),
                _change("cost excluding test generation", reductions["cost_excl"]),
            ]
        )
        + ". Cost depends on the price table in use; tokens do not."
    )
    text = " ".join(parts)
    return DRY_RUN_PREFIX + text if dry_run else text


def compare_results(
    baseline: PolicyResult, policy: PolicyResult, *, dry_run: bool = False
) -> Comparison:
    """Compare ``policy`` against ``baseline``; both must cover exactly the same module ids."""
    base_rows = sorted(baseline.rows, key=lambda r: r.id)
    policy_rows = sorted(policy.rows, key=lambda r: r.id)
    if [r.id for r in base_rows] != [r.id for r in policy_rows]:
        raise ValueError("the two policies were not run on the same modules")
    if not base_rows:
        raise ValueError("cannot compare policies on zero modules")

    generated = _pass_comparison("generated", base_rows, policy_rows, lambda r: r.generated_success)
    measured = all(r.reference_success is not None for r in (*base_rows, *policy_rows))
    reference = (
        _pass_comparison("reference", base_rows, policy_rows, lambda r: bool(r.reference_success))
        if measured
        else None
    )
    reductions = {
        "tokens": reduction_pct(baseline.tokens.total, policy.tokens.total),
        "tokens_excl": reduction_pct(
            baseline.tokens_excluding_testgen.total, policy.tokens_excluding_testgen.total
        ),
        "cost": reduction_pct(baseline.cost_usd_total, policy.cost_usd_total),
        "cost_excl": reduction_pct(
            baseline.cost_usd_excluding_testgen, policy.cost_usd_excluding_testgen
        ),
    }
    n = len(base_rows)
    return Comparison(
        baseline=baseline.spec.name,
        policy=policy.spec.name,
        n=n,
        token_reduction_pct=reductions["tokens"],
        token_reduction_pct_excluding_testgen=reductions["tokens_excl"],
        cost_reduction_pct=reductions["cost"],
        cost_reduction_pct_excluding_testgen=reductions["cost_excl"],
        generated=generated,
        reference=reference,
        verdict=build_verdict(
            baseline.spec.name,
            policy.spec.name,
            n,
            generated=generated,
            reference=reference,
            reductions=reductions,
            dry_run=dry_run,
            generated_tests_ran=baseline.generated_tests_run and policy.generated_tests_run,
        ),
    )
