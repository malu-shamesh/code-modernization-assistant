from __future__ import annotations

import pytest

from modernizer.evaluation import compare_results, sign_test_pvalue, wilson_interval

from .helpers import make_result, make_row


def rows(n, *, fail=(), tokens=100, cost=0.0, generated_fail=(), reference=True):
    return [
        make_row(
            f"m{i:02d}",
            reference=(i not in fail) if reference else None,
            generated=i not in generated_fail,
            tokens=tokens,
            cost=cost,
        )
        for i in range(n)
    ]


def test_token_and_cost_reductions_with_and_without_testgen():
    baseline = make_result(
        "base", rows(10, tokens=100, cost=1.0), testgen_tokens=50, testgen_cost=0.5
    )
    policy = make_result(
        "cheap", rows(10, tokens=60, cost=0.4), testgen_tokens=50, testgen_cost=0.5
    )
    c = compare_results(baseline, policy)

    assert c.token_reduction_pct == pytest.approx((1050 - 650) / 1050 * 100)
    assert c.token_reduction_pct_excluding_testgen == pytest.approx(40.0)
    assert c.cost_reduction_pct == pytest.approx((10.5 - 4.5) / 10.5 * 100)
    assert c.cost_reduction_pct_excluding_testgen == pytest.approx(60.0)


def test_more_expensive_policy_has_negative_reduction():
    baseline = make_result("base", rows(5, tokens=100))
    policy = make_result("big", rows(5, tokens=150))
    c = compare_results(baseline, policy)
    assert c.token_reduction_pct == pytest.approx(-50.0)
    assert "50.0% higher" in c.verdict


def test_zero_cost_baseline_gives_undefined_cost_reduction():
    c = compare_results(make_result("a", rows(5)), make_result("b", rows(5)))
    assert c.cost_reduction_pct is None
    assert c.cost_reduction_pct_excluding_testgen is None
    assert "not computable" in c.verdict


def test_discordant_pairs_deltas_intervals_and_p_value():
    baseline = make_result("base", rows(10))
    policy = make_result("cheap", rows(10, fail={0, 1}))
    ref = compare_results(baseline, policy).reference

    assert ref is not None
    assert (ref.baseline_pass, ref.policy_pass) == (10, 8)
    assert (ref.only_baseline_passes, ref.only_policy_passes) == (2, 0)
    assert ref.delta == pytest.approx(-0.2)
    assert ref.p_value == pytest.approx(sign_test_pvalue(2, 0))
    assert ref.baseline_interval == wilson_interval(10, 10)
    assert ref.policy_interval == wilson_interval(8, 10)


def test_policy_can_win_some_pairs_too():
    baseline = make_result("base", rows(10, fail={0, 1, 2}))
    policy = make_result("cheap", rows(10, fail={2, 3}))
    ref = compare_results(baseline, policy).reference
    assert (ref.only_baseline_passes, ref.only_policy_passes) == (1, 2)


def test_generated_and_held_out_metrics_are_kept_apart():
    baseline = make_result("base", rows(10))
    policy = make_result("cheap", rows(10, fail={4}, generated_fail=set()))
    c = compare_results(baseline, policy)
    assert c.generated.policy_pass == 10
    assert c.reference.policy_pass == 9


def test_small_sample_verdict_says_evidence_is_limited_and_never_claims_equality():
    c = compare_results(make_result("base", rows(10)), make_result("cheap", rows(10)))
    verdict = c.verdict
    assert "No detectable" in verdict
    assert "n = 10" in verdict
    assert "Evidence is limited" in verdict
    assert "[" in verdict and "Wilson" in verdict
    lowered = verdict.lower()
    assert "same pass rate" not in lowered
    assert "same quality" not in lowered
    assert "absence of evidence" in lowered


def test_discordant_pairs_make_the_evidence_limited_even_with_many_modules():
    c = compare_results(make_result("base", rows(40)), make_result("cheap", rows(40, fail={3})))
    assert "Evidence is limited" in c.verdict
    assert "n = 40 is below" not in c.verdict


def test_large_identical_sample_is_not_flagged_limited_but_stays_qualified():
    c = compare_results(make_result("base", rows(40)), make_result("cheap", rows(40)))
    assert "Evidence is limited" not in c.verdict
    assert "No detectable held-out-test pass-rate difference at n = 40" in c.verdict
    assert "not proof of equal quality" in c.verdict


def test_a_detected_difference_is_reported_as_such():
    baseline = make_result("base", rows(40))
    policy = make_result("cheap", rows(40, fail=set(range(10))))
    c = compare_results(baseline, policy)
    assert c.reference.p_value < 0.05
    assert "difference was detected" in c.verdict
    assert "lower" in c.verdict
    assert "No detectable held-out-test" not in c.verdict


def test_without_held_out_scores_only_the_generated_metric_is_compared():
    baseline = make_result("base", rows(10, reference=False), tests_run=True)
    policy = make_result("cheap", rows(10, reference=False))
    c = compare_results(baseline, policy)
    assert c.reference is None
    assert "Held-out tests were not run" in c.verdict


def test_dry_run_verdict_is_prefixed():
    c = compare_results(make_result("a", rows(5)), make_result("b", rows(5)), dry_run=True)
    assert c.verdict.startswith("DRY RUN")


def test_policies_must_cover_the_same_modules():
    with pytest.raises(ValueError, match="same modules"):
        compare_results(make_result("a", rows(5)), make_result("b", rows(4)))
    with pytest.raises(ValueError, match="zero"):
        compare_results(make_result("a", []), make_result("b", []))


def test_row_order_does_not_matter():
    baseline = make_result("a", rows(6, fail={1}))
    policy = make_result("b", list(reversed(rows(6, fail={1}))))
    ref = compare_results(baseline, policy).reference
    assert ref.discordant == 0


def test_result_counts_are_derived_from_rows():
    result = make_result(
        "x",
        [
            make_row("a", routed="open", final="frontier", generated=True, reference=False),
            make_row("b", routed="open", final="open", generated=True, reference=True),
            make_row("c", routed="frontier", generated=False, reference=False),
        ],
    )
    assert result.n == 3
    assert result.generated_pass == 2
    assert result.reference_pass == 1
    assert result.routed_open == 2
    assert result.final_open_successes == 1
    assert result.final_open_reference_successes == 1
    assert result.escalated == 1
    assert result.reference_test_fraction == pytest.approx((3 / 8 + 1.0 + 3 / 8) / 3)
    assert result.by_difficulty()["easy"]["n"] == 3


def test_verdict_names_static_validation_when_no_tests_ran():
    baseline = make_result("base", rows(10, reference=False), tests_run=False)
    policy = make_result("cheap", rows(10, reference=False), tests_run=False)
    verdict = compare_results(baseline, policy).verdict
    assert "Static validation only" in verdict
    assert "Generated tests" not in verdict
    assert "static-validation pass-rate" in verdict


def test_verdict_warns_when_the_reductions_came_with_fewer_passes():
    baseline = make_result("base", rows(10, tokens=100))
    policy = make_result("cheap", rows(10, fail={0}, tokens=50))
    verdict = compare_results(baseline, policy).verdict
    assert "passed fewer modules than base" in verdict
    assert "not obtained at equal quality" in verdict


def test_no_fewer_passes_warning_when_the_policy_matches_or_beats_the_baseline():
    baseline = make_result("base", rows(10, fail={0}))
    policy = make_result("cheap", rows(10))
    assert "fewer modules" not in compare_results(baseline, policy).verdict
