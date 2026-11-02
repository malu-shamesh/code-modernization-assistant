from __future__ import annotations

import pytest

from modernizer.evaluation import BenchmarkCase, calibrate_open_threshold
from modernizer.routing import calibrate_from_outcomes

from .helpers import make_result, make_row


def case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(case_id, "easy", (), "", "", "", None)  # type: ignore[arg-type]


def open_run(*, tests_run: bool = True):
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    generated = [True, True, True, False, False, True]
    # Held-out outcomes are the opposite on purpose: calibration must ignore them.
    reference = [not g for g in generated]
    rows = [
        make_row(f"m{i}", score=s, routed="open", generated=g, reference=r, difficulty="easy")
        for i, (s, g, r) in enumerate(zip(scores, generated, reference, strict=True))
    ]
    return make_result("always_open", rows, escalate=False, tests_run=tests_run)


def test_threshold_uses_generated_success_not_held_out_tests():
    result = open_run()
    cases = [case(f"m{i}") for i in range(6)]
    expected = calibrate_from_outcomes(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [True, True, True, False, False, True], 0.9
    )
    assert expected == 0.3
    assert calibrate_open_threshold(cases, result, min_pass_rate=0.9) == expected


def test_lower_target_allows_a_higher_threshold():
    result = open_run()
    cases = [case(f"m{i}") for i in range(6)]
    assert calibrate_open_threshold(cases, result, min_pass_rate=0.5) >= 0.6


def test_only_the_given_cases_are_used():
    result = open_run()
    first_three = [case(f"m{i}") for i in range(3)]
    assert calibrate_open_threshold(first_three, result, min_pass_rate=1.0) == 0.3


def test_run_must_be_an_open_only_run_without_escalation():
    cases = [case("m0")]
    rows = [make_row("m0", routed="open")]
    with pytest.raises(ValueError, match="always_open"):
        calibrate_open_threshold(cases, make_result("complexity", rows, escalate=True))
    with pytest.raises(ValueError, match="always_open"):
        calibrate_open_threshold(cases, make_result("always_frontier", rows))
    with pytest.raises(ValueError, match="always_open"):
        calibrate_open_threshold(cases, make_result("always_open", rows, escalate=True))


def test_run_without_tests_is_rejected():
    with pytest.raises(ValueError, match="skipped tests"):
        calibrate_open_threshold([case("m0")], open_run(tests_run=False))


def test_cases_missing_from_the_run_are_an_error():
    with pytest.raises(ValueError, match="zz"):
        calibrate_open_threshold([case("zz")], open_run())
    with pytest.raises(ValueError, match="no calibration cases"):
        calibrate_open_threshold([], open_run())
