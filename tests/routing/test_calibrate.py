from __future__ import annotations

import random

import pytest

from modernizer.routing import calibrate_from_outcomes, calibrate_threshold, open_fraction


def test_open_fraction_basic_and_empty() -> None:
    scores = [0.1, 0.2, 0.3, 0.4]
    assert open_fraction(scores, 0.25) == 0.5
    assert open_fraction(scores, 0.3) == 0.75
    assert open_fraction(scores, 0.05) == 0.0
    assert open_fraction(scores, 1.0) == 1.0
    assert open_fraction([], 0.5) == 0.0


@pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_calibrate_threshold_hits_target_without_ties(target: float) -> None:
    scores = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    scores = scores[:4] if target == 0.25 else scores
    threshold = calibrate_threshold(scores, target)
    assert open_fraction(scores, threshold) == pytest.approx(target, abs=1 / len(scores))


def test_target_zero_routes_nothing_open_even_for_zero_scores() -> None:
    scores = [0.0, 0.0, 0.3]
    assert open_fraction(scores, calibrate_threshold(scores, 0.0)) == 0.0


def test_target_one_routes_everything_open() -> None:
    scores = [0.9, 0.1, 0.5]
    assert open_fraction(scores, calibrate_threshold(scores, 1.0)) == 1.0


def test_ties_choose_closest_fraction_preferring_fewer_open() -> None:
    scores = [0.2, 0.2, 0.2, 0.2, 0.8, 0.8]
    assert open_fraction(scores, calibrate_threshold(scores, 0.5)) == pytest.approx(4 / 6)
    assert open_fraction(scores, calibrate_threshold(scores, 0.35)) == pytest.approx(4 / 6)
    assert open_fraction(scores, calibrate_threshold(scores, 0.3)) == 0.0
    equidistant = [0.2, 0.2, 0.8, 0.8]
    assert open_fraction(equidistant, calibrate_threshold(equidistant, 0.25)) == 0.0
    assert open_fraction([0.5, 0.5], calibrate_threshold([0.5, 0.5], 0.5)) == 0.0


def test_single_and_all_equal_scores() -> None:
    assert open_fraction([0.4], calibrate_threshold([0.4], 1.0)) == 1.0
    assert open_fraction([0.4], calibrate_threshold([0.4], 0.0)) == 0.0


def test_empty_and_invalid_inputs() -> None:
    assert calibrate_threshold([], 0.5) == 0.0
    with pytest.raises(ValueError):
        calibrate_threshold([0.1], 1.5)
    with pytest.raises(ValueError):
        calibrate_threshold([0.1], -0.1)
    with pytest.raises(ValueError):
        calibrate_threshold([float("nan")], 0.5)


def test_calibrate_threshold_random_fractions_are_close() -> None:
    rng = random.Random(7)
    scores = [rng.random() for _ in range(400)]
    for target in (0.1, 0.33, 0.5, 0.9):
        got = open_fraction(scores, calibrate_threshold(scores, target))
        assert abs(got - target) <= 1 / len(scores)


def test_calibrate_from_outcomes_picks_largest_qualifying_threshold() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ok = [True, True, True, True, False, False]
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=1.0) == 0.4
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.8) == 0.5
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.7) == 0.5
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.6) == 0.6
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.0) == 0.6


def test_calibrate_from_outcomes_is_order_independent() -> None:
    scores = [0.6, 0.1, 0.5, 0.3, 0.2, 0.4]
    ok = [False, True, False, True, True, True]
    assert calibrate_from_outcomes(scores, ok, 1.0) == 0.4


def test_calibrate_from_outcomes_none_qualifies_returns_zero() -> None:
    assert calibrate_from_outcomes([0.1, 0.2], [False, False]) == 0.0
    assert calibrate_from_outcomes([], []) == 0.0


def test_calibrate_from_outcomes_ties_evaluated_as_a_group() -> None:
    scores = [0.2, 0.3, 0.3, 0.3]
    ok = [True, True, False, True]
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=1.0) == 0.2
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.75) == 0.3


def test_calibrate_from_outcomes_exact_boundary_counts_as_passing() -> None:
    scores = [i / 100 for i in range(1, 21)]
    ok = [True] * 19 + [False]
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.95) == 0.2
    ok_early_fail = [False] + [True] * 19
    assert calibrate_from_outcomes(scores, ok_early_fail, min_pass_rate=0.96) == 0.0


def test_calibrate_from_outcomes_can_recover_after_early_dip() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    ok = [False, True, True, True, True]
    assert calibrate_from_outcomes(scores, ok, min_pass_rate=0.8) == 0.5


def test_calibrate_from_outcomes_validation() -> None:
    with pytest.raises(ValueError):
        calibrate_from_outcomes([0.1], [True, False])
    with pytest.raises(ValueError):
        calibrate_from_outcomes([0.1], [True], min_pass_rate=1.2)
    with pytest.raises(ValueError):
        calibrate_from_outcomes([float("nan")], [True])


def test_docstring_documents_small_sample_caveat() -> None:
    assert "caveat" in (calibrate_from_outcomes.__doc__ or "").lower()
