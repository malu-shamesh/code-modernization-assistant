from __future__ import annotations

import pytest

from modernizer.evaluation import paired_counts, reduction_pct, sign_test_pvalue, wilson_interval


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (5, 10, 0.2366, 0.7634),
        (7, 10, 0.3968, 0.8922),
        (81, 263, 0.2553, 0.3662),  # the textbook worked example
        (27, 30, 0.7438, 0.9654),
    ],
)
def test_wilson_matches_known_values(k, n, low, high):
    got_low, got_high = wilson_interval(k, n)
    assert got_low == pytest.approx(low, abs=5e-5)
    assert got_high == pytest.approx(high, abs=5e-5)


def test_wilson_extremes_stay_inside_the_unit_interval():
    z2 = 1.959963984540054**2
    assert wilson_interval(0, 10) == pytest.approx((0.0, z2 / (10 + z2)))
    assert wilson_interval(10, 10) == pytest.approx((10 / (10 + z2), 1.0))
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_narrows_with_more_data():
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert large[1] - large[0] < small[1] - small[0]


@pytest.mark.parametrize("k", [-1, 11])
def test_wilson_rejects_impossible_counts(k):
    with pytest.raises(ValueError):
        wilson_interval(k, 10)


@pytest.mark.parametrize(
    ("a", "b", "p"),
    [
        (0, 5, 0.0625),
        (5, 0, 0.0625),
        (0, 0, 1.0),
        (3, 3, 1.0),
        (1, 4, 0.375),
        (2, 8, 0.109375),
        (0, 10, 2 / 1024),
        (1, 0, 1.0),
    ],
)
def test_sign_test_known_values(a, b, p):
    assert sign_test_pvalue(a, b) == pytest.approx(p)


def test_sign_test_is_symmetric_and_bounded():
    for a in range(8):
        for b in range(8):
            p = sign_test_pvalue(a, b)
            assert p == sign_test_pvalue(b, a)
            assert 0.0 < p <= 1.0


def test_sign_test_rejects_negative_counts():
    with pytest.raises(ValueError):
        sign_test_pvalue(-1, 2)


def test_paired_counts():
    baseline = [True, True, False, False, True]
    other = [True, False, True, False, False]
    assert paired_counts(baseline, other) == (2, 1)
    with pytest.raises(ValueError):
        paired_counts([True], [True, False])


def test_reduction_pct():
    assert reduction_pct(200.0, 150.0) == pytest.approx(25.0)
    assert reduction_pct(100.0, 130.0) == pytest.approx(-30.0)
    assert reduction_pct(0.0, 5.0) is None
