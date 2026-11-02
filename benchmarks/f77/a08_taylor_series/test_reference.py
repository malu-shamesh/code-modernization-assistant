import math

import numpy as np
import pytest

import target_module as tm


def test_texp_loose_tolerance_stops_after_six_terms():
    # 1 + 1 + 1/2 + 1/6 + 1/24 + 1/120 + 1/720; the k = 6 term is the first below tol*sum
    val, nterms = tm.texp(1.0, 1e-3, 0)
    assert nterms == 7
    assert val == pytest.approx(1957.0 / 720.0, rel=1e-12)


def test_texp_tight_tolerance_needs_fifteen_terms_after_the_first():
    val, nterms = tm.texp(1.0, 1e-12, 0)
    assert nterms == 16  # 1/15! is the first term below 1e-12 * e
    assert val == pytest.approx(math.e, rel=1e-9)


def test_texp_negative_argument_alternates():
    val, nterms = tm.texp(-1.0, 1e-3, 0)
    assert nterms == 8
    assert val == pytest.approx(103.0 / 280.0, rel=1e-12)


def test_texp_zero_argument_stops_at_the_first_term():
    assert tm.texp(0.0, 1e-6, 0) == (1.0, 2)


def test_texp_larger_argument():
    val, _ = tm.texp(5.0, 1e-12, 0)
    assert val == pytest.approx(math.exp(5.0), rel=1e-9)


def test_tsin_small_argument_term_count():
    # terms after x: 2.4e-2, 3.3e-4, 2.1e-6, 8.2e-9, 2.0e-11 -> stop at the fifth
    val, nterms = tm.tsin(math.pi / 6.0, 1e-9, 0)
    assert nterms == 6
    assert val == pytest.approx(0.5, abs=1e-10)


def test_tsin_zero():
    assert tm.tsin(0.0, 1e-9, 0) == (0.0, 2)


def test_tsin_reduces_large_arguments():
    val, _ = tm.tsin(100.0, 1e-12, 0)
    assert val == pytest.approx(-0.5063656411097588, abs=1e-9)


def test_tsin_negative_argument_reduces_with_the_sign_of_the_dividend():
    pos_val, pos_n = tm.tsin(100.0, 1e-12, 0)
    neg_val, neg_n = tm.tsin(-100.0, 1e-12, 0)
    assert neg_val == pytest.approx(-pos_val, abs=1e-9)
    assert neg_val == pytest.approx(0.5063656411097588, abs=1e-9)
    assert neg_n == pos_n


def test_tsin_full_turn_matches_the_base_angle():
    val, nterms = tm.tsin(2.0 * math.pi + math.pi / 6.0, 1e-9, 0)
    assert nterms == 6
    assert val == pytest.approx(0.5, abs=1e-9)


def test_tcos_values_and_term_count():
    val, nterms = tm.tcos(0.0, 1e-9, 0)
    assert val == pytest.approx(1.0, abs=1e-9)
    # sin(pi/2): the series term x**15/15! = 6.7e-10 is the first below 1e-9
    assert nterms == 8


@pytest.mark.parametrize(
    "x, expected",
    [
        (math.pi, -1.0),
        (math.pi / 3.0, 0.5),
        (-math.pi / 2.0, 0.0),
        (1.0, 0.5403023058681398),
    ],
)
def test_tcos_known_values(x, expected):
    val, _ = tm.tcos(x, 1e-12, 0)
    assert val == pytest.approx(expected, abs=1e-9)


def test_exptab_quarter_steps_hit_the_endpoint():
    tab = np.full(8, -1.0)
    n = tm.exptab(0.0, 1.0, 0.25, 1e-12, tab, 99)
    assert n == 5  # trip count INT((1 - 0 + 0.25)/0.25)
    assert tab[:5] == pytest.approx(np.exp([0.0, 0.25, 0.5, 0.75, 1.0]), rel=1e-9)
    assert tab[5:].tolist() == [-1.0, -1.0, -1.0]


def test_exptab_descending_range_with_negative_step():
    tab = np.zeros(4)
    n = tm.exptab(1.0, 0.0, -0.5, 1e-12, tab, 0)
    assert n == 3
    assert tab[:3] == pytest.approx([math.e, math.exp(0.5), 1.0], rel=1e-9)


@pytest.mark.parametrize(
    "x0, x1, dx, expected_n",
    [
        (1.0, 0.0, 0.25, 0),  # zero trips: nothing is written
        (2.0, 2.0, 1.0, 1),
        (0.0, 1.0, 0.4, 3),  # INT(1.4/0.4) = 3, the limit is not reached
        (0.0, 1.0, 0.35, 3),  # INT(1.35/0.35) = INT(3.857) = 3, NINT would give 4
        (-1.0, 1.0, 0.5, 5),
    ],
)
def test_exptab_trip_counts(x0, x1, dx, expected_n):
    tab = np.full(10, -5.0)
    n = tm.exptab(x0, x1, dx, 1e-10, tab, 7)
    assert n == expected_n
    assert (tab[expected_n:] == -5.0).all()
    for i in range(expected_n):
        assert tab[i] == pytest.approx(math.exp(x0 + i * dx), rel=1e-8)
