import math

import numpy as np
import pytest

import target_module
from target_module import fittrn, linreg, rssq


def fit(xs, ys):
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    return linreg(len(xs), x, y, 0.0, 0.0, 0.0, 0)


def test_noisy_line_matches_hand_computed_sums():
    # x mean 3, y mean 6.02, Sxy = 19.9, Sxx = 10, Syy = 39.708
    a, b, r2, ierr = fit([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1])
    assert ierr == 0
    assert b == pytest.approx(1.99, rel=1e-9)
    assert a == pytest.approx(0.05, rel=1e-9)
    assert r2 == pytest.approx(19.9**2 / (10.0 * 39.708), rel=1e-9)


def test_symmetric_data_gives_r2_of_sixty_four_percent():
    a, b, r2, ierr = fit([1, 2, 3, 4], [1, 3, 2, 4])
    assert ierr == 0
    assert a == pytest.approx(0.5, rel=1e-9)
    assert b == pytest.approx(0.8, rel=1e-9)
    assert r2 == pytest.approx(0.64, rel=1e-9)


def test_exact_lines_have_unit_r2():
    a, b, r2, ierr = fit([0, 1, 2, 3], [-2, 1, 4, 7])
    assert ierr == 0
    assert a == pytest.approx(-2.0, rel=1e-9)
    assert b == pytest.approx(3.0, rel=1e-9)
    assert r2 == pytest.approx(1.0, rel=1e-9)
    a, b, r2, ierr = fit([1, 2, 3], [8, 6, 4])
    assert a == pytest.approx(10.0, rel=1e-9)
    assert b == pytest.approx(-2.0, rel=1e-9)
    assert r2 == pytest.approx(1.0, rel=1e-9)


def test_constant_y_is_a_flat_exact_fit():
    assert fit([1, 2, 3], [4, 4, 4]) == (4.0, 0.0, 1.0, 0)


def test_too_few_points():
    assert fit([1.0], [2.0]) == (0.0, 0.0, 0.0, 1)
    assert fit([], []) == (0.0, 0.0, 0.0, 1)


def test_two_points_are_enough_for_a_fit():
    # the guard is N .LT. 2, so exactly two points give an exact line
    a, b, r2, ierr = fit([1.0, 3.0], [2.0, 8.0])
    assert ierr == 0
    assert a == pytest.approx(-1.0, rel=1e-9)
    assert b == pytest.approx(3.0, rel=1e-9)
    assert r2 == pytest.approx(1.0, rel=1e-9)


def test_vertical_line_is_rejected():
    assert fit([2.5, 2.5, 2.5], [1.0, 2.0, 3.0]) == (0.0, 0.0, 0.0, 2)


def test_linreg_uses_only_first_n_points():
    x = np.array([0.0, 1.0, 2.0, 99.0])
    y = np.array([1.0, 3.0, 5.0, -50.0])
    a, b, r2, ierr = linreg(3, x, y, 0.0, 0.0, 0.0, 0)
    assert ierr == 0
    assert a == pytest.approx(1.0, rel=1e-9)
    assert b == pytest.approx(2.0, rel=1e-9)
    assert r2 == pytest.approx(1.0, rel=1e-9)


def test_fittrn_with_squared_abscissa():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([3.0, 9.0, 19.0, 33.0])  # y = 1 + 2 x^2
    xt = np.zeros(4)
    a, b, r2, ierr = fittrn(4, x, y, lambda v: v * v, xt, 0.0, 0.0, 0.0, 0)
    assert ierr == 0
    assert a == pytest.approx(1.0, rel=1e-9)
    assert b == pytest.approx(2.0, rel=1e-9)
    assert r2 == pytest.approx(1.0, rel=1e-9)
    assert xt.tolist() == [1.0, 4.0, 9.0, 16.0]


def test_fittrn_with_square_root_abscissa():
    x = np.array([1.0, 4.0, 9.0, 16.0])
    y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 1 + 2 sqrt(x)
    a, b, r2, ierr = fittrn(4, x, y, math.sqrt, np.zeros(4), 0.0, 0.0, 0.0, 0)
    assert ierr == 0
    assert a == pytest.approx(1.0, rel=1e-9)
    assert b == pytest.approx(2.0, rel=1e-9)
    assert r2 == pytest.approx(1.0, rel=1e-9)


def test_fittrn_forwards_the_error_code():
    x = np.array([1.0, -1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    result = fittrn(3, x, y, abs, np.zeros(3), 5.0, 5.0, 5.0, 9)
    assert result == (0.0, 0.0, 0.0, 2)


def test_rssq_of_the_fitted_line_is_syy_minus_b_sxy():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    assert rssq(5, x, y, 0.05, 1.99) == pytest.approx(0.107, rel=1e-9)


def test_rssq_of_other_lines():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    assert rssq(4, x, y, 0.5, 0.8) == pytest.approx(1.8, rel=1e-9)
    assert rssq(4, x, y, 0.0, 1.0) == pytest.approx(2.0, rel=1e-9)
    assert rssq(0, x, y, 0.0, 1.0) == 0.0


def test_module_exposes_the_fortran_units():
    for name in ("linreg", "fittrn", "rssq"):
        assert callable(getattr(target_module, name))
