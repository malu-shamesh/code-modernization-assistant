import numpy as np
import pytest

import target_module as tm

# P(x) = 1 + 2x + 3x^2 + 4x^3
CUBIC = np.array([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "x, expected",
    [
        (2.0, 49.0),  # 1 + 4 + 12 + 32
        (-1.0, -2.0),  # 1 - 2 + 3 - 4
        (0.5, 3.25),  # 1 + 1 + 0.75 + 0.5
        (0.0, 1.0),
    ],
)
def test_horner_cubic(x, expected):
    assert tm.horner(3, CUBIC, x) == pytest.approx(expected, rel=1e-9)


def test_horner_degree_zero_is_the_constant():
    assert tm.horner(0, np.array([7.0]), 123.0) == pytest.approx(7.0, rel=1e-9)


def test_horner_ignores_coefficients_beyond_degree():
    c = np.array([1.0, 1.0, 1.0, 99.0, 99.0])
    assert tm.horner(2, c, 2.0) == pytest.approx(7.0, rel=1e-9)


def test_horner_degree_one_uses_both_coefficients():
    assert tm.horner(1, np.array([3.0, -2.0]), 5.0) == pytest.approx(-7.0, rel=1e-9)


@pytest.mark.parametrize(
    "x, p_expected, dp_expected",
    [
        (2.0, 49.0, 62.0),  # P' = 2 + 6x + 12x^2
        (-1.0, -2.0, 8.0),
        (0.0, 1.0, 2.0),
    ],
)
def test_polder_value_and_derivative(x, p_expected, dp_expected):
    p, dp = tm.polder(3, CUBIC, x, 0.0, 0.0)
    assert p == pytest.approx(p_expected, rel=1e-9)
    assert dp == pytest.approx(dp_expected, rel=1e-9)


def test_polder_constant_has_zero_derivative():
    p, dp = tm.polder(0, np.array([4.5]), 3.0, -1.0, -1.0)
    assert p == pytest.approx(4.5, rel=1e-9)
    assert dp == 0.0


@pytest.mark.parametrize(
    "r, quotient, remainder",
    [
        (1.0, [6.0, -5.0, 1.0], 0.0),  # (x-1)(x-2)(x-3) / (x-1) = x^2 - 5x + 6
        (2.0, [3.0, -4.0, 1.0], 0.0),
        (0.0, [11.0, -6.0, 1.0], -6.0),
        (4.0, [3.0, -2.0, 1.0], 6.0),  # P(4) = 64 - 96 + 44 - 6
    ],
)
def test_deflat_synthetic_division(r, quotient, remainder):
    c = np.array([-6.0, 11.0, -6.0, 1.0])
    q = np.zeros(3)
    rem = tm.deflat(3, c, r, q, 99.0)
    assert rem == pytest.approx(remainder, abs=1e-12)
    assert q == pytest.approx(quotient, rel=1e-9)


def test_deflat_linear_polynomial():
    # 5 + 2x divided by (x + 1): quotient 2, remainder 3
    q = np.zeros(1)
    rem = tm.deflat(1, np.array([5.0, 2.0]), -1.0, q, 0.0)
    assert q[0] == pytest.approx(2.0, rel=1e-9)
    assert rem == pytest.approx(3.0, rel=1e-9)


def test_deflat_leaves_coefficients_untouched():
    c = np.array([-6.0, 11.0, -6.0, 1.0])
    tm.deflat(3, c, 1.0, np.zeros(3), 0.0)
    assert c.tolist() == [-6.0, 11.0, -6.0, 1.0]
