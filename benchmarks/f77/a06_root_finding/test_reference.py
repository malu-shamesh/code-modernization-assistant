import math

import pytest

import target_module as tm


def square_minus_two(x):
    return x * x - 2.0


def square_minus_two_deriv(x):
    return 2.0 * x


@pytest.mark.parametrize(
    "tol, expected_nit, expected_x",
    [
        (1e-10, 5, math.sqrt(2.0)),
        (1e-3, 4, math.sqrt(2.0)),
        (0.1, 2, 17.0 / 12.0),  # x1 = 3/2, x2 = 17/12
    ],
)
def test_newton_sqrt2_iteration_counts(tol, expected_nit, expected_x):
    x, nit, ierr = tm.newton(square_minus_two, square_minus_two_deriv, 1.0, tol, 50, 0, 9)
    assert ierr == 0
    assert nit == expected_nit
    assert x == pytest.approx(expected_x, rel=1e-6 if tol > 1e-5 else 1e-9)


def test_newton_two_steps_from_one_are_exact_rationals():
    x, nit, ierr = tm.newton(square_minus_two, square_minus_two_deriv, 1.0, 1e-12, 2, 0, 9)
    assert (nit, ierr) == (2, 1)  # not converged: reports MAXIT, not MAXIT + 1
    assert x == pytest.approx(17.0 / 12.0, rel=1e-12)


def test_newton_zero_maxit_leaves_x_alone():
    x, nit, ierr = tm.newton(square_minus_two, square_minus_two_deriv, 1.0, 1e-8, 0, 7, 7)
    assert (x, nit, ierr) == (1.0, 0, 1)


def test_newton_flat_derivative_stops_at_once():
    x, nit, ierr = tm.newton(square_minus_two, square_minus_two_deriv, 0.0, 1e-8, 10, 0, 0)
    assert (x, nit, ierr) == (0.0, 1, 2)


def test_newton_linear_function_converges_on_second_step():
    x, nit, ierr = tm.newton(lambda t: 2.0 * t - 1.0, lambda t: 2.0, 10.0, 1e-12, 10, 0, 0)
    assert x == pytest.approx(0.5, rel=1e-12)
    assert (nit, ierr) == (2, 0)


def test_newton_tolerance_is_absolute_for_roots_below_one():
    # first step lands on 0.001 with |dx| = 0.001 <= tol * MAX(1, |x|) = 0.01
    x, nit, ierr = tm.newton(lambda t: t - 0.001, lambda t: 1.0, 0.0, 1e-2, 10, 0, 0)
    assert x == pytest.approx(0.001, rel=1e-12)
    assert (nit, ierr) == (1, 0)


def test_newton_tolerance_is_relative_for_large_roots():
    # the deliberately wrong derivative 2 halves the error each step: steps are 8, 4, 2, 1;
    # the step 1 passes only against tol * MAX(1, |x|) = 1.001, not against tol = 1e-3
    x, nit, ierr = tm.newton(lambda t: t - 1000.0, lambda t: 2.0, 1016.0, 1e-3, 50, 0, 0)
    assert (x, nit, ierr) == (1001.0, 4, 0)


def test_newton_cube_root():
    x, _, ierr = tm.newton(lambda t: t**3 - 27.0, lambda t: 3.0 * t * t, 5.0, 1e-12, 50, 0, 0)
    assert ierr == 0
    assert x == pytest.approx(3.0, rel=1e-9)


def test_newton_evaluates_f_and_df_once_per_step():
    calls = {"f": 0, "df": 0}

    def f(t):
        calls["f"] += 1
        return t * t - 2.0

    def df(t):
        calls["df"] += 1
        return 2.0 * t

    _, nit, _ = tm.newton(f, df, 1.0, 1e-10, 50, 0, 0)
    assert nit == 5
    assert calls == {"f": 5, "df": 5}


def test_bisect_sqrt2_to_loose_tolerance():
    root, nit, alt = tm.bisect(square_minus_two, 1.0, 2.0, 1e-3, 50, 0.0, 0)
    assert alt == 0
    assert nit == 10  # half-width 2**-k first drops below 1e-3 at k = 10
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-3)


def test_bisect_cosine_needs_34_halvings_for_1e_minus_10():
    root, nit, alt = tm.bisect(math.cos, 1.0, 2.0, 1e-10, 100, 0.0, 0)
    assert alt == 0
    assert nit == 34
    assert root == pytest.approx(math.pi / 2.0, abs=1e-10)


def test_bisect_exact_midpoint_root_stops_immediately():
    root, nit, alt = tm.bisect(lambda t: t - 0.5, 0.0, 1.0, 1e-9, 50, 0.0, 0)
    assert (root, nit, alt) == (0.5, 1, 0)


def test_bisect_second_midpoint_exact_root():
    root, nit, alt = tm.bisect(lambda t: t - 2.5, 0.0, 10.0, 1e-9, 50, 0.0, 0)
    assert (root, nit, alt) == (2.5, 2, 0)


def test_bisect_iteration_limit_reports_maxit_and_last_midpoint():
    root, nit, alt = tm.bisect(square_minus_two, 1.0, 2.0, 1e-12, 5, 0.0, 0)
    # midpoints: 1.5, 1.25, 1.375, 1.4375, 1.40625
    assert (root, nit, alt) == (1.40625, 5, 0)


def test_bisect_decreasing_function():
    root, _, alt = tm.bisect(lambda t: 3.0 - t, 0.0, 10.0, 1e-6, 100, 0.0, 0)
    assert alt == 0
    assert root == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize(
    "f, a, b",
    [
        (square_minus_two, 2.0, 3.0),  # both ends positive
        (lambda t: t * t + 1.0, -1.0, 1.0),
        (lambda t: -1.0 - t, 0.0, 4.0),  # both ends negative
    ],
)
def test_bisect_without_sign_change_takes_the_alternate_return(f, a, b):
    result = tm.bisect(f, a, b, 1e-6, 50, -99.0, 5)
    assert result[-1] == 1
    assert result[1] == 0


def test_bisect_evaluates_endpoints_then_once_per_halving():
    calls = []

    def f(t):
        calls.append(t)
        return t - 2.5

    _, nit, _ = tm.bisect(f, 0.0, 10.0, 1e-9, 50, 0.0, 0)
    assert nit == 2
    assert calls == [0.0, 10.0, 5.0, 2.5]
