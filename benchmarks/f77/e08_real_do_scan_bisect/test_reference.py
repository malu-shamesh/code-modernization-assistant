import math

import pytest

import target_module as tm


def square_minus_two(x):
    return x * x - 2.0


def scan(f, a, b, h, tol=1e-6, maxit=60):
    return tm.rootsc(f, a, b, h, tol, maxit, 0.0, 0, 0.0)


def test_bisection_finds_sqrt2_after_a_positive_step_scan():
    root, nit, xend, alt = scan(square_minus_two, 0.0, 4.0, 1.0)
    assert alt == 0
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-6)
    assert nit == 20
    assert xend == 2.0


def test_scan_with_negative_step_and_bracket_order_reversed():
    root, nit, xend, alt = scan(square_minus_two, 4.0, 0.0, -1.0)
    assert alt == 0
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-6)
    assert nit == 20
    assert xend == 1.0


def test_sample_hits_zero_exactly_no_bisection():
    root, nit, xend, alt = scan(lambda x: x - 1.5, 0.0, 4.0, 0.5)
    assert (root, nit, xend, alt) == (1.5, 0, 1.5, 0)


def test_zero_at_the_starting_point_is_reported_at_a():
    root, nit, xend, alt = scan(lambda x: x, 0.0, 2.0, 0.5)
    assert (root, nit, xend, alt) == (0.0, 0, 0.5, 0)


def test_no_sign_change_returns_1_and_scan_variable_is_one_step_past_the_end():
    root, nit, xend, alt = scan(lambda x: x * x + 1.0, -1.0, 1.0, 0.5)
    assert (root, nit, alt) == (0.0, 0, 1)
    assert xend == 1.5


def test_zero_trip_scan_leaves_the_scan_variable_at_its_start_value():
    root, nit, xend, alt = scan(square_minus_two, 0.0, 0.25, 0.5)
    assert (root, nit, alt) == (0.0, 0, 1)
    assert xend == 0.5


def test_zero_step_is_rejected_without_calling_f():
    calls = []

    def f(x):
        calls.append(x)
        return x

    root, nit, xend, alt = scan(f, 1.0, 5.0, 0.0)
    assert (root, nit, xend, alt) == (0.0, 0, 1.0, 1)
    assert calls == []


def test_iteration_limit_returns_midpoint_of_current_bracket_with_return_2():
    root, nit, xend, alt = scan(square_minus_two, 0.0, 4.0, 1.0, tol=1e-6, maxit=3)
    assert alt == 2
    assert nit == 3
    assert xend == 2.0
    assert root == pytest.approx(1.4375, rel=1e-12)


def test_scan_uses_exactly_the_documented_samples():
    seen = []

    def f(x):
        seen.append(x)
        return 1.0

    scan(f, 0.0, 2.0, 0.5)
    assert seen == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_scan_trip_count_is_truncated_not_rounded():
    # INT((2.3 - 0.5 + 0.5) / 0.5) = INT(4.6) = 4, so 2.5 is never sampled
    seen = []

    def f(x):
        seen.append(x)
        return 1.0

    root, nit, xend, alt = scan(f, 0.0, 2.3, 0.5)
    assert seen == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert (root, nit, alt) == (0.0, 0, 1)
    assert xend == 2.5


@pytest.mark.parametrize(
    "a, b, h, expected",
    [
        (0.0, 1.0, 0.25, 5),
        (0.0, 0.9, 0.5, 2),
        (0.0, 1.0, 0.3, 4),
        (1.0, 0.0, 0.5, 0),
        (1.0, 0.0, -0.5, 3),
        (0.0, 0.0, 1.0, 1),
        (0.0, 0.9, 1.0, 1),
        (5.0, 4.0, 1.0, 0),
        (2.0, 10.0, 2.0, 5),
    ],
)
def test_isteps(a, b, h, expected):
    assert tm.isteps(a, b, h) == expected


def test_fixpt_converges_to_two():
    xfix, nit, alt = tm.fixpt(lambda x: 0.5 * x + 1.0, 0.0, 0.01, 100, 0.0, 0)
    assert alt == 0
    assert nit == 8
    assert xfix == pytest.approx(1.9921875, rel=1e-12)


def test_fixpt_divergence_return_1_keeps_the_offending_iterate():
    xfix, nit, alt = tm.fixpt(lambda x: 3.0 * x, 1.0, 1e-9, 100, 0.0, 0)
    assert alt == 1
    assert nit == 7
    assert xfix == pytest.approx(2187.0, rel=1e-12)


def test_fixpt_iteration_limit_return_2():
    xfix, nit, alt = tm.fixpt(lambda x: 0.5 * x + 1.0, 0.0, 1e-12, 5, 0.0, 0)
    assert alt == 2
    assert nit == 5
    assert xfix == pytest.approx(1.9375, rel=1e-12)


def test_fixpt_no_iterations_allowed():
    xfix, nit, alt = tm.fixpt(lambda x: x, 7.0, 1e-3, 0, 0.0, 0)
    assert (xfix, nit, alt) == (7.0, 0, 2)


def test_fixpt_tolerance_is_inclusive():
    xfix, nit, alt = tm.fixpt(lambda x: 0.5 * x + 1.0, 0.0, 0.5, 100, 0.0, 0)
    assert (xfix, nit, alt) == (1.5, 2, 0)
