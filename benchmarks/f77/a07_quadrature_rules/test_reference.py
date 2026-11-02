import math

import pytest

import target_module as tm


def test_trapz_single_panel_is_exact_for_lines():
    assert tm.trapz(lambda x: x, 0.0, 2.0, 1) == pytest.approx(2.0, rel=1e-9)


def test_trapz_parabola_four_panels():
    # h = 1/4, sum = 0.5*(0 + 1) + 1/16 + 1/4 + 9/16 = 1.375, times h
    assert tm.trapz(lambda x: x * x, 0.0, 1.0, 4) == pytest.approx(0.34375, rel=1e-9)


def test_trapz_constant_function_any_panel_count():
    assert tm.trapz(lambda x: 3.0, 0.0, 5.0, 7) == pytest.approx(15.0, rel=1e-9)


def test_trapz_reversed_limits_give_negative_area():
    assert tm.trapz(lambda x: x, 2.0, 0.0, 4) == pytest.approx(-2.0, rel=1e-9)


def test_trapz_calls_f_at_both_ends_and_each_interior_node():
    xs = []

    def f(x):
        xs.append(x)
        return x

    tm.trapz(f, 0.0, 1.0, 4)
    assert sorted(xs) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_simpsn_parabola_two_panels_is_exact():
    val, n = tm.simpsn(lambda x: x * x, 0.0, 1.0, 2)
    assert val == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert n == 2


def test_simpsn_cubic_is_exact():
    val, n = tm.simpsn(lambda x: x**3, 0.0, 1.0, 2)
    assert val == pytest.approx(0.25, rel=1e-9)
    assert n == 2


def test_simpsn_odd_panel_count_is_bumped_and_reported():
    val, n = tm.simpsn(lambda x: x**3, 0.0, 1.0, 3)
    assert n == 4
    assert val == pytest.approx(0.25, rel=1e-9)
    val1, n1 = tm.simpsn(lambda x: math.sin(x), 0.0, math.pi, 1)
    assert n1 == 2
    assert val1 == pytest.approx(2.0 * math.pi / 3.0, rel=1e-9)


def test_simpsn_quartic_weights_alternate_four_and_two():
    # f = x**4: two panels give 5/24, four panels give 2.40625 * 0.25 / 3
    v2, _ = tm.simpsn(lambda x: x**4, 0.0, 1.0, 2)
    v4, _ = tm.simpsn(lambda x: x**4, 0.0, 1.0, 4)
    assert v2 == pytest.approx(5.0 / 24.0, rel=1e-9)
    assert v4 == pytest.approx(2.40625 * 0.25 / 3.0, rel=1e-9)


def test_simpsn_sine_four_panels():
    val, n = tm.simpsn(math.sin, 0.0, math.pi, 4)
    assert n == 4
    assert val == pytest.approx(math.pi / 12.0 * (2.0 + 4.0 * math.sqrt(2.0)), rel=1e-9)


def test_simpsn_evaluates_f_n_plus_one_times_after_rounding_n_up():
    count = [0]

    def f(x):
        count[0] += 1
        return x

    tm.simpsn(f, 0.0, 1.0, 3)
    assert count[0] == 5


def test_quadex_parabola_converges_on_second_pass():
    val, nused = tm.quadex(lambda x: x * x, 0.0, 1.0, 1e-12, 1000, 0.0, -1)
    assert nused == 4
    assert val == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_quadex_sine_with_moderate_tolerance():
    val, nused = tm.quadex(math.sin, 0.0, math.pi, 1e-5, 1000, 0.0, -1)
    assert nused == 64
    assert val == pytest.approx(2.0, abs=1e-6)


def test_quadex_gives_up_when_nmax_is_too_small():
    val, nused = tm.quadex(math.sin, 0.0, math.pi, 1e-5, 16, 0.0, -1)
    s16, _ = tm.simpsn(math.sin, 0.0, math.pi, 16)
    assert nused == 0
    assert val == pytest.approx(s16, rel=1e-12)


def test_quadex_nmax_below_four_returns_first_estimate():
    val, nused = tm.quadex(math.sin, 0.0, math.pi, 1e-5, 2, 5.0, 5)
    assert nused == 0
    assert val == pytest.approx(2.0 * math.pi / 3.0, rel=1e-9)


def test_quadex_nmax_equal_to_needed_panels_is_accepted():
    val, nused = tm.quadex(lambda x: x * x, 0.0, 1.0, 1e-12, 4, 0.0, -1)
    assert nused == 4
    assert val == pytest.approx(1.0 / 3.0, rel=1e-9)
