import numpy as np
import pytest

from target_module import cbrtnw, cbrtv, dcbrt, reltol


def test_tolerance_is_ten_machine_epsilons():
    # D1MACH(4) = 2**-52, so the tolerance is 10 * 2**-52
    assert reltol() == pytest.approx(10.0 * 2.0**-52, rel=1e-12)
    assert reltol() == pytest.approx(2.220446049250313e-15, rel=1e-12)


def test_exact_cube_converges_on_first_test():
    # X starts at A = 1 where F(1) = 0 already
    assert cbrtnw(1.0, 0.0, 10, 0, 0) == (1.0, 1, 0)


@pytest.mark.parametrize(
    "a, root",
    [(8.0, 2.0), (27.0, 3.0), (-27.0, -3.0), (0.125, 0.5), (1.0e6, 100.0), (-0.001, -0.1)],
)
def test_cube_roots(a, root):
    x, iter_, ierr = cbrtnw(a, 0.0, 60, 0, 0)
    assert ierr == 0
    assert 1 <= iter_ <= 60
    assert x == pytest.approx(root, rel=1e-9)


def test_small_argument_starts_from_one():
    x, iter_, ierr = cbrtnw(1.0e-6, 0.0, 60, 0, 0)
    assert ierr == 0
    assert x == pytest.approx(1.0e-2, rel=1e-6)


def test_zero_argument_converges_linearly():
    # start X = 1, every step multiplies X by 2/3; the residual (2/3)**(3*m) first drops
    # below 10*2**-52 for m = 28, which is detected by the 29th test
    x, iter_, ierr = cbrtnw(0.0, 0.0, 60, 0, 0)
    assert (iter_, ierr) == (29, 0)
    assert x == pytest.approx((2.0 / 3.0) ** 28, rel=1e-6)


def test_failure_after_one_step_keeps_loop_counter_past_limit():
    # one Newton step from 8: (2*8**3 + 8) / (3*8**2) = 1032 / 192 = 5.375
    x, iter_, ierr = cbrtnw(8.0, 0.0, 1, 0, 0)
    assert (iter_, ierr) == (2, 1)
    assert x == pytest.approx(5.375, rel=1e-12)


def test_failure_after_two_steps():
    # 1000 -> 666.667 -> 444.4454166659166...
    x, iter_, ierr = cbrtnw(1000.0, 0.0, 2, 0, 0)
    assert (iter_, ierr) == (3, 1)
    assert x == pytest.approx(444.44541666591664, rel=1e-12)


def test_zero_trip_loop_leaves_counter_at_start_value():
    assert cbrtnw(5.0, 0.0, 0, 0, 0) == (5.0, 1, 1)
    assert cbrtnw(0.5, 0.0, 0, 0, 0) == (1.0, 1, 1)
    assert cbrtnw(5.0, 0.0, -3, 0, 0) == (5.0, 1, 1)


def test_input_value_of_outputs_is_ignored():
    first = cbrtnw(27.0, 99.0, 60, 77, 5)
    second = cbrtnw(27.0, 0.0, 60, 0, 0)
    assert first == second


@pytest.mark.parametrize("a, root", [(27.0, 3.0), (-8.0, -2.0), (0.001, 0.1), (125.0, 5.0)])
def test_function_wrapper(a, root):
    assert dcbrt(a) == pytest.approx(root, rel=1e-9)


def test_wrapper_returns_last_iterate_without_convergence():
    # 60 steps cannot reduce 1e60 to its cube root 1e20: every step only multiplies by 2/3
    assert dcbrt(1.0e60) == pytest.approx(1.0e60 * (2.0 / 3.0) ** 60, rel=1e-9)


def test_vector_version_fills_outputs_and_counts_failures():
    a = np.array([8.0, 1.0e60, 27.0, -1.0])
    x = np.zeros(4)
    failed = cbrtv(4, a, x)
    assert failed == 1
    assert x[0] == pytest.approx(2.0, rel=1e-9)
    assert x[1] == pytest.approx(1.0e60 * (2.0 / 3.0) ** 60, rel=1e-9)
    assert x[2] == pytest.approx(3.0, rel=1e-9)
    assert x[3] == pytest.approx(-1.0, rel=1e-9)


def test_vector_version_counts_every_failure():
    a = np.array([1.0e60, 8.0, -1.0e60, 27.0])
    x = np.zeros(4)
    assert cbrtv(4, a, x) == 2
    assert x[2] == pytest.approx(-1.0e60 * (2.0 / 3.0) ** 60, rel=1e-9)
    assert x[3] == pytest.approx(3.0, rel=1e-9)


@pytest.mark.parametrize("a", [3.0e9, 5.0e9])
def test_tolerance_scales_with_magnitude_of_a(a):
    # rounding in X**3 alone is about 5e-7 here, far above the unscaled tolerance 2.2e-15
    x, iter_, ierr = cbrtnw(a, 0.0, 60, 0, 0)
    assert ierr == 0
    assert x == pytest.approx(a ** (1.0 / 3.0), rel=1e-9)


def test_vector_version_respects_n():
    a = np.array([8.0, 27.0, 64.0])
    x = np.full(3, -5.0)
    assert cbrtv(2, a, x) == 0
    assert x[0] == pytest.approx(2.0, rel=1e-9)
    assert x[1] == pytest.approx(3.0, rel=1e-9)
    assert x[2] == -5.0
