import numpy as np
import pytest

from target_module import cdft, cpower, cquad, quadrt, qstats

REL = 1e-5
ABS = 1e-5


def cx(value):
    return pytest.approx(value, rel=REL, abs=ABS)


def roots(a, b, c):
    return quadrt(a, b, c, 5 + 5j, 5 + 5j, 9)


@pytest.fixture(autouse=True)
def reset_statistics_afterwards():
    # No reset before the test: the first test below must see the BLOCK DATA state.
    yield
    qstats(0, 0, 1)


def test_block_data_starts_both_counters_at_zero():
    assert qstats(0, 0, 0) == (0, 0)


def test_two_real_roots_negative_b():
    # D = 9 - 8 = 1, Q = -0.5*(-3 - 1) = 2, roots 2/1 and 2/2
    z1, z2, kind = roots(1.0, -3.0, 2.0)
    assert kind == 0
    assert z1 == cx(2.0)
    assert z2 == cx(1.0)


def test_two_real_roots_positive_b():
    # Q = -0.5*(3 + 1) = -2, roots -2/1 and 2/(-2)
    z1, z2, kind = roots(1.0, 3.0, 2.0)
    assert kind == 0
    assert z1 == cx(-2.0)
    assert z2 == cx(-1.0)


def test_real_roots_with_zero_b_and_negative_a():
    # SIGN(4, 0) = +4 so Q = -2, roots Q/A = 2 and C/Q = -2
    z1, z2, kind = roots(-1.0, 0.0, 4.0)
    assert kind == 0
    assert z1 == cx(2.0)
    assert z2 == cx(-2.0)


def test_real_roots_with_non_unit_leading_coefficient():
    # 2z**2 - 10z + 12 = 2(z - 2)(z - 3): Q = -0.5*(-10 - 2) = 6, roots Q/A = 3 and C/Q = 2
    z1, z2, kind = roots(2.0, -10.0, 12.0)
    assert kind == 0
    assert z1 == cx(3.0)
    assert z2 == cx(2.0)
    # 2z**2 + 10z + 12: Q = -0.5*(10 + 2) = -6, roots -3 and -2
    z1, z2, kind = roots(2.0, 10.0, 12.0)
    assert kind == 0
    assert z1 == cx(-3.0)
    assert z2 == cx(-2.0)


def test_double_root_scales_with_leading_coefficient():
    # 3z**2 + 6z + 3 = 3(z + 1)**2
    z1, z2, kind = roots(3.0, 6.0, 3.0)
    assert kind == 1
    assert z1 == cx(-1.0)
    assert z2 == cx(-1.0)


def test_double_root():
    z1, z2, kind = roots(2.0, -4.0, 2.0)
    assert kind == 1
    assert z1 == cx(1.0)
    assert z2 == cx(1.0)


def test_complex_pair_upper_root_first():
    z1, z2, kind = roots(1.0, 2.0, 5.0)
    assert kind == 2
    assert z1 == cx(-1.0 + 2.0j)
    assert z2 == cx(-1.0 - 2.0j)


def test_complex_pair_for_negative_leading_coefficient():
    # -2 z**2 - 8 = 0 gives z = +-2i; Z1 keeps the positive imaginary part
    z1, z2, kind = roots(-2.0, 0.0, -8.0)
    assert kind == 2
    assert z1 == cx(2.0j)
    assert z2 == cx(-2.0j)
    assert z1.imag > 0.0


def test_linear_and_degenerate_cases():
    z1, z2, kind = roots(0.0, 2.0, -6.0)
    assert (kind, z2) == (3, 0j)
    assert z1 == cx(3.0)
    assert roots(0.0, 0.0, 5.0) == (0j, 0j, -1)
    assert roots(0.0, 0.0, 0.0)[2] == -1


def test_call_statistics_count_every_call_and_complex_results():
    roots(1.0, -3.0, 2.0)
    roots(1.0, 2.0, 5.0)
    roots(0.0, 0.0, 1.0)
    roots(1.0, 0.0, 1.0)
    roots(2.0, -4.0, 2.0)
    assert qstats(0, 0, 0) == (5, 2)
    # reading without reset leaves the counters alone
    assert qstats(0, 0, 0) == (5, 2)
    assert qstats(0, 0, 1) == (5, 2)
    assert qstats(0, 0, 0) == (0, 0)


def test_complex_coefficients_pure_imaginary_roots():
    z1, z2, ierr = cquad(1 + 0j, 0j, 4 + 0j, 5 + 5j, 5 + 5j, 9)
    assert ierr == 0
    assert z1 == cx(2.0j)
    assert z2 == cx(-2.0j)


def test_complex_coefficients_general_case():
    # (z - (1+2i)) (z - (3-i)) = z**2 - (4+i) z + (5+5i); sqrt(-5-12i) = 2-3i
    z1, z2, ierr = cquad(1 + 0j, -(4 + 1j), 5 + 5j, 5 + 5j, 5 + 5j, 9)
    assert ierr == 0
    assert z1 == cx(3.0 - 1.0j)
    assert z2 == cx(1.0 + 2.0j)


def test_complex_leading_coefficient_and_zero_leading_coefficient():
    # 2i (z**2 - 1) = 0
    z1, z2, ierr = cquad(2j, 0j, -2j, 5 + 5j, 5 + 5j, 9)
    assert ierr == 0
    assert z1 == cx(1.0)
    assert z2 == cx(-1.0)
    assert cquad(0j, 1 + 0j, 2 + 0j, 5j, 5j, 9) == (0j, 0j, 1)


def test_dft_of_impulses_and_constant():
    xf = np.zeros(4, dtype=np.complex128)
    cdft(4, np.array([1, 0, 0, 0], dtype=np.complex128), xf, -1)
    assert xf == cx(np.ones(4, dtype=complex))
    cdft(4, np.ones(4, dtype=np.complex128), xf, -1)
    assert xf == cx(np.array([4, 0, 0, 0], dtype=complex))
    # a shifted impulse picks up a negative phase
    cdft(4, np.array([0, 1, 0, 0], dtype=np.complex128), xf, -1)
    assert xf == cx(np.array([1, -1j, -1, 1j]))


def test_dft_of_ramp_and_inverse_round_trip():
    x = np.array([1, 2, 3, 4], dtype=np.complex128)
    xf = np.zeros(4, dtype=np.complex128)
    cdft(4, x, xf, -1)
    assert xf == cx(np.array([10, -2 + 2j, -2, -2 - 2j]))
    back = np.zeros(4, dtype=np.complex128)
    cdft(4, xf, back, 1)
    assert back == cx(x)


def test_dft_zero_sign_is_an_unscaled_plain_sum():
    # ISIGN = 0 gives angle 0 everywhere and no 1/N factor (only ISIGN .GT. 0 divides)
    xf = np.zeros(4, dtype=np.complex128)
    cdft(4, np.array([1, 2, 3, 4], dtype=np.complex128), xf, 0)
    assert xf == cx(np.full(4, 10, dtype=complex))


def test_dft_length_three_uses_modular_angles():
    xf = np.zeros(3, dtype=np.complex128)
    cdft(3, np.array([0, 1, 0], dtype=np.complex128), xf, -1)
    w = -0.5 - 0.8660254037844386j
    assert xf == cx(np.array([1, w, w.conjugate()]))
    cdft(3, np.array([0, 0, 1], dtype=np.complex128), xf, -1)
    assert xf == cx(np.array([1, w.conjugate(), w]))


def test_dft_small_lengths_and_empty_transform():
    xf = np.full(3, 7 + 7j)
    cdft(1, np.array([2 + 3j]), xf, -1)
    assert xf[0] == cx(2 + 3j)
    cdft(2, np.array([5 + 0j, 3 + 0j]), xf, -1)
    assert xf[:2] == cx(np.array([8, 2], dtype=complex))
    assert xf[2] == 7 + 7j
    cdft(0, np.array([1 + 0j]), xf, -1)
    assert xf[0] == cx(8)


def test_power_sum_and_parseval():
    assert cpower(2, np.array([1 + 2j, 3 - 4j])) == pytest.approx(30.0, rel=REL)
    assert cpower(1, np.array([1 + 2j, 3 - 4j])) == pytest.approx(5.0, rel=REL)
    assert cpower(0, np.array([1 + 2j])) == 0.0
    x = np.array([1, 2, 3, 4], dtype=np.complex128)
    xf = np.zeros(4, dtype=np.complex128)
    cdft(4, x, xf, -1)
    assert cpower(4, xf) == pytest.approx(4.0 * cpower(4, x), rel=REL)
