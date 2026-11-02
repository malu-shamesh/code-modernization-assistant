import target_module as m
import numpy as np
import pytest


def arr(*values):
    return np.array(values, dtype=float)


def test_dasum_unit_stride():
    assert m.dasum(3, arr(1, -2, 3), 1) == pytest.approx(6.0, rel=1e-9)


def test_dasum_strided_skips_other_elements():
    dx = arr(1, -100, -2, 100, 3)
    assert m.dasum(3, dx, 2) == pytest.approx(6.0, rel=1e-9)


def test_dasum_degenerate_arguments_give_zero():
    assert m.dasum(0, arr(5), 1) == 0.0
    assert m.dasum(3, arr(1, 2, 3), 0) == 0.0
    assert m.dasum(3, arr(1, 2, 3), -1) == 0.0


def test_idamax_returns_first_of_equal_maxima_one_based():
    assert m.idamax(4, arr(1, -5, 3, 5), 1) == 2


def test_idamax_returns_element_number_not_storage_index():
    # elements seen with stride 2 are 1, -7, 2; the maximum is element 2 (storage index 3)
    assert m.idamax(3, arr(1, 0, -7, 0, 2, 0), 2) == 2


def test_idamax_edge_cases():
    assert m.idamax(1, arr(-3), 1) == 1
    assert m.idamax(0, arr(1), 1) == 0
    assert m.idamax(2, arr(1, 2), 0) == 0
    assert m.idamax(3, arr(0, 0, 0), 1) == 1
    assert m.idamax(3, arr(1, 2, 9), 1) == 3


def test_dnrm2_pythagorean_triple():
    assert m.dnrm2(2, arr(3, 4), 1) == pytest.approx(5.0, rel=1e-9)


def test_dnrm2_no_overflow_for_huge_entries():
    assert m.dnrm2(2, arr(3e200, 4e200), 1) == pytest.approx(5e200, rel=1e-9)


def test_dnrm2_no_underflow_for_tiny_entries():
    assert m.dnrm2(2, arr(3e-200, 4e-200), 1) == pytest.approx(5e-200, rel=1e-9, abs=0.0)


def test_dnrm2_order_independent_and_zero_entries_skipped():
    assert m.dnrm2(5, arr(0, 12, 0, 5, 0), 1) == pytest.approx(13.0, rel=1e-9)
    assert m.dnrm2(5, arr(0, 5, 0, 12, 0), 1) == pytest.approx(13.0, rel=1e-9)
    assert m.dnrm2(3, arr(0, 0, 0), 1) == 0.0


def test_dnrm2_stride_and_degenerate_arguments():
    assert m.dnrm2(2, arr(3, 99, 4), 2) == pytest.approx(5.0, rel=1e-9)
    assert m.dnrm2(0, arr(3), 1) == 0.0
    assert m.dnrm2(2, arr(3, 4), 0) == 0.0


def test_dnrm2_single_element_is_magnitude():
    assert m.dnrm2(1, arr(-7.5), 1) == pytest.approx(7.5, rel=1e-9)


def test_dnrmi_maximum_norm():
    assert m.dnrmi(3, arr(1, -9, 4), 1) == pytest.approx(9.0, rel=1e-9)


def test_dnrmi_with_stride_uses_storage_position_of_the_maximum():
    assert m.dnrmi(3, arr(2, 50, -8, 50, 3), 2) == pytest.approx(8.0, rel=1e-9)


def test_dnrmi_empty_vector():
    assert m.dnrmi(0, arr(1), 1) == 0.0


def test_two_element_pivot_search_and_single_element_sum():
    assert m.idamax(2, arr(1, -2), 1) == 2
    assert m.idamax(2, arr(3, 1), 1) == 1
    assert m.idamax(2, arr(-4, 4), 1) == 1
    assert m.dasum(1, arr(-4, 9), 1) == pytest.approx(4.0, rel=1e-9)
