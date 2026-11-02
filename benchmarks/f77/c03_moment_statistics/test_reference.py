import math

import numpy as np
import pytest

import target_module
from target_module import dcovar, minmax, moment


def stats(values):
    x = np.array(values, dtype=np.float64)
    return moment(len(values), x, 0.0, 0.0, 0.0, 0)


def test_classic_eight_point_sample():
    # mean 5, sum of squared deviations 32, sample variance 32/7
    xmean, xvar, xsd, ierr = stats([2, 4, 4, 4, 5, 5, 7, 9])
    assert xmean == pytest.approx(5.0, rel=1e-9)
    assert xvar == pytest.approx(32.0 / 7.0, rel=1e-9)
    assert xsd == pytest.approx(math.sqrt(32.0 / 7.0), rel=1e-9)
    assert ierr == 0


def test_one_to_five():
    xmean, xvar, xsd, ierr = stats([1, 2, 3, 4, 5])
    assert (xmean, ierr) == (3.0, 0)
    assert xvar == pytest.approx(2.5, rel=1e-9)
    assert xsd == pytest.approx(math.sqrt(2.5), rel=1e-9)


def test_large_offset_keeps_precision():
    base = 1.0e9
    xmean, xvar, xsd, ierr = stats([base + 4, base + 7, base + 13, base + 16])
    assert xmean == pytest.approx(base + 10.0, rel=1e-12)
    assert xvar == pytest.approx(30.0, rel=1e-9)
    assert xsd == pytest.approx(math.sqrt(30.0), rel=1e-9)
    assert ierr == 0


def test_negative_values():
    xmean, xvar, xsd, ierr = stats([-3, -1, -2])
    assert xmean == pytest.approx(-2.0, rel=1e-9)
    assert xvar == pytest.approx(1.0, rel=1e-9)
    assert xsd == pytest.approx(1.0, rel=1e-9)
    assert ierr == 0


def test_constant_sample_has_zero_variance():
    assert stats([4.5, 4.5, 4.5]) == (4.5, 0.0, 0.0, 0)


def test_single_element_reports_undefined_variance():
    assert stats([7.25]) == (7.25, 0.0, 0.0, 2)


def test_empty_sample_reports_error_and_zeros():
    assert stats([]) == (0.0, 0.0, 0.0, 1)


def test_moment_uses_only_first_n_elements():
    x = np.array([1.0, 2.0, 3.0, 1000.0, -1000.0])
    xmean, xvar, xsd, ierr = moment(3, x, 0.0, 0.0, 0.0, 0)
    assert (xmean, ierr) == (2.0, 0)
    assert xvar == pytest.approx(1.0, rel=1e-9)
    assert xsd == pytest.approx(1.0, rel=1e-9)


def test_covariance_of_perfectly_correlated_vectors():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert dcovar(5, x, 2.0 * x) == pytest.approx(5.0, rel=1e-9)
    assert dcovar(5, x, np.array([5.0, 4.0, 3.0, 2.0, 1.0])) == pytest.approx(-2.5, rel=1e-9)


def test_covariance_with_itself_is_the_variance():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert dcovar(8, x, x) == pytest.approx(32.0 / 7.0, rel=1e-9)


def test_covariance_short_inputs_and_partial_n():
    x = np.array([1.0, 2.0, 3.0, 50.0])
    y = np.array([2.0, 4.0, 6.0, -50.0])
    assert dcovar(1, x, y) == 0.0
    assert dcovar(0, x, y) == 0.0
    assert dcovar(3, x, y) == pytest.approx(2.0, rel=1e-9)


def test_minmax_reports_first_occurrence_of_ties():
    x = np.array([3.0, -1.0, 4.0, -1.0, 5.0, 9.0, 2.0, 6.0, 9.0])
    assert minmax(9, x, 0.0, 0.0, 0, 0) == (-1.0, 9.0, 2, 6)


def test_minmax_single_and_empty():
    x = np.array([2.5, 9.0])
    assert minmax(1, x, 0.0, 0.0, 0, 0) == (2.5, 2.5, 1, 1)
    assert minmax(0, x, 1.0, 1.0, 1, 1) == (0.0, 0.0, 0, 0)


def test_minmax_respects_n():
    x = np.array([5.0, 3.0, 8.0, -20.0, 20.0])
    assert minmax(3, x, 0.0, 0.0, 0, 0) == (3.0, 8.0, 2, 3)


def test_module_exposes_the_fortran_units():
    for name in ("moment", "dcovar", "minmax"):
        assert callable(getattr(target_module, name))
