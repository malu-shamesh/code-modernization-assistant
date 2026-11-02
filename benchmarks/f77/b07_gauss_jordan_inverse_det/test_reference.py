import target_module
import numpy as np
import pytest


def invert(rows, lda=None):
    rows = np.array(rows, dtype=float)
    n = rows.shape[0]
    lda = n if lda is None else lda
    a = np.full((lda, n), 321.0)
    a[:n, :] = rows
    det, ierr = target_module.gjinv(a, lda, n, 0.0, 0)
    return a, det, ierr


def test_inverse_of_symmetric_two_by_two():
    a, det, ierr = invert([[2, 1], [1, 1]])
    assert ierr == 0
    assert det == pytest.approx(1.0, rel=1e-9)
    assert a == pytest.approx(np.array([[1.0, -1.0], [-1.0, 2.0]]), rel=1e-9)


def test_inverse_and_negative_determinant():
    a, det, ierr = invert([[1, 2], [3, 4]])
    assert ierr == 0
    assert det == pytest.approx(-2.0, rel=1e-9)
    assert a == pytest.approx(np.array([[-2.0, 1.0], [1.5, -0.5]]), rel=1e-9)


def test_classic_three_by_three_with_integer_inverse():
    a, det, ierr = invert([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    assert ierr == 0
    assert det == pytest.approx(1.0, rel=1e-9)
    expected = np.array([[-24.0, 18.0, 5.0], [20.0, -15.0, -4.0], [-5.0, 4.0, 1.0]])
    assert a == pytest.approx(expected, rel=1e-9)


def test_exchange_matrix_needs_the_column_unscrambling():
    a, det, ierr = invert([[0, 1], [1, 0]])
    assert ierr == 0
    assert det == pytest.approx(-1.0, rel=1e-9)
    assert a == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]), abs=1e-12)


def test_scaled_cyclic_permutation_has_positive_determinant():
    a, det, ierr = invert([[0, 2, 0], [0, 0, 3], [4, 0, 0]])
    assert ierr == 0
    assert det == pytest.approx(24.0, rel=1e-9)
    expected = np.array([[0.0, 0.0, 0.25], [0.5, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0]])
    assert a == pytest.approx(expected, abs=1e-12)


def test_inverting_twice_restores_the_matrix():
    rows = [[4, 1, 2], [1, 5, 3], [2, 3, 6]]
    a, det, ierr = invert(rows)
    assert ierr == 0
    b, det2, ierr2 = invert(a.tolist())
    assert ierr2 == 0
    assert b == pytest.approx(np.array(rows, dtype=float), rel=1e-9)
    assert det * det2 == pytest.approx(1.0, rel=1e-9)


def test_leading_dimension_larger_than_n_leaves_padding_alone():
    a, det, ierr = invert([[2, 1], [1, 1]], lda=4)
    assert ierr == 0
    assert a[:2, :] == pytest.approx(np.array([[1.0, -1.0], [-1.0, 2.0]]), rel=1e-9)
    assert a[2:, :].tolist() == [[321.0, 321.0]] * 2


def test_singular_matrix_reports_ierr_one_and_zero_determinant():
    a, det, ierr = invert([[1, 2], [2, 4]])
    assert ierr == 1
    assert det == 0.0


def test_zero_matrix_is_singular():
    a, det, ierr = invert([[0, 0], [0, 0]])
    assert (det, ierr) == (0.0, 1)


def test_late_singularity_in_three_by_three():
    # third row is the sum of the first two; all pivots are powers of two, so the zero is exact
    a, det, ierr = invert([[2, 0, 1], [0, 2, 1], [2, 2, 2]])
    assert (det, ierr) == (0.0, 1)


@pytest.mark.parametrize("n", [0, -3, 51])
def test_size_out_of_range_gives_ierr_two_with_det_preset_to_one(n):
    a = np.zeros((60, 60))
    det, ierr = target_module.gjinv(a, 60, n, 99.0, 0)
    assert (det, ierr) == (1.0, 2)
    assert not a.any()


def test_largest_supported_size():
    a = np.zeros((50, 50))
    np.fill_diagonal(a, 2.0)
    det, ierr = target_module.gjinv(a, 50, 50, 0.0, 0)
    assert ierr == 0
    assert det == 2.0**50
    assert np.diag(a).tolist() == [0.5] * 50
    assert np.count_nonzero(a) == 50


def test_mxresi_measures_deviation_from_identity():
    eye = np.eye(2)
    assert target_module.mxresi(eye, eye, 2, 2) == 0.0
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    # A*I - I = [[0, 2], [3, 3]]
    assert target_module.mxresi(a, eye, 2, 2) == pytest.approx(3.0, rel=1e-9)
    # the largest deviation may sit in any row or column
    assert target_module.mxresi(np.array([[5.0, 0.0], [0.0, 1.0]]), eye, 2, 2) == 4.0
    assert target_module.mxresi(np.array([[1.0, 0.0], [0.0, 5.0]]), eye, 2, 2) == 4.0
    assert target_module.mxresi(np.array([[1.0, 0.0], [7.0, 1.0]]), eye, 2, 2) == 7.0
    assert target_module.mxresi(np.array([[1.0, 9.0], [0.0, 1.0]]), eye, 2, 2) == 9.0


def test_mxresi_of_computed_inverse_is_tiny():
    rows = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    inv, det, ierr = invert(rows.tolist())
    assert target_module.mxresi(rows, inv, 3, 3) < 1e-12


def test_one_by_one_matrix():
    a, det, ierr = invert([[4]])
    assert (det, ierr) == (4.0, 0)
    assert a.tolist() == [[0.25]]
    a, det, ierr = invert([[0]])
    assert (det, ierr) == (0.0, 1)


def test_column_interchanges_are_undone_for_a_pivot_chain_of_three_steps():
    # full pivoting starts at the entry 2 in position (3,2), so the recorded column swaps matter
    a, det, ierr = invert([[0, 0, 1], [0, 1, 0], [1, 2, 0]])
    assert ierr == 0
    assert det == pytest.approx(-1.0, rel=1e-9)
    expected = np.array([[0.0, -2.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert a == pytest.approx(expected, abs=1e-12)
