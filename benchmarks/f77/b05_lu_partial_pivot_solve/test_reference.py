import target_module
import numpy as np
import pytest


def mat(rows, lda=None, fill=555.0):
    rows = np.array(rows, dtype=float)
    lda = rows.shape[0] if lda is None else lda
    a = np.full((lda, rows.shape[1]), fill)
    a[: rows.shape[0], :] = rows
    return a


def factor(rows, lda=None):
    a = mat(rows, lda)
    n = len(rows)
    ipvt = np.zeros(n, dtype=int)
    info = target_module.lufac(a, a.shape[0], n, ipvt, 0)
    return a, ipvt, info


def test_two_by_two_factors_hold_negated_multipliers_and_one_based_pivots():
    a, ipvt, info = factor([[1, 2], [3, 4]])
    assert info == 0
    assert ipvt.tolist() == [2, 2]
    assert a == pytest.approx(np.array([[3.0, 4.0], [-1.0 / 3.0, 2.0 / 3.0]]), rel=1e-9)


def test_three_by_three_factors_tie_keeps_first_pivot():
    a, ipvt, info = factor([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    assert info == 0
    assert ipvt.tolist() == [2, 2, 3]
    expected = np.array([[4.0, -6.0, 0.0], [-0.5, 4.0, 1.0], [0.5, -1.0, 1.0]])
    assert a == pytest.approx(expected, rel=1e-9)


def test_rows_below_n_are_not_touched_when_lda_exceeds_n():
    a, ipvt, info = factor([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], lda=5)
    assert info == 0
    assert a[3:, :].tolist() == [[555.0] * 3] * 2
    assert a[:3, 0] == pytest.approx([4.0, -0.5, 0.5], rel=1e-9)


def test_singular_matrix_reports_last_pivot_index():
    a, ipvt, info = factor([[1, 2], [2, 4]])
    assert info == 2
    assert ipvt.tolist() == [2, 2]
    assert a == pytest.approx(np.array([[2.0, 4.0], [-0.5, 0.0]]), abs=1e-12)


def test_zero_first_column_sets_info_one_and_leaves_matrix_alone():
    a, ipvt, info = factor([[0, 1], [0, 2]])
    assert info == 1
    assert ipvt.tolist() == [1, 2]
    assert a.tolist() == [[0.0, 1.0], [0.0, 2.0]]


def test_one_by_one_system():
    a, ipvt, info = factor([[4]])
    assert (info, ipvt.tolist()) == (0, [1])
    b = np.array([8.0])
    target_module.lusol(a, 1, 1, ipvt, b)
    assert b.tolist() == [2.0]
    _, _, info0 = factor([[0]])
    assert info0 == 1


def test_solve_three_by_three():
    a, ipvt, info = factor([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    b = np.array([5.0, -2.0, 9.0])
    target_module.lusol(a, 3, 3, ipvt, b)
    assert b == pytest.approx([1.0, 1.0, 2.0], rel=1e-9)


def test_solve_with_padded_leading_dimension():
    a, ipvt, info = factor([[1, 2], [3, 4]], lda=6)
    b = np.array([5.0, 11.0, -1.0])
    target_module.lusol(a, 6, 2, ipvt, b)
    assert b[:2] == pytest.approx([1.0, 2.0], rel=1e-9)
    assert b[2] == -1.0


def test_solve_tridiagonal_without_row_swaps():
    rows = [[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]]
    a, ipvt, info = factor(rows)
    assert info == 0
    assert ipvt.tolist() == [1, 2, 3, 4]
    b = np.array([6.0, 12.0, 18.0, 19.0])
    target_module.lusol(a, 4, 4, ipvt, b)
    assert b == pytest.approx([1.0, 2.0, 3.0, 4.0], rel=1e-9)


def test_solve_permutation_matrix_needs_two_interchanges():
    a, ipvt, info = factor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert info == 0
    assert ipvt.tolist() == [2, 3, 3]
    b = np.array([3.0, 1.0, 2.0])
    target_module.lusol(a, 3, 3, ipvt, b)
    assert b == pytest.approx([1.0, 2.0, 3.0], rel=1e-9)


@pytest.mark.parametrize(
    "rows, det",
    [
        ([[1, 2], [3, 4]], -2.0),
        ([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], -16.0),
        ([[0, 1], [1, 0]], -1.0),
        ([[0, 0, 1], [1, 0, 0], [0, 1, 0]], 1.0),
        ([[5]], 5.0),
        ([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]], 209.0),
    ],
)
def test_determinant(rows, det):
    a, ipvt, info = factor(rows)
    assert info == 0
    assert target_module.ludet(a, a.shape[0], len(rows), ipvt) == pytest.approx(det, rel=1e-9)


def test_luslv_solves_in_one_call():
    a = mat([[1, 2], [3, 4]])
    ipvt = np.zeros(2, dtype=int)
    b = np.array([5.0, 11.0])
    info = target_module.luslv(a, 2, 2, b, ipvt, 0)
    assert info == 0
    assert b == pytest.approx([1.0, 2.0], rel=1e-9)


def test_luslv_leaves_rhs_alone_for_singular_system():
    a = mat([[1, 2], [2, 4]])
    ipvt = np.zeros(2, dtype=int)
    b = np.array([1.0, 2.0])
    info = target_module.luslv(a, 2, 2, b, ipvt, 0)
    assert info == 2
    assert b.tolist() == [1.0, 2.0]
