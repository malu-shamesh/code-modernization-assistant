import target_module
import numpy as np
import pytest


def padded(rows, lda):
    """Column-major style storage A(LDA,N): rows beyond len(rows) hold a sentinel."""
    rows = np.array(rows, dtype=float)
    a = np.full((lda, rows.shape[1]), 999.0)
    a[: rows.shape[0], :] = rows
    return a


A23 = [[1, 2, 3], [4, 5, 6]]


def call(trans, a, x, y, *, m=2, n=3, alpha=1.0, beta=0.0, incx=1, incy=1, lda=None):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    lda = a.shape[0] if lda is None else lda
    info = target_module.dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, 0)
    return info, y


def test_no_transpose_respects_leading_dimension():
    info, y = call("N", padded(A23, 3), [1, 1, 1], [0, 0])
    assert info == 0
    assert y.tolist() == [6.0, 15.0]


def test_no_transpose_alpha_beta():
    info, y = call("N", padded(A23, 4), [1, 1, 1], [1, 1], alpha=2.0, beta=1.0)
    assert info == 0
    assert y.tolist() == [13.0, 31.0]


def test_transpose_uses_row_dot_products():
    info, y = call("T", padded(A23, 3), [1, 2], [0, 0, 0])
    assert info == 0
    assert y.tolist() == [9.0, 12.0, 15.0]


@pytest.mark.parametrize("flag", ["C", "c", "t"])
def test_conjugate_and_lowercase_flags_act_like_transpose(flag):
    info, y = call(flag, padded(A23, 2), [1, 2], [1, 1, 1], alpha=1.0, beta=1.0)
    assert info == 0
    assert y.tolist() == [10.0, 13.0, 16.0]


def test_beta_zero_does_not_read_y():
    info, y = call("N", padded(A23, 3), [1, 1, 1], [np.nan, np.nan], beta=0.0)
    assert info == 0
    assert y.tolist() == [6.0, 15.0]


def test_alpha_zero_only_scales_y_and_never_reads_a():
    a = np.full((2, 3), np.nan)
    info, y = call("N", a, [1, 1, 1], [1, 2], alpha=0.0, beta=3.0)
    assert info == 0
    assert y.tolist() == [3.0, 6.0]


def test_alpha_zero_beta_one_is_a_quick_return():
    info, y = call("N", padded(A23, 3), [1, 1, 1], [7, 8], alpha=0.0, beta=1.0)
    assert info == 0
    assert y.tolist() == [7.0, 8.0]


def test_zero_x_entry_skips_infinite_column():
    a = np.array([[1.0, np.inf], [1.0, np.inf]])
    info, y = call("N", a, [1, 0], [0, 0], n=2)
    assert info == 0
    assert y.tolist() == [1.0, 1.0]


def test_strided_x_and_reversed_y():
    # x elements are 1,1,1 (stride 2); y has stride -1 so row 1 lands in y(2)
    info, y = call("N", padded(A23, 3), [1, 9, 1, 9, 1], [0, 0], incx=2, incy=-1)
    assert info == 0
    assert y.tolist() == [15.0, 6.0]


def test_negative_incx_reads_x_backwards():
    info, y = call("N", padded(A23, 3), [1, 2, 3], [0, 0], incx=-1)
    assert info == 0
    assert y.tolist() == [10.0, 28.0]


def test_transpose_with_negative_strides():
    # x walked backwards: (2, 1); y walked backwards: y(3), y(2), y(1)
    info, y = call("T", padded(A23, 3), [1, 2], [0, 0, 0], incx=-1, incy=-1)
    assert info == 0
    # A'*(2,1) = (6, 9, 12); the first result is stored in y(3)
    assert y.tolist() == [12.0, 9.0, 6.0]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"trans": "X"}, 1),
        ({"m": -1}, 2),
        ({"n": -1}, 3),
        ({"lda": 1}, 6),
        ({"incx": 0}, 8),
        ({"incy": 0}, 11),
        ({"trans": "X", "m": -1}, 1),
    ],
)
def test_argument_errors_report_position_and_leave_y_alone(kwargs, expected):
    kw = dict(kwargs)
    trans = kw.pop("trans", "N")
    info, y = call(trans, padded(A23, 3), [1, 1, 1], [5, 6], **kw)
    assert info == expected
    assert y.tolist() == [5.0, 6.0]


def test_empty_dimensions_return_zero_info_and_keep_y():
    info, y = call("N", padded(A23, 3), [1, 1, 1], [5, 6], m=0)
    assert info == 0
    assert y.tolist() == [5.0, 6.0]


@pytest.mark.parametrize(
    "ca, cb, same",
    [
        ("a", "A", True),
        ("A", "a", True),
        ("z", "Z", True),
        ("Z", "z", True),
        ("{", "[", False),
        ("[", "{", False),
        ("`", "@", False),
        ("@", "`", False),
        ("n", "N", True),
        ("N", "N", True),
        ("a", "b", False),
        ("N", "T", False),
    ],
)
def test_lsame_case_folding_stops_at_the_letter_range(ca, cb, same):
    assert target_module.lsame(ca, cb) is same


def test_one_by_one_system_zero_extents_and_minimal_leading_dimension():
    a = np.array([[5.0]])
    for trans in ("N", "T"):
        info, y = call(trans, a, [2], [1], m=1, n=1, alpha=1.0, beta=1.0)
        assert (info, y.tolist()) == (0, [11.0])
    # N = 0 and M = 0 are quick returns, so even BETA = 0 leaves Y alone
    info, y = call("N", padded(A23, 3), [1, 1, 1], [5, 6], n=0)
    assert (info, y.tolist()) == (0, [5.0, 6.0])
    info, y = call("T", padded(A23, 3), [1, 2], [5, 6, 7], m=0)
    assert (info, y.tolist()) == (0, [5.0, 6.0, 7.0])
    # LDA must be at least max(1, M): 1 is fine for M = 1 or M = 0, 0 never is
    assert call("N", a, [2], [1], m=1, n=1, lda=0)[0] == 6
    assert call("N", a, [2], [1], m=0, n=1, lda=1)[0] == 0
    assert call("N", a, [2], [1], m=0, n=1, lda=0)[0] == 6


def test_entries_of_y_beyond_its_length_are_not_touched():
    info, y = call("N", padded(A23, 3), [1, 1, 1], [9, 9, 777], beta=0.0)
    assert (info, y.tolist()) == (0, [6.0, 15.0, 777.0])
    info, y = call("N", padded(A23, 3), [1, 1, 1], [1, 2, 777], alpha=0.0, beta=2.0)
    assert (info, y.tolist()) == (0, [2.0, 4.0, 777.0])
