import target_module
import numpy as np
import pytest


def stored(rows, ld, fill=777.0):
    """Array A(LD,NCOL): the matrix occupies the leading rows, the rest is sentinel."""
    rows = np.array(rows, dtype=float)
    a = np.full((ld, rows.shape[1]), fill)
    a[: rows.shape[0], :] = rows
    return a


# product used throughout: A (2x3) times B (3x2)
A = [[1, 2, 3], [4, 5, 6]]
B = [[1, 0], [0, 1], [1, 1]]
AB = [[4.0, 5.0], [10.0, 11.0]]


def transposed(rows):
    return [list(r) for r in zip(*rows)]


def run(transa, transb, a, lda, b, ldb, c, ldc, *, m=2, n=2, k=3, alpha=1.0, beta=0.0):
    info = target_module.dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 0)
    return info


@pytest.mark.parametrize("ta", ["N", "T"])
@pytest.mark.parametrize("tb", ["N", "T"])
def test_all_four_transpose_cases_give_the_same_product(ta, tb):
    a_rows = A if ta == "N" else transposed(A)
    b_rows = B if tb == "N" else transposed(B)
    a = stored(a_rows, 5)
    b = stored(b_rows, 4)
    c = stored([[1, 1], [1, 1]], 3, fill=-5.0)
    info = run(ta, tb, a, 5, b, 4, c, 3, alpha=2.0, beta=0.5)
    assert info == 0
    expected = [[2 * v + 0.5 for v in row] for row in AB]
    assert c[:2, :] == pytest.approx(np.array(expected), rel=1e-9)
    # rows beyond M in C belong to the caller
    assert c[2, :].tolist() == [-5.0, -5.0]


def test_square_known_products():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = {
        ("N", "N"): [[19, 22], [43, 50]],
        ("T", "N"): [[26, 30], [38, 44]],
        ("N", "T"): [[17, 23], [39, 53]],
        ("T", "T"): [[23, 31], [34, 46]],
    }
    for (ta, tb), want in expected.items():
        c = np.zeros((2, 2))
        assert run(ta, tb, a, 2, b, 2, c, 2, k=2) == 0
        assert c.tolist() == want


def test_lowercase_and_conjugate_flags():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.zeros((2, 2))
    assert run("c", "t", a, 2, b, 2, c, 2, k=2) == 0
    assert c.tolist() == [[23.0, 31.0], [34.0, 46.0]]


def test_beta_zero_overwrites_nan_in_c():
    c = np.full((2, 2), np.nan)
    assert run("N", "N", stored(A, 2), 2, stored(B, 3), 3, c, 2) == 0
    assert c.tolist() == AB


def test_alpha_zero_scales_c_without_reading_a_or_b():
    a = np.full((2, 3), np.nan)
    b = np.full((3, 2), np.nan)
    c = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert run("N", "N", a, 2, b, 3, c, 2, alpha=0.0, beta=2.0) == 0
    assert c.tolist() == [[2.0, 4.0], [6.0, 8.0]]


def test_k_zero_scales_c_only():
    c = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.full((2, 1), np.nan)
    b = np.full((1, 2), np.nan)
    assert run("N", "N", a, 2, b, 1, c, 2, k=0, alpha=1.0, beta=3.0) == 0
    assert c.tolist() == [[3.0, 6.0], [9.0, 12.0]]


def test_quick_return_leaves_c_untouched():
    c = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.full((2, 3), np.nan)
    b = np.full((3, 2), np.nan)
    assert run("N", "N", a, 2, b, 3, c, 2, alpha=0.0, beta=1.0) == 0
    assert c.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_zero_entry_of_b_skips_infinite_column_of_a():
    a = np.array([[1.0, np.inf], [1.0, np.inf]])
    b = np.array([[1.0], [0.0]])
    c = np.zeros((2, 1))
    assert run("N", "N", a, 2, b, 2, c, 2, m=2, n=1, k=2) == 0
    assert c.tolist() == [[1.0], [1.0]]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"transa": "X"}, 1),
        ({"transb": "Z"}, 2),
        ({"m": -1}, 3),
        ({"n": -1}, 4),
        ({"k": -1}, 5),
        ({"lda": 1}, 8),
        ({"ldb": 2}, 10),
        ({"ldc": 1}, 13),
        ({"transa": "X", "m": -1}, 1),
    ],
)
def test_argument_errors_leave_c_untouched(kwargs, expected):
    kw = {"transa": "N", "transb": "N", "lda": 2, "ldb": 3, "ldc": 2}
    kw.update(kwargs)
    c = np.full((2, 2), 9.0)
    info = run(
        kw["transa"],
        kw["transb"],
        stored(A, 2),
        kw["lda"],
        stored(B, 3),
        kw["ldb"],
        c,
        kw["ldc"],
        m=kw.get("m", 2),
        n=kw.get("n", 2),
        k=kw.get("k", 3),
    )
    assert info == expected
    assert c.tolist() == [[9.0, 9.0], [9.0, 9.0]]


def test_leading_dimension_of_transposed_a_is_checked_against_k():
    # TRANSA='T': A is stored K by M = 3 by 2, so LDA must be at least K = 3
    a = stored(transposed(A), 3)
    c = np.zeros((2, 2))
    assert run("T", "N", a, 2, stored(B, 3), 3, c, 2) == 8
    assert run("T", "N", a, 3, stored(B, 3), 3, c, 2) == 0
    assert c.tolist() == AB


def test_leading_dimension_of_transposed_b_is_checked_against_n():
    b = stored(transposed(B), 2)
    c = np.zeros((2, 2))
    assert run("N", "T", stored(A, 2), 2, b, 1, c, 2) == 10
    assert run("N", "T", stored(A, 2), 2, b, 2, c, 2) == 0
    assert c.tolist() == AB


def test_dgescl_scales_only_the_leading_block():
    c = np.arange(1.0, 10.0).reshape(3, 3)
    target_module.dgescl(2, 2, 10.0, c, 3)
    assert c.tolist() == [[10.0, 20.0, 3.0], [40.0, 50.0, 6.0], [7.0, 8.0, 9.0]]
    # BETA = 0 stores zeros in the leading block only
    z = np.full((3, 3), 7.0)
    target_module.dgescl(2, 2, 0.0, z, 3)
    assert z.tolist() == [[0.0, 0.0, 7.0], [0.0, 0.0, 7.0], [7.0, 7.0, 7.0]]


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
    ],
)
def test_lsame_case_folding_stops_at_the_letter_range(ca, cb, same):
    assert target_module.lsame(ca, cb) is same


@pytest.mark.parametrize("ta", ["N", "T"])
@pytest.mark.parametrize("tb", ["N", "T"])
def test_inner_dimension_one_is_an_outer_product_and_beta_zero_spares_padding(ta, tb):
    col = [[1], [2]]
    row = [[3, 4]]
    a = stored(col if ta == "N" else transposed(col), 3)
    b = stored(row if tb == "N" else transposed(row), 3)
    c = np.full((3, 3), 8.0)
    assert run(ta, tb, a, 3, b, 3, c, 3, k=1, alpha=1.0, beta=0.0) == 0
    assert c[:2, :2].tolist() == [[3.0, 4.0], [6.0, 8.0]]
    assert (c[2, :] == 8.0).all()
    assert (c[:, 2] == 8.0).all()


def test_one_by_one_product_and_empty_extents_are_quick_returns():
    a = np.array([[2.0]])
    b = np.array([[5.0]])
    c = np.array([[1.0]])
    assert run("N", "N", a, 1, b, 1, c, 1, m=1, n=1, k=1, alpha=1.0, beta=1.0) == 0
    assert c.tolist() == [[11.0]]
    c = np.full((2, 2), 4.0)
    assert run("N", "N", stored(A, 2), 2, stored(B, 3), 3, c, 2, m=0) == 0
    assert run("N", "N", stored(A, 2), 2, stored(B, 3), 3, c, 2, n=0) == 0
    assert c.tolist() == [[4.0, 4.0], [4.0, 4.0]]


def test_leading_dimensions_need_only_be_one_for_empty_matrices():
    one = np.zeros((1, 1))
    empty = {"m": 0, "n": 0, "k": 0}
    assert run("N", "N", one, 1, one, 1, one, 1, **empty) == 0
    assert run("N", "N", one, 0, one, 1, one, 1, **empty) == 8
    assert run("N", "N", one, 1, one, 0, one, 1, **empty) == 10
    assert run("N", "N", one, 1, one, 1, one, 0, **empty) == 13
