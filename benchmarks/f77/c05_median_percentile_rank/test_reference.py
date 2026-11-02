import numpy as np
import pytest

import target_module
from target_module import copysr, nearrk, ranks, xmedn, xpctl

SAMPLE = [15.0, 20.0, 35.0, 40.0, 50.0]


def median_of(values):
    x = np.array(values, dtype=np.float32)
    w = np.zeros(len(values), dtype=np.float32)
    return xmedn(len(values), x, w), x, w


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([3, 1, 2], 2.0),
        ([4, 1, 3, 2], 2.5),
        ([9], 9.0),
        ([2, 2, 2, 2], 2.0),
        ([-5, -1], -3.0),
        ([10, 0, 0, 0, 10, 10, 10, 0], 5.0),
        ([7, 1, 9, 3, 5, 11, 13], 7.0),
    ],
)
def test_median(values, expected):
    result, _, _ = median_of(values)
    assert result == pytest.approx(expected, rel=1e-5)


def test_median_of_empty_sample_is_zero():
    assert median_of([])[0] == 0.0


def test_median_leaves_input_alone_and_sorts_the_work_vector():
    result, x, w = median_of([4, 1, 3, 2])
    assert result == pytest.approx(2.5, rel=1e-5)
    assert x.tolist() == [4, 1, 3, 2]
    assert w.tolist() == [1, 2, 3, 4]


def test_copysr_sorts_a_copy():
    x = np.array([2.0, -1.0, 2.0, 0.5], dtype=np.float32)
    w = np.zeros(4, dtype=np.float32)
    copysr(4, x, w)
    assert w.tolist() == [-1.0, 0.5, 2.0, 2.0]
    assert x.tolist() == [2.0, -1.0, 2.0, 0.5]


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (25.0, 20.0),  # H = 2.0 exactly
        (40.0, 29.0),  # H = 2.6: 20 + 0.6 * 15
        (90.0, 46.0),  # H = 4.6: 40 + 0.6 * 10
        (0.0, 15.0),
        (100.0, 50.0),
        (150.0, 50.0),  # clamped to 100
        (-5.0, 15.0),  # clamped to 0
    ],
)
def test_percentile_interpolation(p, expected):
    x = np.array(SAMPLE[::-1], dtype=np.float32)
    w = np.zeros(5, dtype=np.float32)
    assert xpctl(5, x, w, p) == pytest.approx(expected, rel=1e-5)


def test_percentile_50_equals_median_for_even_n():
    x = np.array([7.0, 1.0, 3.0, 5.0], dtype=np.float32)
    w = np.zeros(4, dtype=np.float32)
    assert xpctl(4, x, w, 50.0) == pytest.approx(4.0, rel=1e-5)
    assert xmedn(4, x, w) == pytest.approx(4.0, rel=1e-5)


def test_percentile_of_empty_sample_is_zero():
    empty = np.zeros(0, dtype=np.float32)
    assert xpctl(0, empty, empty.copy(), 50.0) == 0.0


@pytest.mark.parametrize(
    ("n", "p", "expected"),
    [
        (5, 50.0, 3),  # 2.5 rounds up, Python round() would give 2
        (5, 30.0, 2),  # 1.5
        (5, 10.0, 1),  # 0.5
        (5, 0.0, 1),  # 0 is raised to 1
        (5, 100.0, 5),
        (5, 120.0, 5),  # 6 is capped at N
        (5, -30.0, 1),
        (4, 50.0, 2),
        (10, 95.0, 10),  # 9.5
        (3, 25.0, 1),  # 0.75
        (3, 49.0, 1),  # 1.47
    ],
)
def test_nearest_rank(n, p, expected):
    assert nearrk(n, p) == expected


def run_ranks(values):
    n = len(values)
    x = np.array(values, dtype=np.float32)
    r = np.zeros(n, dtype=np.float32)
    idx = np.zeros(n, dtype=np.int64)
    ntie = ranks(n, x, r, idx, 0)
    return r.tolist(), idx.tolist(), ntie


def test_ranks_average_the_tied_positions():
    r, idx, ntie = run_ranks([10, 20, 20, 30, 10])
    assert r == [1.5, 3.5, 3.5, 5.0, 1.5]
    assert idx == [1, 5, 2, 3, 4]
    assert ntie == 2


def test_ranks_without_ties_are_a_permutation():
    r, idx, ntie = run_ranks([3, 1, 2])
    assert r == [3.0, 1.0, 2.0]
    assert idx == [2, 3, 1]
    assert ntie == 0


def test_ranks_all_equal_share_the_middle_rank():
    r, _, ntie = run_ranks([4, 4, 4, 4])
    assert r == [2.5, 2.5, 2.5, 2.5]
    assert ntie == 1


def test_ranks_single_and_empty():
    assert run_ranks([8]) == ([1.0], [1], 0)
    assert run_ranks([]) == ([], [], 0)


def test_ranks_negative_and_fractional_values():
    r, _, ntie = run_ranks([0.5, -2.0, 0.5, 7.0, -2.0, 0.5])
    # -2 -2 -> 1.5; 0.5 x3 -> 4; 7 -> 6
    assert r == [4.0, 1.5, 4.0, 6.0, 1.5, 4.0]
    assert ntie == 2


def test_module_exposes_the_fortran_units():
    for name in ("copysr", "xmedn", "xpctl", "nearrk", "ranks"):
        assert callable(getattr(target_module, name))
