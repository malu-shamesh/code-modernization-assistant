import numpy as np
import pytest

import target_module as tm


@pytest.mark.parametrize(
    "n, maxstp, expected",
    [
        (1, 100, (0, 1, 0)),
        (2, 100, (1, 2, 0)),
        (3, 100, (7, 16, 0)),
        (6, 100, (8, 16, 0)),
        (7, 100, (16, 52, 0)),
        (27, 1000, (111, 9232, 0)),
        (1024, 100, (10, 1024, 0)),
    ],
)
def test_colltz_reaches_one(n, maxstp, expected):
    assert tm.colltz(n, maxstp, 0, 0, 0) == expected


@pytest.mark.parametrize(
    "n, maxstp, expected",
    [
        (6, 5, (5, 16, 2)),
        (6, 8, (8, 16, 0)),
        (6, 0, (0, 6, 2)),
        (1, 0, (0, 1, 0)),
        (27, 10, (10, 214, 2)),
    ],
)
def test_colltz_step_limit(n, maxstp, expected):
    assert tm.colltz(n, maxstp, 0, 0, 0) == expected


@pytest.mark.parametrize("n, expected_peak", [(0, 0), (-5, -5), (-1, -1)])
def test_colltz_rejects_non_positive(n, expected_peak):
    assert tm.colltz(n, 50, 0, 0, 0) == (0, expected_peak, 1)


@pytest.mark.parametrize(
    "m, n, expected",
    [
        (12, 18, 6),
        (-12, 18, 6),
        (12, -18, 6),
        (-8, -12, 4),
        (17, 5, 1),
        (0, 9, 9),
        (9, 0, 9),
        (-9, 0, 9),
        (0, -9, 9),
        (0, 0, 0),
        (100, 75, 25),
        (1071, 462, 21),
    ],
)
def test_igcd(m, n, expected):
    assert tm.igcd(m, n) == expected


@pytest.mark.parametrize(
    "m, n, expected",
    [
        (4, 6, 12),
        (-4, 6, 12),
        (4, -6, 12),
        (-3, -5, 15),
        (21, 6, 42),
        (7, 13, 91),
        (0, 5, 0),
        (5, 0, 0),
        (9, 9, 9),
    ],
)
def test_ilcm(m, n, expected):
    assert tm.ilcm(m, n) == expected


@pytest.mark.parametrize(
    "n, values, expected",
    [
        (4, [12, 18, 30, 7], (1, 4)),
        (5, [12, 18, 30, 24, 7], (1, 5)),
        (5, [8, 12, 20, 5, 3], (1, 4)),
        (3, [8, 12, 20], (4, 3)),
        (2, [0, 0], (0, 2)),
        (2, [-4, 6], (2, 2)),
        (1, [-9], (9, 1)),
        (0, [5, 5], (0, 0)),
        (2, [10, 15, 7, 1], (5, 2)),
    ],
)
def test_gcdseq(n, values, expected):
    a = np.array(values, dtype=np.int64)
    assert tm.gcdseq(n, a, 0, 0) == expected


def test_gcdseq_leaves_array_untouched():
    a = np.array([6, 9, 12], dtype=np.int64)
    tm.gcdseq(3, a, 0, 0)
    assert a.tolist() == [6, 9, 12]


def test_gcd_lcm_product_identity():
    for m, n in [(12, 18), (35, 14), (-8, 20), (7, 7)]:
        assert tm.igcd(m, n) * tm.ilcm(m, n) == abs(m * n)


def test_colltz_powers_of_two_only_halve():
    for k in range(11):
        assert tm.colltz(2**k, 100, 0, 0, 0) == (k, 2**k, 0)
