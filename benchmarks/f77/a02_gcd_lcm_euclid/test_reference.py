import numpy as np
import pytest

import target_module as tm


@pytest.mark.parametrize(
    "m, n, expected",
    [
        (12, 18, 6),
        (-12, 18, 6),
        (12, -18, 6),
        (-48, -180, 12),
        (0, 5, 5),
        (7, 0, 7),
        (0, 0, 0),
        (17, 5, 1),
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
        (21, 6, 42),
        (7, 7, 7),
        (0, 5, 0),
        (5, 0, 0),
        (13, 17, 221),
    ],
)
def test_ilcm(m, n, expected):
    assert tm.ilcm(m, n) == expected


def test_igcdv_basic():
    assert tm.igcdv(3, np.array([12, 18, 30, 99])) == 6


def test_igcdv_only_uses_first_n_elements():
    ia = np.array([8, 12, 5, 7])
    assert tm.igcdv(2, ia) == 4
    assert tm.igcdv(4, ia) == 1


def test_igcdv_empty_and_negative_and_zero_entries():
    ia = np.array([-9, 0, 6])
    assert tm.igcdv(0, ia) == 0
    assert tm.igcdv(-2, ia) == 0
    assert tm.igcdv(2, ia) == 9
    assert tm.igcdv(3, ia) == 3


def test_igcdv_early_exit_on_one_matches_full_result():
    assert tm.igcdv(5, np.array([9, 4, 0, 0, 0])) == 1


@pytest.mark.parametrize(
    "a, p, expected",
    [
        (7, 3, 1),
        (-7, 3, 2),  # MOD(-7,3) is -1, shifted into range
        (-6, 3, 0),
        (0, 5, 0),
        (-1, 10, 9),
        (13, 13, 0),
    ],
)
def test_imodp(a, p, expected):
    assert tm.imodp(a, p) == expected


@pytest.mark.parametrize(
    "num, den, expected",
    [
        (6, 8, (3, 4)),
        (6, -8, (-3, 4)),
        (-10, -4, (5, 2)),
        (-10, 4, (-5, 2)),
        (0, 5, (0, 1)),
        (7, 1, (7, 1)),
        (0, 0, (0, 0)),
        (5, 0, (1, 0)),
    ],
)
def test_reduce_fraction(num, den, expected):
    assert tm.reduce(num, den) == expected
