import importlib

import numpy as np
import pytest

import target_module as tm


@pytest.fixture(autouse=True)
def fresh_module():
    # BLOCK DATA state is established when the module is initialised
    importlib.reload(tm)


def perm(values):
    return np.array(values, dtype=np.int64)


def step(values):
    ip = perm(values)
    last = tm.nxtprm(len(values), ip, False)
    return ip.tolist(), last


def test_block_data_starts_both_counters_at_zero():
    assert tm.ncount(1) == 0
    assert tm.ncount(2) == 0


def test_successor_sequence_of_three_elements_and_wrap():
    expected = [[1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    ip = perm([1, 2, 3])
    for want in expected:
        assert tm.nxtprm(3, ip, True) is False
        assert ip.tolist() == want
    # the last permutation wraps to the first one
    assert tm.nxtprm(3, ip, False) is True
    assert ip.tolist() == [1, 2, 3]
    assert tm.ncount(1) == 5


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ([1, 5, 4, 3, 2], [2, 1, 3, 4, 5]),  # ascent at the very first position
        ([2, 3, 1], [3, 1, 2]),
        ([4, 1, 3, 2], [4, 2, 1, 3]),
    ],
)
def test_successor_of_selected_permutations(before, after):
    assert step(before) == (after, False)


@pytest.mark.parametrize("n", [2, 5])
def test_wrap_around_reverses_the_whole_vector(n):
    descending = list(range(n, 0, -1))
    assert step(descending) == (list(range(1, n + 1)), True)


def test_single_element_and_empty_vector_are_their_own_last_permutation():
    assert step([1]) == ([1], True)
    assert step([]) == ([], True)
    assert tm.ncount(1) == 0


def test_enumerating_four_elements_visits_24_distinct_permutations_in_order():
    ip = perm([1, 2, 3, 4])
    seen = [tuple(ip.tolist())]
    while not tm.nxtprm(4, ip, False):
        seen.append(tuple(ip.tolist()))
    assert len(seen) == 24
    assert seen == sorted(set(seen))
    assert ip.tolist() == [1, 2, 3, 4]
    assert tm.ncount(1) == 23
    assert [tm.prmrnk(4, perm(p)) for p in seen] == list(range(24))


@pytest.mark.parametrize(
    ("values", "rank"),
    [
        ([1, 2, 3], 0),
        ([1, 3, 2], 1),
        ([2, 3, 1], 3),
        ([3, 2, 1], 5),
        ([3, 4, 2, 1], 17),
        ([4, 3, 2, 1], 23),
        ([1], 0),
        ([], 0),
    ],
)
def test_prmrnk(values, rank):
    assert tm.prmrnk(len(values), perm(values)) == rank


def test_prmrnk_rejects_large_n_without_counting():
    assert tm.prmrnk(13, perm(list(range(13, 0, -1)))) == -1
    assert tm.ncount(2) == 0
    tm.prmrnk(3, perm([1, 2, 3]))
    tm.prmrnk(3, perm([3, 2, 1]))
    assert tm.ncount(2) == 2


def test_prmrnk_of_the_last_permutation_of_twelve_elements():
    # 12! - 1 = 479001599 still fits in 32 bits
    assert tm.prmrnk(12, perm(list(range(12, 0, -1)))) == 479001599


@pytest.mark.parametrize(
    ("n", "k", "expected"),
    [
        (4, 17, [3, 4, 2, 1]),
        (4, 0, [1, 2, 3, 4]),
        (4, 23, [4, 3, 2, 1]),
        (4, 3, [1, 3, 4, 2]),
        (3, 4, [3, 1, 2]),
        (1, 0, [1]),
    ],
)
def test_prmunr(n, k, expected):
    ip = perm([0] * n)
    assert tm.prmunr(n, k, ip, 9) == 0
    assert ip.tolist() == expected


@pytest.mark.parametrize(
    ("n", "k", "ierr"), [(4, 24, 2), (4, -1, 2), (3, 6, 2), (13, 0, 1), (-1, 0, 1)]
)
def test_prmunr_error_codes_leave_the_vector_alone(n, k, ierr):
    ip = perm([7, 7, 7, 7])
    assert tm.prmunr(n, k, ip, 0) == ierr
    assert ip.tolist() == [7, 7, 7, 7]


def test_prmunr_zero_elements_accepts_only_rank_zero():
    empty = perm([])
    assert tm.prmunr(0, 0, empty, 5) == 0
    assert tm.prmunr(0, 1, empty, 0) == 2


def test_prmunr_and_prmrnk_are_inverse_for_five_elements():
    ip = perm([0] * 5)
    for k in range(120):
        assert tm.prmunr(5, k, ip, 0) == 0
        assert sorted(ip.tolist()) == [1, 2, 3, 4, 5]
        assert tm.prmrnk(5, ip) == k
    assert tm.ncount(2) == 120


@pytest.mark.parametrize(
    ("start", "istep", "expected"),
    [
        ([3, 4, 2, 1], 10, [1, 3, 4, 2]),  # rank 17 + 10 = 27, mod 24 = 3
        ([3, 4, 2, 1], -20, [4, 2, 3, 1]),  # rank -3 -> MOD gives -3 -> +24 = 21
        ([1, 2, 3, 4], -25, [4, 3, 2, 1]),  # MOD(-25, 24) = -1 -> 23
        ([1, 2, 3, 4], -48, [1, 2, 3, 4]),  # MOD(-48, 24) = 0
        ([4, 3, 2, 1], 5, [1, 4, 2, 3]),  # 23 + 5 = 28 mod 24 = 4
        ([1, 2, 3], -7, [3, 2, 1]),  # MOD(-7, 6) = -1 -> 5
        ([2, 3, 1], 0, [2, 3, 1]),
        ([2, 3, 1], 6, [2, 3, 1]),
    ],
)
def test_prmskp_moves_cyclically(start, istep, expected):
    ip = perm(start)
    assert tm.prmskp(len(start), ip, istep, 9) == 0
    assert ip.tolist() == expected


def test_prmskp_counts_one_rank_call_and_rejects_large_n():
    ip = perm([1, 2, 3])
    tm.prmskp(3, ip, 1, 0)
    assert tm.ncount(2) == 1
    big = perm(list(range(13)))
    assert tm.prmskp(13, big, 1, 0) == 1
    assert big.tolist() == list(range(13))
    assert tm.ncount(2) == 1


def test_counters_are_independent_and_selected_by_argument():
    ip = perm([1, 2, 3])
    tm.nxtprm(3, ip, False)
    tm.nxtprm(3, ip, False)
    tm.prmrnk(3, ip)
    assert tm.ncount(1) == 2
    assert tm.ncount(2) == 1
    assert tm.ncount(7) == 1
