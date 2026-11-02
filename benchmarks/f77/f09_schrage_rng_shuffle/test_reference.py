import importlib

import numpy as np
import pytest

import target_module
from target_module import rbound, rnext, rsamp, rseed, rshuf, rstat

# Park-Miller minimal standard sequence from seed 1: X(n+1) = 16807 * X(n) mod (2**31 - 1)
SEED1_HEAD = [
    16807,
    282475249,
    1622650073,
    984943658,
    1144108930,
    470211272,
    101027544,
    1457850878,
    1458777923,
    2007237709,
]


def shuffled(n):
    perm = np.zeros(n + 2, dtype=np.int32)
    rshuf(n, perm)
    return [int(v) for v in perm[:n]]


def sampled(n, m):
    isamp = np.zeros(max(m, 1) + 1, dtype=np.int32)
    nsel = rsamp(n, m, isamp, 9)
    return nsel, [int(v) for v in isamp[:nsel]]


def test_block_data_state_at_program_start():
    # re-executing the module restores the state BLOCK DATA RNGBD gives at program start
    importlib.reload(target_module)
    assert rstat(0) == 0
    assert rnext() == 16807
    assert rstat(0) == 1


def test_first_terms_from_seed_one():
    rseed(1)
    assert [rnext() for _ in range(10)] == SEED1_HEAD


def test_ten_thousandth_term_is_the_published_check_value():
    rseed(1)
    last = 0
    for _ in range(10000):
        last = rnext()
    assert last == 1043618065


@pytest.mark.parametrize("seed", [-1, 0, 2147483647, -2147483647])
def test_seed_one_equivalents(seed):
    rseed(seed)
    assert rnext() == 16807


def test_seed_sign_is_ignored():
    rseed(5)
    pos = [rnext() for _ in range(3)]
    rseed(-5)
    assert [rnext() for _ in range(3)] == pos
    assert pos[0] == 5 * 16807


def test_negative_intermediate_of_schrage_step_is_corrected():
    # A*X = -16807 (mod M) for X = M - 1, so the first term is M - 16807
    rseed(2147483646)
    assert rnext() == 2147483647 - 16807
    assert rnext() == (16807 * (2147483647 - 16807)) % 2147483647


def test_draw_counter_is_reset_by_seed_and_counts_bound_calls():
    rseed(1)
    assert rstat(0) == 0
    rnext()
    rbound(10)
    assert rstat(0) == 2
    assert rstat(0) == 2
    rseed(1)
    assert rstat(0) == 0


def test_bound_uses_modulus_plus_one():
    rseed(1)
    # 16807 mod 5 = 2, 282475249 mod 7 = 0 (digit alternating sum), 1622650073 mod 1 = 0
    assert rbound(5) == 3
    assert rbound(2) == 2  # 282475249 is odd
    assert rbound(1) == 1


def test_bound_below_one_draws_nothing():
    rseed(1)
    assert rbound(0) == 0
    assert rbound(-4) == 0
    assert rstat(0) == 0
    assert rnext() == 16807


def test_bound_stays_in_range_for_many_draws():
    rseed(12345)
    values = [rbound(6) for _ in range(600)]
    assert min(values) == 1 and max(values) == 6


def test_shuffle_small_cases_by_hand():
    rseed(1)
    # n=4: draws 16807 (j=4, no swap), 282475249 (j=2, swap 3<->2), 1622650073 (j=2)
    assert shuffled(4) == [1, 3, 2, 4]
    assert rstat(0) == 3
    rseed(1)
    assert shuffled(5) == [4, 1, 5, 2, 3]
    assert rstat(0) == 4
    rseed(1)
    assert shuffled(3) == [1, 3, 2]
    rseed(1)
    assert shuffled(2) == [1, 2]


def test_shuffle_of_zero_and_one_elements_uses_no_draws():
    rseed(1)
    perm = np.full(3, -7, dtype=np.int32)
    rshuf(0, perm)
    assert list(perm) == [-7, -7, -7]
    rshuf(1, perm)
    assert list(perm) == [1, -7, -7]
    assert rstat(0) == 0


@pytest.mark.parametrize("seed", [1, 2, 99, 2024])
def test_shuffle_is_a_reproducible_permutation(seed):
    rseed(seed)
    first = shuffled(12)
    assert sorted(first) == list(range(1, 13))
    rseed(seed)
    assert shuffled(12) == first


def test_sampling_by_hand():
    rseed(1)
    # draws: 16807 -> 3 (>2), 282475249 -> 2 (<=2, take 2), 1622650073 -> 3 (>1),
    # 984943658 -> 1 (<=1, take 4); the scan stops after the fourth draw
    assert sampled(5, 2) == (2, [2, 4])
    assert rstat(0) == 4
    rseed(1)
    assert sampled(5, 1) == (1, [4])
    assert rstat(0) == 4
    rseed(1)
    assert sampled(3, 3) == (3, [1, 2, 3])
    assert rstat(0) == 3


def test_sampling_invalid_sizes_do_nothing():
    rseed(1)
    assert sampled(5, 0) == (0, [])
    assert sampled(5, 6) == (0, [])
    assert sampled(5, -1) == (0, [])
    assert rstat(0) == 0


@pytest.mark.parametrize("seed", [3, 17, 4242])
def test_sampling_returns_increasing_distinct_values(seed):
    rseed(seed)
    nsel, values = sampled(20, 7)
    assert nsel == 7
    assert values == sorted(set(values))
    assert 1 <= values[0] and values[-1] <= 20
    assert rstat(0) <= 20
