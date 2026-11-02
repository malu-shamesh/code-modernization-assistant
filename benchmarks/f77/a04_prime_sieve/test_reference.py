import numpy as np
import pytest

import target_module as tm

PRIMES_TO_50 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


@pytest.mark.parametrize("n", PRIMES_TO_50 + [97, 7919, 104729])
def test_isprim_accepts_primes(n):
    assert tm.isprim(n)


@pytest.mark.parametrize("n", [-5, 0, 1, 4, 9, 15, 25, 49, 91, 121, 169, 7917, 10201])
def test_isprim_rejects_non_primes(n):
    assert not tm.isprim(n)


def test_isprim_agrees_with_the_list_of_small_primes():
    found = [n for n in range(1, 51) if tm.isprim(n)]
    assert found == PRIMES_TO_50


def test_sieve_30_from_zeroed_array():
    isp = np.zeros(30, dtype=bool)
    count = tm.sieve(30, isp, 0)
    assert count == 10
    assert (np.flatnonzero(isp) + 1).tolist() == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_sieve_does_not_depend_on_initial_contents():
    isp = np.ones(30, dtype=bool)
    count = tm.sieve(30, isp, 99)
    assert count == 10
    assert not isp[0]
    assert not isp[3]  # 4 is composite
    assert isp[28]  # 29 is prime


def test_sieve_square_of_prime_is_crossed_out():
    isp = np.zeros(49, dtype=bool)
    assert tm.sieve(49, isp, 0) == 15
    assert not isp[48]  # 49 = 7 * 7
    assert isp[46]  # 47


def test_sieve_tiny_ranges():
    one = np.ones(1, dtype=bool)
    assert tm.sieve(1, one, 5) == 0
    assert not one[0]
    two = np.zeros(2, dtype=bool)
    assert tm.sieve(2, two, 0) == 1
    assert two.tolist() == [False, True]


def test_sieve_uses_only_the_first_n_elements():
    isp = np.ones(20, dtype=bool)
    assert tm.sieve(10, isp, 0) == 4
    assert isp[10:].all()


@pytest.mark.parametrize(
    "n, expected",
    [(1, 0), (2, 1), (10, 4), (100, 25), (1000, 168), (1001, -1), (5000, -1)],
)
def test_nprime(n, expected):
    assert tm.nprime(n) == expected
