import numpy as np
import pytest

from target_module import (
    bitcnt,
    bitrev,
    bsclr,
    bscnt,
    bsnext,
    bsset,
    bstest,
    graycd,
    graydc,
    rotl32,
)

INT_MIN = -2147483648
INT_MAX = 2147483647


@pytest.mark.parametrize(
    "word, count",
    [(0, 0), (1, 1), (255, 8), (-1, 32), (INT_MIN, 1), (INT_MAX, 31), (-2, 31), (1024, 1)],
)
def test_bitcnt_counts_sign_bit_too(word, count):
    assert bitcnt(word) == count


@pytest.mark.parametrize(
    "word, nb, expected",
    [
        (1, 8, 128),
        (1, 4, 8),
        (6, 4, 6),
        (19, 5, 25),
        (18, 5, 9),
        (-1, 8, 255),
        (-1, 32, -1),
        (1, 32, INT_MIN),
        (INT_MIN, 32, 1),
        (5, 0, 0),
        (5, -3, 0),
        (1, 40, INT_MIN),
    ],
)
def test_bitrev(word, nb, expected):
    assert bitrev(word, nb) == expected


@pytest.mark.parametrize(
    "word, gray",
    [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 2),
        (4, 6),
        (5, 7),
        (7, 4),
        (8, 12),
        (-1, INT_MIN),
        (INT_MIN, -1073741824),
    ],
)
def test_gray_code_known_values_and_inverse(word, gray):
    assert graycd(word) == gray
    assert graydc(gray) == word


def test_gray_round_trip_over_many_words():
    for w in [0, 1, 12345, 65535, 2**30, INT_MAX, INT_MIN, INT_MIN + 1, -12345, -1]:
        assert graydc(graycd(w)) == w
    # neighbouring integers differ in exactly one bit of their Gray codes
    for w in range(0, 200):
        assert bitcnt(graycd(w) ^ graycd(w + 1)) == 1


@pytest.mark.parametrize(
    "word, shift, expected",
    [
        (1, 0, 1),
        (1, 1, 2),
        (1, 31, INT_MIN),
        (INT_MIN, 1, 1),
        (3, 31, -2147483647),
        (5, -1, -2147483646),
        (7, -33, -2147483645),
        (305419896, 8, 878082066),
        (305419896, 24, 2014458966),
        (-1, 13, -1),
        (0, 7, 0),
    ],
)
def test_rotl32(word, shift, expected):
    assert rotl32(word, shift) == expected


def test_rotl32_full_turns_and_inverse():
    for w in [1, 305419896, INT_MIN, -12345]:
        assert rotl32(w, 32) == w
        assert rotl32(w, 0) == w
        assert rotl32(w, 33) == rotl32(w, 1)
        assert rotl32(rotl32(w, 5), -5) == w


def test_set_bits_across_words_including_sign_bit():
    iw = np.zeros(3, dtype=np.int32)
    assert bsset(iw, 3, 0, 9) == 0
    assert iw[0] == 1
    assert bsset(iw, 3, 31, 9) == 0
    assert iw[0] == -2147483647
    assert bsset(iw, 3, 32, 9) == 0
    assert iw[1] == 1
    assert bsset(iw, 3, 95, 9) == 0
    assert iw[2] == INT_MIN


def test_set_out_of_range_changes_nothing():
    iw = np.zeros(3, dtype=np.int32)
    for k in (96, 100000, -1, -32):
        assert bsset(iw, 3, k, 9) == 1
    assert bsset(iw, 0, 0, 9) == 1
    assert not iw.any()


def test_membership_tests():
    iw = np.zeros(3, dtype=np.int32)
    for k in (0, 31, 32, 95):
        bsset(iw, 3, k, 9)
    for k in (0, 31, 32, 95):
        assert bstest(iw, 3, k)
    for k in (1, 30, 33, 63, 94, 96, -1, 1000):
        assert not bstest(iw, 3, k)
    assert not bstest(iw, 1, 32)


def test_clear_bits():
    iw = np.zeros(3, dtype=np.int32)
    for k in (0, 31, 95):
        bsset(iw, 3, k, 9)
    assert bsclr(iw, 3, 31, 9) == 0
    assert iw[0] == 1
    assert bsclr(iw, 3, 5, 9) == 0
    assert iw[0] == 1
    assert bsclr(iw, 3, 95, 9) == 0
    assert iw[2] == 0
    assert bsclr(iw, 3, 96, 9) == 1
    assert bsclr(iw, 3, -1, 9) == 1
    assert iw[0] == 1


def test_second_and_third_words_are_addressed_independently():
    iw = np.zeros(3, dtype=np.int32)
    # bit 32 is bit 0 of word 2 and bit 64 is bit 0 of word 3
    assert bsset(iw, 3, 32, 9) == 0
    assert bsset(iw, 3, 64, 9) == 0
    assert list(iw) == [0, 1, 1]
    assert not bstest(iw, 3, 0)
    assert bstest(iw, 3, 32)
    assert bstest(iw, 3, 64)
    assert not bstest(iw, 3, 33)
    assert bsclr(iw, 3, 32, 9) == 0
    assert list(iw) == [0, 0, 1]
    assert bsclr(iw, 3, 64, 9) == 0
    assert list(iw) == [0, 0, 0]
    assert bsset(iw, 3, 63, 9) == 0
    assert list(iw) == [0, INT_MIN, 0]
    assert bsclr(iw, 3, 63, 9) == 0
    assert list(iw) == [0, 0, 0]


def test_member_count():
    iw = np.zeros(3, dtype=np.int32)
    assert bscnt(iw, 3) == 0
    for k in (0, 31, 32, 95):
        bsset(iw, 3, k, 9)
    assert bscnt(iw, 3) == 4
    assert bscnt(iw, 2) == 3
    assert bscnt(np.array([-1, 0, INT_MIN], dtype=np.int32), 3) == 33
    assert bscnt(iw, 0) == 0
    # only the leading NW words are counted: bits 0 and 31 are in word 1
    assert bscnt(iw, 1) == 2


def test_next_member_search():
    iw = np.zeros(3, dtype=np.int32)
    for k in (3, 31, 70):
        bsset(iw, 3, k, 9)
    assert bsnext(iw, 3, 0) == 3
    # bit 0 itself is found when it is a member
    bsset(iw, 3, 0, 9)
    assert bsnext(iw, 3, 0) == 0
    assert bsnext(iw, 3, -5) == 0
    assert bsnext(iw, 3, 1) == 3
    bsclr(iw, 3, 0, 9)
    assert bsnext(iw, 3, 3) == 3
    assert bsnext(iw, 3, 4) == 31
    assert bsnext(iw, 3, 32) == 70
    assert bsnext(iw, 3, 71) == -1
    assert bsnext(iw, 3, -5) == 3
    assert bsnext(iw, 3, 500) == -1
    assert bsnext(iw, 0, 0) == -1
    assert bsnext(iw, 2, 32) == -1
    # the very last bit of the set is still reachable
    bsset(iw, 3, 95, 9)
    assert bsnext(iw, 3, 71) == 95
    assert bsnext(iw, 3, 95) == 95
