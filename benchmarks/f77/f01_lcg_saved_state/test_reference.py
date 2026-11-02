import importlib

import numpy as np
import pytest

import target_module
from target_module import lcgfil, lcgint, lcgrng, lcguni

# Hand-computed terms of X(N+1) = MOD(75*X(N) + 74, 65537) starting from seed 1:
#   1 -> 149 -> 11249 -> 57305 -> 38044
SEQ_SEED_1 = [149, 11249, 57305, 38044]


def test_data_statement_state_before_any_restart():
    # re-executing the module restores the DATA initial values IX = 1, NCALL = 0
    importlib.reload(target_module)
    assert lcgint(2, 0) == 0
    assert lcgint(1, 0) == 149
    assert lcgint(2, 0) == 1


def test_restart_returns_seed():
    assert lcgint(0, 1) == 1
    assert lcgint(0, 12345) == 12345


def test_sequence_from_seed_one():
    lcgint(0, 1)
    assert [lcgint(1, 0) for _ in range(4)] == SEQ_SEED_1


def test_negative_seed_uses_magnitude():
    lcgint(0, -1)
    assert lcgint(1, 0) == 149


def test_seed_reduced_modulo_m():
    assert lcgint(0, 70000) == 70000 - 65537
    assert lcgint(1, 0) == (75 * 4463 + 74) % 65537


def test_seed_equal_to_modulus_gives_zero_state():
    assert lcgint(0, 65537) == 0
    assert lcgint(1, 0) == 74
    assert lcgint(1, 0) == 74 * 75 + 74


def test_draw_counter_and_restart():
    lcgint(0, 1)
    assert lcgint(2, 0) == 0
    for _ in range(3):
        lcgint(1, 0)
    assert lcgint(2, 0) == 3
    lcgint(0, 1)
    assert lcgint(2, 0) == 0


def test_counter_mode_does_not_advance():
    lcgint(0, 1)
    lcgint(2, 0)
    lcgint(2, 0)
    assert lcgint(1, 0) == 149


def test_unknown_mode_returns_minus_one():
    lcgint(0, 1)
    assert lcgint(7, 0) == -1
    assert lcgint(-3, 0) == -1
    assert lcgint(1, 0) == 149


def test_uniform_deviates():
    lcgint(0, 1)
    got = [lcguni() for _ in range(4)]
    assert got == pytest.approx([s / 65537.0 for s in SEQ_SEED_1], rel=1e-5)
    assert all(0.0 <= g < 1.0 for g in got)


def test_integer_range_n10():
    lcgint(0, 1)
    # (149*10)/65537 = 0, (11249*10)/65537 = 1, (57305*10)/65537 = 8, (38044*10)/65537 = 5
    assert [lcgrng(10) for _ in range(4)] == [1, 2, 9, 6]


def test_integer_range_one_is_always_one():
    lcgint(0, 99)
    assert {lcgrng(1) for _ in range(20)} == {1}


def test_integer_range_large_n_uses_truncation():
    lcgint(0, 1)
    # 149*32767 = 4882283 and 4882283 // 65537 = 74
    assert lcgrng(32767) == 75


def test_integer_range_divisor_is_exactly_65537():
    lcgint(0, 1)
    lcgint(1, 0)
    lcgint(1, 0)
    # third and fourth terms are 57305 and 38044:
    # 57305*32767 = 28651*65537 + 12348 and 38044*32767 = 19021*65537 + 8471
    assert lcgrng(32767) == 28652
    assert lcgrng(32767) == 19022


def test_fill_matches_sequence():
    lcgint(0, 1)
    x = np.zeros(4, dtype=np.float32)
    lcgfil(4, x)
    assert x == pytest.approx([s / 65537.0 for s in SEQ_SEED_1], rel=1e-5)
    assert lcgint(2, 0) == 4


def test_fill_partial_leaves_tail_untouched():
    lcgint(0, 1)
    x = np.full(5, -1.0)
    lcgfil(2, x)
    assert x[:2] == pytest.approx([149 / 65537.0, 11249 / 65537.0], rel=1e-5)
    assert list(x[2:]) == [-1.0, -1.0, -1.0]


def test_fill_zero_length_draws_nothing():
    lcgint(0, 1)
    x = np.full(3, 7.0)
    lcgfil(0, x)
    assert list(x) == [7.0, 7.0, 7.0]
    assert lcgint(2, 0) == 0
