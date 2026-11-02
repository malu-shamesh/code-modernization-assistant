import importlib

import numpy as np
import pytest

import target_module


@pytest.fixture
def st():
    """A freshly imported module, so the DATA-initialised state is the starting point."""
    return importlib.reload(target_module)


def get(m):
    return m.rstget(0, 0.0, 0.0)


def rng(m):
    return m.rstrng(0.0, 0.0)


def feed(m, values):
    m.rstall(len(values), np.array(values, dtype=np.float64))


def test_fresh_state_is_empty_and_takes_the_alternate_return(st):
    assert get(st) == (0, 0.0, 0.0, 1)
    assert rng(st) == (0.0, 0.0, 1)


def test_classic_eight_values(st):
    feed(st, [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (8, 0)
    assert mean == pytest.approx(5.0, rel=1e-9)
    assert var == pytest.approx(32.0 / 7.0, rel=1e-9)
    xmin, xmax, alt = rng(st)
    assert (xmin, xmax, alt) == (2.0, 9.0, 0)


def test_single_observation_has_zero_variance(st):
    st.rstadd(3.5)
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (1, 0)
    assert mean == pytest.approx(3.5, rel=1e-9)
    assert var == 0.0
    assert rng(st) == (3.5, 3.5, 0)


def test_two_observations(st):
    st.rstadd(1.0)
    st.rstadd(3.0)
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (2, 0)
    assert mean == pytest.approx(2.0, rel=1e-9)
    assert var == pytest.approx(2.0, rel=1e-9)


def test_negative_values_do_not_stick_at_the_initial_zero_extremes(st):
    for v in (-5.0, -1.0, -9.0):
        st.rstadd(v)
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (3, 0)
    assert mean == pytest.approx(-5.0, rel=1e-9)
    assert var == pytest.approx(16.0, rel=1e-9)
    assert rng(st) == (-9.0, -1.0, 0)


def test_state_accumulates_across_separate_calls(st):
    st.rstadd(10.0)
    assert get(st)[0] == 1
    st.rstadd(20.0)
    st.rstadd(30.0)
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (3, 0)
    assert mean == pytest.approx(20.0, rel=1e-9)
    assert var == pytest.approx(100.0, rel=1e-9)
    assert rng(st) == (10.0, 30.0, 0)


def test_reset_entry_clears_everything(st):
    feed(st, [1.0, 2.0, 3.0])
    st.rstrst()
    assert get(st) == (0, 0.0, 0.0, 1)
    assert rng(st) == (0.0, 0.0, 1)
    st.rstadd(-2.0)
    assert get(st)[:2] == (1, -2.0)


def test_rstall_replaces_earlier_data(st):
    feed(st, [100.0, 200.0, 300.0])
    feed(st, [1.0, 3.0])
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (2, 0)
    assert mean == pytest.approx(2.0, rel=1e-9)
    assert var == pytest.approx(2.0, rel=1e-9)
    assert rng(st) == (1.0, 3.0, 0)


def test_rstall_uses_only_the_first_n_elements(st):
    st.rstall(3, np.array([1.0, 2.0, 3.0, 1000.0, -1000.0]))
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (3, 0)
    assert mean == pytest.approx(2.0, rel=1e-9)
    assert var == pytest.approx(1.0, rel=1e-9)
    assert rng(st) == (1.0, 3.0, 0)


def test_rstall_with_zero_elements_leaves_an_empty_state(st):
    feed(st, [4.0])
    st.rstall(0, np.array([9.0]))
    assert get(st) == (0, 0.0, 0.0, 1)


def test_constant_stream_has_zero_variance(st):
    feed(st, [7.0, 7.0, 7.0, 7.0])
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (4, 0)
    assert mean == pytest.approx(7.0, rel=1e-9)
    assert var == pytest.approx(0.0, abs=1e-12)


def test_unsorted_data_min_max_and_variance(st):
    feed(st, [5.0, 1.0, 9.0, 3.0])
    cnt, mean, var, alt = get(st)
    assert (cnt, alt) == (4, 0)
    assert mean == pytest.approx(4.5, rel=1e-9)
    assert var == pytest.approx(35.0 / 3.0, rel=1e-9)
    assert rng(st) == (1.0, 9.0, 0)
