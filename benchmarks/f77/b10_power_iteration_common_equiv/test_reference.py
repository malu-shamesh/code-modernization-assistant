import target_module
import importlib
import math

import numpy as np
import pytest


@pytest.fixture()
def m():
    """Fresh module state (the BLOCK DATA values) for every test."""
    return importlib.reload(target_module)


def run(m, rows, x0, *, lda=None):
    rows = np.array(rows, dtype=float)
    n = rows.shape[0]
    lda = n if lda is None else lda
    a = np.full((lda, n), 99.0)
    a[:n, :] = rows
    return m.pwritr(a, lda, n, np.array(x0, dtype=float), 0.0, 0, 0)


SLOW = [[1.0, 0.0], [0.0, 0.99]]  # eigenvalue ratio 0.99: converges very slowly
EXACT = [[2.0, 1.0], [1.0, 2.0]]  # (1, 1) is an eigenvector with eigenvalue 3


def test_block_data_defaults_fifty_sweeps_and_default_tolerance(m):
    assert m.pwrget(3, 0) == 0.0
    assert m.pwrget(4, 0) == 0.0
    assert m.pwrget(1, 1) == 0.0
    assert m.pwrget(2, 1) == 0.0
    assert m.pwrrq(np.eye(2), 2, 2) == 0.0
    eig, iters, ierr = run(m, SLOW, [1.0, 1.0])
    assert (iters, ierr) == (50, 1)
    assert m.pwrget(3, 0) == 1.0
    assert m.pwrget(4, 0) == 50.0
    # a zero tolerance request keeps the default tolerance instead of replacing it
    baseline = run(m, [[3.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    m.pwrcfg(50, 0.0)
    again = run(m, [[3.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    assert again[1:] == baseline[1:]
    assert baseline[1] < 20


def test_pwrcfg_sets_limits_and_ignores_nonpositive_values(m):
    m.pwrcfg(5, 0.0)
    assert run(m, SLOW, [1.0, 1.0])[1:] == (5, 1)
    m.pwrcfg(0, -1.0)
    assert run(m, SLOW, [1.0, 1.0])[1:] == (5, 1)
    m.pwrcfg(-4, 1.0e-3)
    eig, iters, ierr = run(m, SLOW, [1.0, 1.0])
    assert (iters, ierr) == (2, 0)


def test_loose_tolerance_stops_at_second_sweep_with_hand_formulas(m):
    m.pwrcfg(50, 1.0e-3)
    eig, iters, ierr = run(m, SLOW, [1.0, 1.0])
    assert (iters, ierr) == (2, 0)
    # V after the first sweep is (1, 0.99)/norm; its Rayleigh quotient is
    # (1 + 0.99**3) / (1 + 0.99**2)
    norm = math.sqrt(1.0 + 0.99**2)
    assert eig == pytest.approx((1.0 + 0.99**3) / (1.0 + 0.99**2), rel=1e-9)
    assert m.pwrget(1, 1) == pytest.approx(1.0 / norm, rel=1e-9)
    assert m.pwrget(1, 2) == pytest.approx(0.99 / norm, rel=1e-9)
    # residual R = A*V - eig*V, stored over W
    e = eig
    assert m.pwrget(2, 1) == pytest.approx((1.0 - e) / norm, rel=1e-6)
    assert m.pwrget(2, 2) == pytest.approx((0.99**2 - e * 0.99) / norm, rel=1e-6)


def test_exhausted_sweeps_keep_v_consistent_with_the_last_product(m):
    m.pwrcfg(3, 0.0)
    eig, iters, ierr = run(m, SLOW, [1.0, 1.0])
    assert (iters, ierr) == (3, 1)
    # V was updated after sweeps 1 and 2 only: it is (1, 0.99**2)/norm
    q = 0.99**2
    norm = math.sqrt(1.0 + q * q)
    assert m.pwrget(1, 1) == pytest.approx(1.0 / norm, rel=1e-9)
    assert m.pwrget(1, 2) == pytest.approx(q / norm, rel=1e-9)
    assert eig == pytest.approx((1.0 + 0.99 * q * q) / (1.0 + q * q), rel=1e-9)


def test_exact_eigenvector_start_converges_in_two_sweeps(m):
    eig, iters, ierr = run(m, EXACT, [1.0, 1.0])
    assert (iters, ierr) == (2, 0)
    assert eig == pytest.approx(3.0, rel=1e-9)
    assert m.pwrget(1, 1) == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-9)
    assert m.pwrget(1, 2) == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-9)
    assert m.pwrget(2, 1) == pytest.approx(0.0, abs=1e-12)
    assert m.pwrget(2, 2) == pytest.approx(0.0, abs=1e-12)


def test_dominant_eigenvalue_of_a_diagonal_matrix(m):
    eig, iters, ierr = run(m, [[3.0, 0.0], [0.0, 1.0]], [1.0, 1.0], lda=4)
    assert ierr == 0
    assert eig == pytest.approx(3.0, rel=1e-8)
    assert m.pwrget(1, 1) == pytest.approx(1.0, rel=1e-4)
    assert abs(m.pwrget(1, 2)) < 1e-4


def test_zero_rayleigh_quotient_counts_as_converged_at_once(m):
    eig, iters, ierr = run(m, [[0.0, 1.0], [-1.0, 0.0]], [1.0, 0.0])
    assert (eig, iters, ierr) == (0.0, 1, 0)
    # W = A*V = (0, -1) and the residual equals W because the eigenvalue estimate is zero
    assert m.pwrget(1, 1) == 1.0
    assert m.pwrget(1, 2) == 0.0
    assert m.pwrget(2, 1) == 0.0
    assert m.pwrget(2, 2) == -1.0


def test_vanishing_product_means_eigenvalue_zero_even_if_the_estimate_just_changed(m):
    eig, iters, ierr = run(m, [[0.0, 0.0], [0.0, 0.0]], [1.0, 2.0])
    assert (eig, iters, ierr) == (0.0, 1, 0)
    assert m.pwrget(2, 1) == 0.0 and m.pwrget(2, 2) == 0.0
    # nilpotent matrix: sweep 1 gives the estimate 1/2 and V = (1, 0), sweep 2 gives A*V = 0
    eig, iters, ierr = run(m, [[0.0, 1.0], [0.0, 0.0]], [1.0, 1.0])
    assert (eig, iters, ierr) == (0.0, 2, 0)
    assert m.pwrget(1, 1) == pytest.approx(1.0, rel=1e-9)
    assert m.pwrget(1, 2) == 0.0
    assert m.pwrget(2, 1) == 0.0


def test_rejected_calls_are_counted_but_leave_the_stored_vector_alone(m):
    run(m, EXACT, [1.0, 1.0])
    kept = m.pwrget(1, 1)
    assert run(m, np.eye(3).tolist(), [0.0, 0.0, 0.0])[2] == 3
    big = np.eye(101)
    assert m.pwritr(big, 101, 101, np.ones(101), 0.0, 0, 0) == (0.0, 0, 2)
    assert m.pwritr(np.eye(2), 2, 0, np.ones(2), 5.0, 9, 9) == (0.0, 0, 2)
    assert m.pwritr(np.eye(2), 1, 2, np.ones(2), 5.0, 9, 9) == (0.0, 0, 2)
    assert m.pwrget(1, 1) == pytest.approx(kept, rel=1e-12)
    assert m.pwrget(1, 3) == 0.0  # beyond the order of the last successful run
    assert m.pwrget(3, 0) == 5.0


def test_call_statistics_accumulate_sweeps(m):
    run(m, EXACT, [1.0, 1.0])
    run(m, [[0.0, 1.0], [-1.0, 0.0]], [1.0, 0.0])
    assert m.pwrget(3, 99) == 2.0
    assert m.pwrget(4, 99) == 3.0
    # selector 7 and indices outside 1..NLAST give zero
    assert m.pwrget(7, 1) == 0.0
    assert m.pwrget(1, 0) == 0.0
    assert m.pwrget(1, 3) == 0.0
    assert m.pwrget(2, -1) == 0.0


def test_rayleigh_helper_refuses_a_mismatch_then_overwrites_the_residual_via_shared_storage(m):
    run(m, EXACT, [1.0, 1.0])
    assert m.pwrget(2, 1) == pytest.approx(0.0, abs=1e-12)
    # a mismatched order or leading dimension is refused without touching the storage
    assert m.pwrrq(np.eye(3), 3, 3) == 0.0
    assert m.pwrrq(np.eye(2), 1, 2) == 0.0
    assert m.pwrget(2, 1) == pytest.approx(0.0, abs=1e-12)
    doubled = [[4.0, 2.0], [2.0, 4.0]]
    assert m.pwrrq(np.array(doubled), 2, 2) == pytest.approx(6.0, rel=1e-9)
    # R and W are the same storage: the residual now reads A'*V = 6*V
    assert m.pwrget(2, 1) == pytest.approx(6.0 / math.sqrt(2.0), rel=1e-9)
    assert m.pwrget(2, 2) == pytest.approx(6.0 / math.sqrt(2.0), rel=1e-9)


def test_one_by_one_matrix_and_tolerance_comparison_edge_cases(m):
    eig, iters, ierr = run(m, [[2.5]], [3.0])
    assert (iters, ierr) == (2, 0)
    assert eig == pytest.approx(2.5, rel=1e-9)
    assert m.pwrget(1, 1) == pytest.approx(1.0, rel=1e-9)
    assert m.pwrget(2, 1) == pytest.approx(0.0, abs=1e-12)
    # eigenvalue estimate 0.375: the allowed change is tol*max(1, 0.375) = 0.4*1
    m.pwrcfg(50, 0.4)
    eig, iters, ierr = run(m, [[0.5, 0.0], [0.0, 0.25]], [1.0, 1.0])
    assert (iters, ierr) == (1, 0)
    assert eig == pytest.approx(0.375, rel=1e-9)
    # a change exactly equal to tol*max(1, |eig|) = 1.0*2 counts as converged
    m.pwrcfg(50, 1.0)
    eig, iters, ierr = run(m, [[2.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    assert (iters, ierr) == (1, 0)
    assert eig == 2.0


def test_order_one_hundred_is_the_largest_accepted(m):
    eig, iters, ierr = m.pwritr(np.eye(100), 100, 100, np.ones(100), 0.0, 0, 0)
    assert (iters, ierr) == (2, 0)
    assert eig == pytest.approx(1.0, rel=1e-9)
    assert m.pwrget(1, 100) == pytest.approx(0.1, rel=1e-9)
    assert m.pwrget(1, 101) == 0.0


def test_single_sweep_limit_keeps_v_and_forms_the_residual_from_the_first_product(m):
    m.pwrcfg(1, 0.0)
    eig, iters, ierr = run(m, SLOW, [1.0, 1.0])
    assert (iters, ierr) == (1, 1)
    assert eig == pytest.approx(0.995, rel=1e-9)
    s = 1.0 / math.sqrt(2.0)
    assert m.pwrget(1, 1) == pytest.approx(s, rel=1e-9)
    assert m.pwrget(1, 2) == pytest.approx(s, rel=1e-9)
    assert m.pwrget(2, 1) == pytest.approx(0.005 * s, rel=1e-9)
    assert m.pwrget(2, 2) == pytest.approx(-0.005 * s, rel=1e-9)


def test_rayleigh_helper_starts_from_a_cleared_work_vector(m):
    m.pwrcfg(3, 0.0)
    run(m, SLOW, [1.0, 1.0])  # exhausted sweeps leave a nonzero residual in the shared storage
    q = 0.99**2
    norm = math.sqrt(1.0 + q * q)
    assert m.pwrrq(np.array(SLOW), 2, 2) == pytest.approx(
        (1.0 + 0.99 * q * q) / (1.0 + q * q), rel=1e-9
    )
    assert m.pwrget(2, 1) == pytest.approx(1.0 / norm, rel=1e-9)
    assert m.pwrget(2, 2) == pytest.approx(0.99 * q / norm, rel=1e-9)


def test_rayleigh_helper_ignores_stale_entries_beyond_the_current_order(m):
    m.pwrcfg(1, 0.0)
    run(m, np.diag([5.0, 4.0, 3.0]).tolist(), [1.0, 1.0, 1.0])  # leaves V(3) and W(3) nonzero
    m.pwrcfg(50, 0.0)
    run(m, EXACT, [1.0, 1.0])
    assert m.pwrrq(np.array(EXACT), 2, 2) == pytest.approx(3.0, rel=1e-9)
