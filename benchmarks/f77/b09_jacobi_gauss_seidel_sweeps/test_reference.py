import target_module
import numpy as np
import pytest

A2 = [[4.0, 1.0], [1.0, 3.0]]
B2 = [5.0, 4.0]  # exact solution (1, 1)


def solve(method, rows, rhs, *, x0=None, tol=1e-12, maxit=100, lda=None):
    rows = np.array(rows, dtype=float)
    n = rows.shape[0]
    lda = n if lda is None else lda
    a = np.full((lda, n), 99.0)
    a[:n, :] = rows
    b = np.array(rhs, dtype=float)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    xn = np.zeros(n)
    out = target_module.itsolv(method, a, lda, n, b, x, xn, tol, maxit, -7, -7.0, -7)
    return x, out


def test_one_jacobi_sweep_uses_only_the_old_iterate():
    x, (tol, it, err, ierr) = solve(1, A2, B2, maxit=1)
    assert x == pytest.approx([1.25, 4.0 / 3.0], rel=1e-9)
    assert err == pytest.approx(4.0 / 3.0, rel=1e-9)
    # the single trip is used up without convergence, so the DO variable ends at MAXIT+1
    assert (it, ierr, tol) == (2, 1, 1e-12)


def test_one_gauss_seidel_sweep_reuses_new_components():
    x, (tol, it, err, ierr) = solve(2, A2, B2, maxit=1)
    assert x == pytest.approx([1.25, 11.0 / 12.0], rel=1e-9)
    assert err == pytest.approx(1.25, rel=1e-9)
    assert (it, ierr) == (2, 1)


def test_jacobi_converges_at_sweep_two_for_loose_tolerance():
    x, (tol, it, err, ierr) = solve(1, A2, B2, tol=0.5)
    assert (it, ierr) == (2, 0)
    assert err == pytest.approx(5.0 / 12.0, rel=1e-9)
    assert x == pytest.approx([11.0 / 12.0, 11.0 / 12.0], rel=1e-9)


def test_gauss_seidel_converges_at_sweep_two_for_loose_tolerance():
    x, (tol, it, err, ierr) = solve(2, A2, B2, tol=0.5)
    assert (it, ierr) == (2, 0)
    assert err == pytest.approx(11.0 / 48.0, rel=1e-9)
    assert x == pytest.approx([49.0 / 48.0, 143.0 / 144.0], rel=1e-9)


def test_exit_by_branch_keeps_the_loop_variable_at_the_converged_sweep():
    x, (tol, it, err, ierr) = solve(1, [[2.0, 0.0], [0.0, 4.0]], [2.0, 4.0], tol=1e-9, maxit=10)
    assert (it, ierr, err) == (2, 0, 0.0)
    assert x.tolist() == [1.0, 1.0]


def test_first_sweep_convergence_and_change_equal_to_tolerance():
    x, (tol, it, err, ierr) = solve(1, [[2.0, 0.0], [0.0, 4.0]], [2.0, 4.0], tol=1.0)
    assert (it, err, ierr) == (1, 1.0, 0)
    x, (tol, it, err, ierr) = solve(
        2, [[2.0, 0.0], [0.0, 4.0]], [2.0, 4.0], x0=[1.0, 1.0], tol=1e-9
    )
    assert (it, err, ierr) == (1, 0.0, 0)


def test_default_tolerance_comes_from_machine_precision():
    for requested in (0.0, -1.0):
        x, (tol, it, err, ierr) = solve(1, [[2.0, 0.0], [0.0, 4.0]], [2.0, 4.0], tol=requested)
        assert tol == pytest.approx(10.0 * 2.0**-52, rel=1e-9, abs=0.0)
        assert (it, ierr) == (2, 0)


def test_zero_sweep_limit_leaves_the_loop_variable_at_its_start_value():
    x, (tol, it, err, ierr) = solve(1, A2, B2, maxit=0)
    assert (it, err, ierr) == (1, 0.0, 1)
    assert x.tolist() == [0.0, 0.0]
    # an unknown method is not noticed either, because no sweep runs
    x, (tol, it, err, ierr) = solve(7, A2, B2, maxit=0)
    assert (it, ierr) == (1, 1)


def test_divergent_iteration_runs_out_of_sweeps():
    for method in (1, 2):
        x, (tol, it, err, ierr) = solve(method, [[1.0, 2.0], [3.0, 1.0]], [1.0, 1.0], maxit=5)
        assert (it, ierr) == (6, 1)
        assert err > 1.0


@pytest.mark.parametrize("method", [0, 3, -1])
def test_unknown_method_falls_through_the_computed_goto(method):
    x, (tol, it, err, ierr) = solve(method, A2, B2, x0=[0.5, 0.5], maxit=5)
    assert (it, err, ierr) == (1, 0.0, 3)
    assert x.tolist() == [0.5, 0.5]


def test_zero_diagonal_element_is_rejected_before_iterating():
    x, (tol, it, err, ierr) = solve(1, [[0.0, 1.0], [1.0, 2.0]], [1.0, 1.0], x0=[3.0, 3.0])
    assert (it, err, ierr) == (0, 0.0, 4)
    assert x.tolist() == [3.0, 3.0]


def test_bad_dimensions_are_rejected():
    for n_rows, lda in ((0, 1), (2, 1)):
        a = np.zeros((2, 2))
        b = np.zeros(2)
        x = np.zeros(2)
        out = target_module.itsolv(1, a, lda, n_rows, b, x, np.zeros(2), 0.25, 5, 9, 9.0, 9)
        assert out == (0.25, 0, 0.0, 2)


def test_negative_diagonal_and_padded_leading_dimension():
    x, (tol, it, err, ierr) = solve(2, [[-2.0, 0.0], [0.0, -4.0]], [2.0, 4.0], tol=1e-9, lda=5)
    assert (it, ierr) == (2, 0)
    assert x.tolist() == [-1.0, -1.0]


def test_tridiagonal_system_converges_and_gauss_seidel_is_faster():
    rows = [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]]
    rhs = [2.0, 4.0, 10.0]  # exact solution (1, 2, 3)
    xj, (_, itj, _, ierrj) = solve(1, rows, rhs, tol=1e-12, maxit=200)
    xg, (_, itg, _, ierrg) = solve(2, rows, rhs, tol=1e-12, maxit=200)
    assert ierrj == 0 and ierrg == 0
    assert xj == pytest.approx([1.0, 2.0, 3.0], rel=1e-9)
    assert xg == pytest.approx([1.0, 2.0, 3.0], rel=1e-9)
    assert itg < itj < 200


def test_resnrm_is_the_maximum_norm_of_the_residual():
    a = np.array(A2)
    b = np.array(B2)
    assert target_module.resnrm(a, 2, 2, b, np.array([1.0, 1.0])) == 0.0
    assert target_module.resnrm(a, 2, 2, b, np.zeros(2)) == pytest.approx(5.0, rel=1e-9)
    assert target_module.resnrm(a, 2, 2, b, np.array([1.0, 0.0])) == pytest.approx(3.0, rel=1e-9)


@pytest.mark.parametrize("method", [1, 2])
def test_one_by_one_system(method):
    x, (tol, it, err, ierr) = solve(method, [[2.0]], [6.0])
    assert (it, err, ierr) == (2, 0.0, 0)
    assert x.tolist() == [3.0]
