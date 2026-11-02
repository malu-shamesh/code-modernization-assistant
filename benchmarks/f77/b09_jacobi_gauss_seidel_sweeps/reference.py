"""Python port of the Jacobi / Gauss-Seidel driver ITSOLV and the residual norm RESNRM."""

from __future__ import annotations

import numpy as np


def d1mach(i: int) -> float:
    """Double precision machine constants (SLATEC numbering) for IEEE 754 doubles."""
    table = {
        1: 2.2250738585072014e-308,  # smallest positive normal number
        2: 1.7976931348623157e308,  # largest number
        3: 2.0**-53,  # smallest relative spacing
        4: 2.0**-52,  # largest relative spacing
        5: 0.30102999566398120,  # log10(2)
    }
    return table[i]


def itsolv(
    method: int,
    a: np.ndarray,
    lda: int,
    n: int,
    b: np.ndarray,
    x: np.ndarray,
    xn: np.ndarray,
    tol: float,
    maxit: int,
    iter: int,
    err: float,
    ierr: int,
) -> tuple[float, int, float, int]:
    """Returns (tol, iter, err, ierr), the scalar arguments assigned by the routine."""
    ierr = 0
    iter = 0
    err = 0.0
    if n < 1 or lda < n:
        ierr = 2
        return tol, iter, err, ierr
    if tol <= 0.0:
        tol = 10.0 * d1mach(4)
    for i in range(1, n + 1):
        if a[i - 1, i - 1] == 0.0:
            ierr = 4
            return tol, iter, err, ierr
    # DO ITER = 1, MAXIT: the variable is set to 1 even for a zero-trip loop and ends at
    # 1 + trips when the loop runs to completion
    iter = 1
    trips = max(0, maxit)
    while trips > 0:
        if method == 1:
            for i in range(1, n + 1):
                s = b[i - 1]
                for j in range(1, n + 1):
                    if j != i:
                        s = s - a[i - 1, j - 1] * x[j - 1]
                xn[i - 1] = s / a[i - 1, i - 1]
            err = 0.0
            for i in range(1, n + 1):
                err = max(err, abs(xn[i - 1] - x[i - 1]))
                x[i - 1] = xn[i - 1]
        elif method == 2:
            err = 0.0
            for i in range(1, n + 1):
                s = b[i - 1]
                for j in range(1, n + 1):
                    if j != i:
                        s = s - a[i - 1, j - 1] * x[j - 1]
                s = s / a[i - 1, i - 1]
                err = max(err, abs(s - x[i - 1]))
                x[i - 1] = s
        else:
            ierr = 3
            return tol, iter, err, ierr
        err = float(err)
        if err - tol <= 0.0:
            ierr = 0
            return tol, iter, err, ierr
        iter += 1
        trips -= 1
    ierr = 1
    return tol, iter, err, ierr


def resnrm(a: np.ndarray, lda: int, n: int, b: np.ndarray, x: np.ndarray) -> float:
    result = 0.0
    for i in range(1, n + 1):
        s = b[i - 1]
        for j in range(1, n + 1):
            s = s - a[i - 1, j - 1] * x[j - 1]
        result = max(result, abs(s))
    return float(result)
