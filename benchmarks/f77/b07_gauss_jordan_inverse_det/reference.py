"""Python port of the Gauss-Jordan inverse with full pivoting."""

from __future__ import annotations

import numpy as np

NMAX = 50


def gjinv(a: np.ndarray, lda: int, n: int, det: float, ierr: int) -> tuple[float, int]:
    ierr = 0
    det = 1.0
    if n < 1 or n > NMAX:
        ierr = 2
        return det, ierr
    ipiv = [0] * NMAX
    indxr = [0] * NMAX
    indxc = [0] * NMAX
    irow = 1
    icol = 1
    for i in range(1, n + 1):
        big = 0.0
        for j in range(1, n + 1):
            if ipiv[j - 1] != 1:
                for k in range(1, n + 1):
                    if ipiv[k - 1] == 0:
                        if abs(a[j - 1, k - 1]) >= big:
                            big = abs(a[j - 1, k - 1])
                            irow = j
                            icol = k
        ipiv[icol - 1] = ipiv[icol - 1] + 1
        if irow != icol:
            for l in range(1, n + 1):
                t = a[irow - 1, l - 1]
                a[irow - 1, l - 1] = a[icol - 1, l - 1]
                a[icol - 1, l - 1] = t
            det = -det
        indxr[i - 1] = irow
        indxc[i - 1] = icol
        if a[icol - 1, icol - 1] == 0.0:
            ierr = 1
            det = 0.0
            return det, ierr
        det = det * a[icol - 1, icol - 1]
        pivinv = 1.0 / a[icol - 1, icol - 1]
        a[icol - 1, icol - 1] = 1.0
        for l in range(1, n + 1):
            a[icol - 1, l - 1] = a[icol - 1, l - 1] * pivinv
        for ll in range(1, n + 1):
            if ll != icol:
                dum = a[ll - 1, icol - 1]
                a[ll - 1, icol - 1] = 0.0
                for l in range(1, n + 1):
                    a[ll - 1, l - 1] = a[ll - 1, l - 1] - a[icol - 1, l - 1] * dum
    for l in range(n, 0, -1):
        if indxr[l - 1] != indxc[l - 1]:
            for k in range(1, n + 1):
                t = a[k - 1, indxr[l - 1] - 1]
                a[k - 1, indxr[l - 1] - 1] = a[k - 1, indxc[l - 1] - 1]
                a[k - 1, indxc[l - 1] - 1] = t
    return float(det), ierr


def mxresi(a: np.ndarray, ainv: np.ndarray, lda: int, n: int) -> float:
    result = 0.0
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            s = 0.0
            for k in range(1, n + 1):
                s = s + a[i - 1, k - 1] * ainv[k - 1, j - 1]
            if i == j:
                s = s - 1.0
            result = max(result, abs(s))
    return float(result)
