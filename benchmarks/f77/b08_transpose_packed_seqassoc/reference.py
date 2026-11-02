"""Python port of the in-place transpose and packed-storage routines.

The Fortran code passes the array elements A(J+1,J) and A(J,J+1) to DSWAP, so the callee sees
the column-major storage sequence of A starting at that element.  A numpy array of any memory
layout cannot give a writable flat view, so the port addresses the storage sequence with
explicit column-major arithmetic: linear position p (1-based) is a[(p-1) % lda, (p-1) // lda].
"""

from __future__ import annotations

import numpy as np


def _kup(i: int, j: int) -> int:
    """Statement function: packed position of element (i, j) with i <= j."""
    # j*(j-1) is never negative, so floor division equals Fortran's truncating division
    return i + (j * (j - 1)) // 2


def dswap(n: int, dx: np.ndarray, incx: int, dy: np.ndarray, incy: int) -> None:
    if n <= 0:
        return
    ix = 1
    iy = 1
    if incx < 0:
        ix = (-n + 1) * incx + 1
    if incy < 0:
        iy = (-n + 1) * incy + 1
    for _ in range(n):
        dtemp = dx[ix - 1]
        dx[ix - 1] = dy[iy - 1]
        dy[iy - 1] = dtemp
        ix = ix + incx
        iy = iy + incy


def _swap_along_storage(
    n: int, a: np.ndarray, lda: int, startx: int, incx: int, starty: int, incy: int
) -> None:
    """DSWAP on the column-major storage sequence of ``a``: 1-based start positions, strides > 0."""
    for t in range(n):
        px = startx + t * incx - 1
        py = starty + t * incy - 1
        dtemp = a[px % lda, px // lda]
        a[px % lda, px // lda] = a[py % lda, py // lda]
        a[py % lda, py // lda] = dtemp


def sqtrns(a: np.ndarray, lda: int, n: int) -> None:
    for j in range(1, n):
        below = (j + 1) + (j - 1) * lda  # position of A(J+1,J)
        right = j + j * lda  # position of A(J,J+1)
        _swap_along_storage(n - j, a, lda, below, 1, right, lda)


def rctrns(a: np.ndarray, m: int, n: int, mark: np.ndarray) -> None:
    if m <= 1 or n <= 1:
        return
    mn = m * n
    mn1 = mn - 1
    for k in range(1, mn + 1):
        mark[k - 1] = False
    mark[0] = True
    mark[mn - 1] = True
    for k in range(2, mn1 + 1):
        if mark[k - 1]:
            continue
        i = k
        t = a[k - 1]
        while True:
            j = ((i - 1) * n) % mn1 + 1  # operands are positive, so % equals MOD
            t2 = a[j - 1]
            a[j - 1] = t
            t = t2
            mark[j - 1] = True
            i = j
            if i == k:
                break


def packup(a: np.ndarray, lda: int, n: int, ap: np.ndarray) -> None:
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            ap[_kup(i, j) - 1] = a[i - 1, j - 1]


def unpkup(ap: np.ndarray, a: np.ndarray, lda: int, n: int) -> None:
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            a[i - 1, j - 1] = ap[_kup(i, j) - 1]
            a[j - 1, i - 1] = ap[_kup(i, j) - 1]
