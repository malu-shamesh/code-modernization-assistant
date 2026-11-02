"""Python port of DGEMV (strided matrix-vector product with leading dimension) and LSAME."""

from __future__ import annotations

import numpy as np


def _char1(c: str) -> str:
    """CHARACTER*1 semantics: pad with a blank or truncate to one character."""
    return (c + " ")[:1]


def lsame(ca: str, cb: str) -> bool:
    ca = _char1(ca)
    cb = _char1(cb)
    result = ca == cb
    if result:
        return True
    inta = ord(ca)
    intb = ord(cb)
    if 97 <= inta <= 122:
        inta = inta - 32
    if 97 <= intb <= 122:
        intb = intb - 32
    return inta == intb


def dgemv(
    trans: str,
    m: int,
    n: int,
    alpha: float,
    a: np.ndarray,
    lda: int,
    x: np.ndarray,
    incx: int,
    beta: float,
    y: np.ndarray,
    incy: int,
    info: int,
) -> int:
    info = 0
    if not lsame(trans, "N") and not lsame(trans, "T") and not lsame(trans, "C"):
        info = 1
    elif m < 0:
        info = 2
    elif n < 0:
        info = 3
    elif lda < max(1, m):
        info = 6
    elif incx == 0:
        info = 8
    elif incy == 0:
        info = 11
    if info != 0:
        return info
    if m == 0 or n == 0 or (alpha == 0.0 and beta == 1.0):
        return info
    if lsame(trans, "N"):
        lenx = n
        leny = m
    else:
        lenx = m
        leny = n
    kx = 1
    if incx < 0:
        kx = 1 - (lenx - 1) * incx
    ky = 1
    if incy < 0:
        ky = 1 - (leny - 1) * incy
    if beta != 1.0:
        iy = ky
        for _ in range(leny):
            if beta == 0.0:
                y[iy - 1] = 0.0
            else:
                y[iy - 1] = beta * y[iy - 1]
            iy = iy + incy
    if alpha == 0.0:
        return info
    if lsame(trans, "N"):
        jx = kx
        for j in range(1, n + 1):
            if x[jx - 1] != 0.0:
                temp = alpha * x[jx - 1]
                iy = ky
                for i in range(1, m + 1):
                    y[iy - 1] = y[iy - 1] + temp * a[i - 1, j - 1]
                    iy = iy + incy
            jx = jx + incx
    else:
        jy = ky
        for j in range(1, n + 1):
            temp = 0.0
            ix = kx
            for i in range(1, m + 1):
                temp = temp + a[i - 1, j - 1] * x[ix - 1]
                ix = ix + incx
            y[jy - 1] = y[jy - 1] + alpha * temp
            jy = jy + incy
    return info
