"""Python port of the vector norm and pivot-search kernels."""

from __future__ import annotations

import math

import numpy as np


def dasum(n: int, dx: np.ndarray, incx: int) -> float:
    if n <= 0 or incx <= 0:
        return 0.0
    dtemp = 0.0
    for i in range(1, n * incx + 1, incx):
        dtemp = dtemp + abs(dx[i - 1])
    return float(dtemp)


def idamax(n: int, dx: np.ndarray, incx: int) -> int:
    """Element number 1..n of the first entry of largest magnitude (0 if none)."""
    if n < 1 or incx <= 0:
        return 0
    result = 1
    if n == 1:
        return result
    ix = 1
    dmax = abs(dx[0])
    for i in range(2, n + 1):
        ix = ix + incx
        if abs(dx[ix - 1]) > dmax:
            result = i
            dmax = abs(dx[ix - 1])
    return result


def dnrm2(n: int, dx: np.ndarray, incx: int) -> float:
    if n < 1 or incx < 1:
        return 0.0
    scale = 0.0
    ssq = 1.0
    for ix in range(1, 1 + (n - 1) * incx + 1, incx):
        if dx[ix - 1] != 0.0:
            absxi = abs(dx[ix - 1])
            if scale < absxi:
                ssq = 1.0 + ssq * (scale / absxi) ** 2
                scale = absxi
            else:
                ssq = ssq + (absxi / scale) ** 2
    return float(scale * math.sqrt(ssq))


def dnrmi(n: int, dx: np.ndarray, incx: int) -> float:
    k = idamax(n, dx, incx)
    if k == 0:
        return 0.0
    return float(abs(dx[1 + (k - 1) * incx - 1]))
