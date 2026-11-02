"""Reference port of LINREG, FITTRN and RSSQ (least-squares line)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

TINY = 1.0e-300


def _sq(z: float) -> float:
    return z * z


def linreg(
    n: int, x: np.ndarray, y: np.ndarray, a: float, b: float, r2: float, ierr: int
) -> tuple[float, float, float, int]:
    a = 0.0
    b = 0.0
    r2 = 0.0
    ierr = 0
    if n < 2:
        ierr = 1
        return a, b, r2, ierr
    sx = 0.0
    sy = 0.0
    for i in range(1, n + 1):
        sx = sx + float(x[i - 1])
        sy = sy + float(y[i - 1])
    xm = sx / float(n)
    ym = sy / float(n)
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for i in range(1, n + 1):
        dx = float(x[i - 1]) - xm
        dy = float(y[i - 1]) - ym
        sxx = sxx + _sq(dx)
        sxy = sxy + dx * dy
        syy = syy + _sq(dy)
    if abs(sxx) <= TINY:
        ierr = 2
        return a, b, r2, ierr
    b = sxy / sxx
    a = ym - b * xm
    if syy <= TINY:
        r2 = 1.0
    else:
        r2 = sxy * sxy / (sxx * syy)
    return a, b, r2, ierr


def fittrn(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    f: Callable[[float], float],
    xt: np.ndarray,
    a: float,
    b: float,
    r2: float,
    ierr: int,
) -> tuple[float, float, float, int]:
    for i in range(1, n + 1):
        xt[i - 1] = f(float(x[i - 1]))
    a0, b0, rr, ie = linreg(n, xt, y, 0.0, 0.0, 0.0, 0)
    a = a0
    b = b0
    r2 = rr
    ierr = ie
    return a, b, r2, ierr


def rssq(n: int, x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    total = 0.0
    for i in range(1, n + 1):
        e = float(y[i - 1]) - (a + b * float(x[i - 1]))
        total = total + e * e
    return total
