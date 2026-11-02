"""Reference port of MOMENT, DCOVAR and MINMAX (double precision sample statistics)."""

from __future__ import annotations

import math

import numpy as np

ZERO = 0.0


def moment(
    n: int, x: np.ndarray, xmean: float, xvar: float, xsd: float, ierr: int
) -> tuple[float, float, float, int]:
    xmean = ZERO
    xvar = ZERO
    xsd = ZERO
    ierr = 0
    if n < 1:
        ierr = 1
        return xmean, xvar, xsd, ierr
    s1 = ZERO
    for i in range(1, n + 1):
        s1 = s1 + x[i - 1]
    xmean = s1 / float(n)
    if n == 1:
        ierr = 2
        return xmean, xvar, xsd, ierr
    s1 = ZERO
    s2 = ZERO
    for i in range(1, n + 1):
        d = x[i - 1] - xmean
        s1 = s1 + d
        s2 = s2 + d * d
    xvar = (s2 - s1 * s1 / float(n)) / float(n - 1)
    xsd = math.sqrt(xvar)
    return xmean, xvar, xsd, ierr


def dcovar(n: int, x: np.ndarray, y: np.ndarray) -> float:
    if n < 2:
        return 0.0
    sx = 0.0
    sy = 0.0
    for i in range(1, n + 1):
        sx = sx + x[i - 1]
        sy = sy + y[i - 1]
    sx = sx / float(n)
    sy = sy / float(n)
    sxy = 0.0
    for i in range(1, n + 1):
        sxy = sxy + (x[i - 1] - sx) * (y[i - 1] - sy)
    return sxy / float(n - 1)


def minmax(
    n: int, x: np.ndarray, xmin: float, xmax: float, imin: int, imax: int
) -> tuple[float, float, int, int]:
    xmin = 0.0
    xmax = 0.0
    imin = 0
    imax = 0
    if n < 1:
        return xmin, xmax, imin, imax
    xmin = x[0]
    xmax = x[0]
    imin = 1
    imax = 1
    for i in range(2, n + 1):
        if x[i - 1] < xmin:
            xmin = x[i - 1]
            imin = i
        if x[i - 1] > xmax:
            xmax = x[i - 1]
            imax = i
    return float(xmin), float(xmax), imin, imax
