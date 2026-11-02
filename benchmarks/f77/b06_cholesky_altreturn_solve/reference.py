"""Python port of the Cholesky factorisation with an alternate return."""

from __future__ import annotations

import math

import numpy as np


def choldc(a: np.ndarray, lda: int, n: int, dmin: float) -> tuple[float, int]:
    """Returns (dmin, alt) where alt is 1 if the alternate return was taken, else 0."""
    dmin = 0.0
    for j in range(1, n + 1):
        s = a[j - 1, j - 1]
        for k in range(1, j):
            s = s - a[j - 1, k - 1] * a[j - 1, k - 1]
        if s <= 0.0:
            dmin = float(s)
            return dmin, 1
        a[j - 1, j - 1] = math.sqrt(s)
        if j == 1 or a[j - 1, j - 1] < dmin:
            dmin = float(a[j - 1, j - 1])
        for i in range(j + 1, n + 1):
            s = a[i - 1, j - 1]
            for k in range(1, j):
                s = s - a[i - 1, k - 1] * a[j - 1, k - 1]
            a[i - 1, j - 1] = s / a[j - 1, j - 1]
    return dmin, 0


def cholsl(a: np.ndarray, lda: int, n: int, b: np.ndarray) -> None:
    for i in range(1, n + 1):
        s = b[i - 1]
        for k in range(1, i):
            s = s - a[i - 1, k - 1] * b[k - 1]
        b[i - 1] = s / a[i - 1, i - 1]
    for i in range(n, 0, -1):
        s = b[i - 1]
        for k in range(i + 1, n + 1):
            s = s - a[k - 1, i - 1] * b[k - 1]
        b[i - 1] = s / a[i - 1, i - 1]


def spdsol(a: np.ndarray, lda: int, n: int, b: np.ndarray, dmin: float) -> tuple[float, int]:
    dmin, alt = choldc(a, lda, n, dmin)
    if alt == 1:
        return dmin, 1
    cholsl(a, lda, n, b)
    return dmin, 0


def choldt(a: np.ndarray, lda: int, n: int) -> float:
    result = 0.0
    for i in range(1, n + 1):
        result = result + 2.0 * math.log(a[i - 1, i - 1])
    return result
