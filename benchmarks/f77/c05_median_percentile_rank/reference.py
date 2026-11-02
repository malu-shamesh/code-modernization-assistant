"""Reference port of COPYSR, XMEDN, XPCTL, NEARRK and RANKS."""

from __future__ import annotations

import math

import numpy as np


def _nint(x: float) -> int:
    # Fortran NINT rounds half away from zero (Python round() rounds half to even)
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def copysr(n: int, x: np.ndarray, w: np.ndarray) -> None:
    for i in range(1, n + 1):
        w[i - 1] = x[i - 1]
    for i in range(2, n + 1):
        t = w[i - 1]
        j = i - 1
        while j >= 1 and w[j - 1] > t:
            w[j] = w[j - 1]
            j -= 1
        w[j] = t


def xmedn(n: int, x: np.ndarray, w: np.ndarray) -> float:
    if n < 1:
        return 0.0
    copysr(n, x, w)
    if n % 2 == 1:
        return float(w[(n + 1) // 2 - 1])
    return 0.5 * float(w[n // 2 - 1] + w[n // 2])


def xpctl(n: int, x: np.ndarray, w: np.ndarray, p: float) -> float:
    if n < 1:
        return 0.0
    copysr(n, x, w)
    pp = max(0.0, min(100.0, p))
    h = float(n - 1) * pp / 100.0 + 1.0
    k = int(h)
    if k >= n:
        return float(w[n - 1])
    f = h - float(k)
    return float(w[k - 1] + f * (w[k] - w[k - 1]))


def nearrk(n: int, p: float) -> int:
    r = _nint(p * float(n) / 100.0)
    return max(1, min(n, r))


def ranks(n: int, x: np.ndarray, r: np.ndarray, idx: np.ndarray, ntie: int) -> int:
    ntie = 0
    for i in range(1, n + 1):
        idx[i - 1] = i
    for i in range(2, n + 1):
        k = int(idx[i - 1])
        j = i - 1
        while j >= 1 and x[idx[j - 1] - 1] > x[k - 1]:
            idx[j] = idx[j - 1]
            j -= 1
        idx[j] = k
    i = 1
    while i <= n:
        j = i
        while j < n and x[idx[j] - 1] == x[idx[i - 1] - 1]:
            j += 1
        avg = 0.5 * float(i + j)
        if j > i:
            ntie += 1
        for ell in range(i, j + 1):
            r[idx[ell - 1] - 1] = avg
        i = j + 1
    return ntie
