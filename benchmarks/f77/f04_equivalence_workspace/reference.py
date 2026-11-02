"""Reference port of legacy.f (EQUIVALENCE views over one COMMON workspace)."""

from __future__ import annotations

import numpy as np

# COMMON /SCRWRK/ W(24); A, V and T are EQUIVALENCEd onto it, so they must be views, not copies.
_w = np.zeros(24, dtype=np.float64)
_v = _w[0:16]
_a = _v.reshape((4, 4), order="F")
_t = _w[16:24]

if not (np.shares_memory(_a, _w) and np.shares_memory(_v, _w) and np.shares_memory(_t, _w)):
    raise RuntimeError("EQUIVALENCE views must alias the COMMON workspace")


def sclear():
    for i in range(1, 25):
        _w[i - 1] = 0.0


def svset(n, x):
    nn = min(n, 16)
    for i in range(1, nn + 1):
        _v[i - 1] = x[i - 1]


def svget(k):
    return float(_v[k - 1])


def smset(i, j, val):
    _a[i - 1, j - 1] = val


def smget(i, j):
    return float(_a[i - 1, j - 1])


def sblksm(n):
    total = 0.0
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            total = total + _a[i - 1, j - 1]
    return float(total)


def strnsp():
    for j in range(2, 5):
        for i in range(1, j):
            tmp = _a[i - 1, j - 1]
            _a[i - 1, j - 1] = _a[j - 1, i - 1]
            _a[j - 1, i - 1] = tmp


def sstash(j):
    for i in range(1, 5):
        _t[i - 1] = _a[i - 1, j - 1]
        _a[i - 1, j - 1] = 0.0


def sunsth(j):
    for i in range(1, 5):
        _a[i - 1, j - 1] = _t[i - 1]
