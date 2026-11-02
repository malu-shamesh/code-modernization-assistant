"""Reference port of CUMSUM, IFIRST, CUMCOL, RCOPY and MRGSRT."""

from __future__ import annotations

import numpy as np


def cumsum(n: int, x: np.ndarray, s: np.ndarray) -> None:
    acc = 0.0
    for i in range(1, n + 1):
        acc = acc + x[i - 1]
        s[i - 1] = acc


def ifirst(n: int, x: np.ndarray, thresh: float) -> int:
    acc = 0.0
    for i in range(1, n + 1):
        acc = acc + x[i - 1]
        if acc >= thresh:
            return i
    return 0


def cumcol(lda: int, m: int, n: int, a: np.ndarray) -> None:
    # a has shape (lda, n); only rows 1..m are summed
    for j in range(1, n + 1):
        for i in range(2, m + 1):
            a[i - 1, j - 1] = a[i - 1, j - 1] + a[i - 2, j - 1]


def rcopy(n: int, a: np.ndarray, b: np.ndarray) -> None:
    for i in range(1, n + 1):
        b[i - 1] = a[i - 1]


def mrgsrt(na: int, a: np.ndarray, nb: int, b: np.ndarray, c: np.ndarray, nc: int) -> int:
    i = 1
    j = 1
    k = 1
    while i <= na:
        if j > nb:
            # A(I) and C(K) are passed as sub-array starts: use views
            rcopy(na - i + 1, a[i - 1 :], c[k - 1 :])
            nc = na + nb
            return nc
        if b[j - 1] < a[i - 1]:
            c[k - 1] = b[j - 1]
            j += 1
        else:
            c[k - 1] = a[i - 1]
            i += 1
        k += 1
    rcopy(nb - j + 1, b[j - 1 :], c[k - 1 :])
    nc = na + nb
    return nc
