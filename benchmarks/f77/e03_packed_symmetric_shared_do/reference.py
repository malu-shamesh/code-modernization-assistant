"""Hand-written Python port of legacy.f (packed symmetric matrix utilities)."""

from __future__ import annotations


def pkidx(i, j):
    ii = min(i, j)
    jj = max(i, j)
    return jj * (jj - 1) // 2 + ii


def packut(n, a, lda, ap):
    k = 0
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            k += 1
            ap[k - 1] = a[i - 1, j - 1]


def unpksy(n, ap, a, lda):
    k = 0
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            k += 1
            a[i - 1, j - 1] = ap[k - 1]
            a[j - 1, i - 1] = ap[k - 1]


def spmv(n, ap, x, y):
    for i in range(1, n + 1):
        y[i - 1] = 0.0
        for j in range(1, n + 1):
            y[i - 1] = y[i - 1] + ap[pkidx(i, j) - 1] * x[j - 1]


def ntrip(n, lim):
    count = 0
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            for k in range(j, n + 1):
                if i + j + k <= lim:
                    count += 1
    return count
