"""Python port of the LINPACK-style LU factorisation, solver and determinant."""

from __future__ import annotations

import numpy as np


def lufac(a: np.ndarray, lda: int, n: int, ipvt: np.ndarray, info: int) -> int:
    info = 0
    if n < 1:
        return info
    nm1 = n - 1
    for k in range(1, nm1 + 1):
        l = k
        for i in range(k + 1, n + 1):
            if abs(a[i - 1, k - 1]) > abs(a[l - 1, k - 1]):
                l = i
        ipvt[k - 1] = l
        if a[l - 1, k - 1] == 0.0:
            info = k
            continue
        if l != k:
            t = a[l - 1, k - 1]
            a[l - 1, k - 1] = a[k - 1, k - 1]
            a[k - 1, k - 1] = t
        t = -1.0 / a[k - 1, k - 1]
        for i in range(k + 1, n + 1):
            a[i - 1, k - 1] = t * a[i - 1, k - 1]
        for j in range(k + 1, n + 1):
            t = a[l - 1, j - 1]
            if l != k:
                a[l - 1, j - 1] = a[k - 1, j - 1]
                a[k - 1, j - 1] = t
            for i in range(k + 1, n + 1):
                a[i - 1, j - 1] = a[i - 1, j - 1] + t * a[i - 1, k - 1]
    ipvt[n - 1] = n
    if a[n - 1, n - 1] == 0.0:
        info = n
    return info


def lusol(a: np.ndarray, lda: int, n: int, ipvt: np.ndarray, b: np.ndarray) -> None:
    nm1 = n - 1
    for k in range(1, nm1 + 1):
        l = int(ipvt[k - 1])
        t = b[l - 1]
        if l != k:
            b[l - 1] = b[k - 1]
            b[k - 1] = t
        for i in range(k + 1, n + 1):
            b[i - 1] = b[i - 1] + t * a[i - 1, k - 1]
    for kb in range(1, n + 1):
        k = n + 1 - kb
        b[k - 1] = b[k - 1] / a[k - 1, k - 1]
        t = -b[k - 1]
        for i in range(1, k):
            b[i - 1] = b[i - 1] + t * a[i - 1, k - 1]


def ludet(a: np.ndarray, lda: int, n: int, ipvt: np.ndarray) -> float:
    result = 1.0
    for k in range(1, n + 1):
        if ipvt[k - 1] != k:
            result = -result
        result = result * a[k - 1, k - 1]
    return float(result)


def luslv(a: np.ndarray, lda: int, n: int, b: np.ndarray, ipvt: np.ndarray, info: int) -> int:
    info = lufac(a, lda, n, ipvt, info)
    if info == 0:
        lusol(a, lda, n, ipvt, b)
    return info
