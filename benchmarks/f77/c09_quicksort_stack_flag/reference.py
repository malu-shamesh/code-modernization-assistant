"""Reference port of QSORTX (iterative quicksort with an explicit stack and an alternate return)."""

from __future__ import annotations

import numpy as np


def qsortx(n: int, x: np.ndarray, kflag: int, ndup: int) -> tuple[int, int]:
    """Return (ndup, alt) where alt is 0 for a normal RETURN and 1 for RETURN 1."""
    ndup = 0
    if n < 1:
        return ndup, 1
    if abs(kflag) != 1:
        return ndup, 1
    # ISTK(0:63) shares storage with the stack pointer ISP, so ISP is simply istk[0]
    istk = [0] * 64
    if kflag == -1:
        for i in range(1, n + 1):
            x[i - 1] = -x[i - 1]
    lo = 1
    hi = n
    while True:
        if hi > lo:
            mid = (lo + hi) // 2
            if x[lo - 1] > x[mid - 1]:
                x[lo - 1], x[mid - 1] = x[mid - 1], x[lo - 1]
            if x[mid - 1] > x[hi - 1]:
                x[mid - 1], x[hi - 1] = x[hi - 1], x[mid - 1]
                if x[lo - 1] > x[mid - 1]:
                    x[lo - 1], x[mid - 1] = x[mid - 1], x[lo - 1]
            piv = x[mid - 1]
            i = lo - 1
            j = hi + 1
            while True:
                i += 1
                while x[i - 1] - piv < 0:
                    i += 1
                j -= 1
                while x[j - 1] - piv > 0:
                    j -= 1
                if i >= j:
                    break
                x[i - 1], x[j - 1] = x[j - 1], x[i - 1]
            istk[0] += 1
            if j - lo > hi - j - 1:
                istk[2 * istk[0] - 1] = lo
                istk[2 * istk[0]] = j
                lo = j + 1
            else:
                istk[2 * istk[0] - 1] = j + 1
                istk[2 * istk[0]] = hi
                hi = j
            continue
        if istk[0] == 0:
            break
        lo = istk[2 * istk[0] - 1]
        hi = istk[2 * istk[0]]
        istk[0] -= 1
    if kflag <= 0:
        for i in range(1, n + 1):
            x[i - 1] = -x[i - 1]
    for i in range(2, n + 1):
        if x[i - 1] == x[i - 2]:
            ndup += 1
    return ndup, 0
