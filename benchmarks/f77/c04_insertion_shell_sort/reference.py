"""Reference port of ISORT, INSIDX and SHSORT."""

from __future__ import annotations

import numpy as np


def isort(n: int, ia: np.ndarray, idir: int) -> None:
    isg = 1 if idir >= 0 else -1  # ISIGN(1, IDIR)
    for i in range(2, n + 1):
        key = ia[i - 1]
        j = i - 1
        while j >= 1 and isg * ia[j - 1] > isg * key:
            ia[j] = ia[j - 1]
            j -= 1
        ia[j] = key


def insidx(n: int, a: np.ndarray, idx: np.ndarray) -> None:
    for i in range(1, n + 1):
        idx[i - 1] = i
    for i in range(2, n + 1):
        k = int(idx[i - 1])
        j = i - 1
        while j >= 1 and a[idx[j - 1] - 1] > a[k - 1]:
            idx[j] = idx[j - 1]
            j -= 1
        idx[j] = k


def shsort(n: int, a: np.ndarray, ncmp: int) -> int:
    ncmp = 0
    igap = 1
    while igap < n // 3:
        igap = 3 * igap + 1
    while igap >= 1:
        for i in range(igap + 1, n + 1):
            t = a[i - 1]
            j = i
            while j > igap:
                ncmp += 1
                if a[j - igap - 1] <= t:
                    break
                a[j - 1] = a[j - igap - 1]
                j -= igap
            a[j - 1] = t
        igap = igap // 3
    return ncmp
