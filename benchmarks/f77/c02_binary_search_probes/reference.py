"""Reference port of the binary search routines IBSRCH, ILBND and ICOUNT."""

from __future__ import annotations

import numpy as np


def ibsrch(n: int, key: int, ia: np.ndarray, nprobe: int) -> tuple[int, int]:
    nprobe = 0
    lo = 1
    hi = n
    while lo <= hi:
        mid = (lo + hi) // 2  # lo + hi is positive, so // equals Fortran truncation
        nprobe += 1
        if ia[mid - 1] == key:
            return mid, nprobe
        if ia[mid - 1] < key:
            lo = mid + 1
        if ia[mid - 1] > key:
            hi = mid - 1
    return -lo, nprobe


def ilbnd(n: int, key: int, ia: np.ndarray) -> int:
    lo = 1
    hi = n + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ia[mid - 1] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def icount(n: int, key: int, ia: np.ndarray) -> int:
    return ilbnd(n, key + 1, ia) - ilbnd(n, key, ia)
