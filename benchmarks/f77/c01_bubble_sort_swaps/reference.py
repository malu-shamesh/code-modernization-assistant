"""Reference port of BSORT and ISSRTD (bubble sort with a shrinking pass bound)."""

from __future__ import annotations

import numpy as np


def bsort(n: int, x: np.ndarray, nswap: int, npass: int) -> tuple[int, int]:
    nswap = 0
    npass = 0
    last = n
    while True:
        swappd = False
        lastsw = 0
        npass += 1
        # DO 20 I = 1, LAST - 1 : the bounds are evaluated once
        for i in range(1, last):
            if x[i - 1] <= x[i]:
                continue
            x[i - 1], x[i] = x[i], x[i - 1]
            nswap += 1
            swappd = True
            lastsw = i
        if not swappd:
            return nswap, npass
        last = lastsw


def issrtd(n: int, x: np.ndarray) -> bool:
    for i in range(1, n):
        if x[i - 1] > x[i]:
            return False
    return True
