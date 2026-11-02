"""Reference port of the permutation units, BLOCK DATA BDPERM and the COMMON blocks."""

from __future__ import annotations

import numpy as np


class _Counters:
    """COMMON /PRMCNT/ NGEN, NRNK, initialised by BLOCK DATA BDPERM."""

    def __init__(self) -> None:
        self.ngen = 0
        self.nrnk = 0


# COMMON /PRMFAC/ IFACT(0:12): index k is stored at position k
IFACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600]
_prmcnt = _Counters()


def nxtprm(n: int, ip: np.ndarray, last: bool) -> bool:
    last = False
    # DO 10 I = N - 1, 1, -1 : the variable keeps start + trips * step afterwards
    i = n - 1
    trips = max(0, n - 1)
    found = False
    for _ in range(trips):
        if ip[i - 1] < ip[i]:
            found = True
            break
        i -= 1
    if found:
        # DO 30 J = N, I + 1, -1 with an early exit
        j = n
        for _ in range(max(0, n - i)):
            if ip[j - 1] > ip[i - 1]:
                break
            j -= 1
        ip[i - 1], ip[j - 1] = ip[j - 1], ip[i - 1]
        _prmcnt.ngen += 1
    else:
        last = True
    lo = i + 1  # i is 0 after the loop ran out, so the whole vector is reversed
    hi = n
    while lo < hi:
        ip[lo - 1], ip[hi - 1] = ip[hi - 1], ip[lo - 1]
        lo += 1
        hi -= 1
    return last


def prmrnk(n: int, ip: np.ndarray) -> int:
    if n < 0 or n > 12:
        return -1
    _prmcnt.nrnk += 1
    ir = 0
    for i in range(1, n):
        icnt = 0
        for j in range(i + 1, n + 1):
            if ip[j - 1] < ip[i - 1]:
                icnt += 1
        ir += icnt * IFACT[n - i]
    return ir


def prmunr(n: int, k: int, ip: np.ndarray, ierr: int) -> int:
    ierr = 0
    if n < 0 or n > 12:
        ierr = 1
        return ierr
    if k < 0 or k >= IFACT[n]:
        ierr = 2
        return ierr
    iav = list(range(1, n + 1))
    irem = k
    for i in range(1, n + 1):
        # irem is never negative here, so // equals Fortran integer division
        idx = irem // IFACT[n - i]
        irem = irem % IFACT[n - i]
        ip[i - 1] = iav[idx]
        del iav[idx]
    return ierr


def _fortran_mod(a: int, p: int) -> int:
    r = abs(a) % abs(p)
    return -r if a < 0 else r


def prmskp(n: int, ip: np.ndarray, istep: int, ierr: int) -> int:
    ierr = 1
    if n < 0 or n > 12:
        return ierr
    nf = IFACT[n]
    k = _fortran_mod(prmrnk(n, ip) + istep, nf)
    if k < 0:
        k = k + nf
    ierr = prmunr(n, k, ip, ierr)
    return ierr


def ncount(iwhich: int) -> int:
    if iwhich == 1:
        return _prmcnt.ngen
    return _prmcnt.nrnk
