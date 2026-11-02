"""Hand-written Python port of legacy.f (assigned GOTO local blocks)."""

from __future__ import annotations


def satacc(n, ix, lo, hi, alo, ahi, nclip, isum):
    def clamp(v, vlo, vhi):
        nonlocal nclip
        if v < vlo:
            nclip += 1
            return vlo
        if v > vhi:
            nclip += 1
            return vhi
        return v

    nclip = 0
    isum = 0
    for i in range(1, n + 1):
        v = clamp(int(ix[i - 1]), lo, hi)
        isum = clamp(isum + v, alo, ahi)
    return nclip, isum


def runlen(n, ix, nrun, longst, istart):
    nrun = 0
    longst = 0
    istart = 0
    if n <= 0:
        return nrun, longst, istart
    ibeg = 1
    i = 1

    def close_run(ibeg, i):
        nonlocal nrun, longst, istart
        nrun += 1
        ilen = i - ibeg
        if ilen > longst:
            longst = ilen
            istart = ibeg

    while True:
        i += 1
        if i > n:
            close_run(ibeg, i)
            return nrun, longst, istart
        if ix[i - 1] != ix[i - 2]:
            close_run(ibeg, i)
            ibeg = i
