"""Hand-written Python port of legacy.f (table-driven number scanner, BLOCK DATA table)."""

from __future__ import annotations

import numpy as np

# BLOCK DATA FSMDAT / COMMON /FSMTAB/: DATA fills ITRANS(5,4) in column-major order.
_itrans = np.array([2, 0, 0, 0, 0, 3, 3, 3, 5, 5, 4, 4, 5, 0, 0] + [0] * 5, dtype=np.int64).reshape(
    (5, 4), order="F"
)
_iacc = np.array([0, 0, 1, 0, 1], dtype=np.int64)


def nclass(c):
    c = (c + " ")[:1]
    cls = 4
    if c == "+" or c == "-":
        cls = 1
    if "0" <= c <= "9":
        cls = 2
    if c == ".":
        cls = 3
    return cls


def numscn(str, ok, ipos, ndig):
    ndig = 0
    ipos = 0
    n = len(str)
    while n > 0 and str[n - 1] == " ":
        n -= 1
    istate = 1
    for i in range(1, n + 1):
        icls = nclass(str[i - 1])
        istate = int(_itrans[istate - 1, icls - 1])
        if istate == 0:
            return False, i, ndig
        if icls == 2:
            ndig += 1
    ok = int(_iacc[istate - 1]) == 1
    if not ok:
        ipos = n + 1
    return ok, ipos, ndig


def numval(str, ierr):
    okl, ipos, _ = numscn(str, False, 0, 0)
    ierr = ipos
    if not okl:
        return 0.0, ierr
    v = 0.0
    scale = 1.0
    isgn = 1
    ifrac = 0
    for ch in str:
        if ch == " ":
            break
        if ch == "-":
            isgn = -1
        if ch == ".":
            ifrac = 1
        idig = ord(ch) - ord("0")
        if idig < 0 or idig > 9:
            continue
        if ifrac == 1:
            scale = scale / 10.0
            v = v + idig * scale
        else:
            v = v * 10.0 + idig
    return isgn * v, ierr
