"""Reference port of legacy.f (integer RPN calculator on a COMMON-block stack)."""

from __future__ import annotations

_MAXD = 8


class _StkCom:
    """COMMON /STKCOM/ ISTK(MAXD), ITOP"""

    def __init__(self) -> None:
        self.istk = [0] * _MAXD
        self.itop = 0


_stkcom = _StkCom()


def _idiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _fmod(a, p):
    return a - p * _idiv(a, p)


def stkclr():
    _stkcom.itop = 0


def stkpsh(ival, ierr):
    ierr = 0
    if _stkcom.itop >= _MAXD:
        return 1
    _stkcom.itop += 1
    _stkcom.istk[_stkcom.itop - 1] = ival
    return ierr


def stkpop(ival, ierr):
    ierr = 0
    ival = 0
    if _stkcom.itop <= 0:
        return ival, 2
    ival = _stkcom.istk[_stkcom.itop - 1]
    _stkcom.itop -= 1
    return ival, ierr


def stkdep():
    return _stkcom.itop


def stkrpn(n, itok, ires, ierr):
    stkclr()
    ires = 0
    ierr = 0
    for i in range(1, n + 1):
        tok = int(itok[i - 1])
        if tok >= 0:
            ierr = stkpsh(tok, ierr)
            if ierr != 0:
                return ires, ierr
            continue
        k = -tok
        ib, ierr = stkpop(0, ierr)
        if ierr != 0:
            return ires, ierr
        ia, ierr = stkpop(0, ierr)
        if ierr != 0:
            return ires, ierr
        if k == 1:
            ia = ia + ib
        elif k == 2:
            ia = ia - ib
        elif k == 3:
            ia = ia * ib
        elif k == 4:
            if ib == 0:
                return ires, 3
            ia = _idiv(ia, ib)
        elif k == 5:
            if ib == 0:
                return ires, 3
            ia = _fmod(ia, ib)
        else:
            return ires, 5
        ierr = stkpsh(ia, ierr)
        if ierr != 0:
            return ires, ierr
    if stkdep() != 1:
        ierr = 4
    else:
        ires, ierr = stkpop(ires, ierr)
    return ires, ierr
