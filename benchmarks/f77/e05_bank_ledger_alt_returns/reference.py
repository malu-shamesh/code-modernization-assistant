"""Hand-written Python port of legacy.f (checking-account ledger with alternate returns)."""

from __future__ import annotations


class _Acct:
    """COMMON /ACCT/ BAL, ODLIM, NTX, NREJ"""

    def __init__(self) -> None:
        self.bal = 0.0
        self.odlim = 0.0
        self.ntx = 0
        self.nrej = 0


_acct = _Acct()


def openac(bal0, limit):
    _acct.bal = bal0
    _acct.odlim = limit
    _acct.ntx = 0
    _acct.nrej = 0


def posttx(itype, amt, newbal):
    """Returns (newbal, alt) where alt is 0 (normal), 1 (invalid) or 2 (declined)."""
    a = _acct
    if amt < 0.0 or itype < 1 or itype > 4:
        a.nrej += 1
        return a.bal, 1
    if itype == 1:
        a.bal = a.bal + amt
    elif itype == 2:
        if a.bal - amt < -a.odlim:
            a.nrej += 1
            return a.bal, 2
        a.bal = a.bal - amt
    elif itype == 3:
        a.bal = a.bal - amt
    elif a.bal > 0:
        a.bal = a.bal + a.bal * amt / 100.0
    a.ntx += 1
    return a.bal, 0


def runled(n, kinds, amts, nok, ninv, ndec, ndone, final):
    nok = ninv = ndec = ndone = 0
    while ndone < n and ndec < 2:
        ndone += 1
        _, alt = posttx(int(kinds[ndone - 1]), float(amts[ndone - 1]), 0.0)
        if alt == 0:
            nok += 1
        elif alt == 1:
            ninv += 1
        else:
            ndec += 1
    return nok, ninv, ndec, ndone, _acct.bal


def acstat(curbal, npost, nrejct):
    return _acct.bal, _acct.ntx, _acct.nrej
