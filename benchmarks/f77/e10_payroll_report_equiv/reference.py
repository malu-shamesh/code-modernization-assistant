"""Hand-written Python port of legacy.f (payroll report: PROGRAM + two subroutines)."""

from __future__ import annotations

import math

import numpy as np

# EQUIVALENCE (TAXTAB(1,1), TAXFLT(1)): the flat DATA list fills the 2 by 3 table column-major.
_TAXFLT = np.array([0.125, 0.0, 0.25, 100.0, 0.375, 300.0])
_TAXTAB = _TAXFLT.reshape((2, 3), order="F")


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a: int, p: int) -> int:
    return a - _idiv(a, p) * p


def _nint(x: float) -> int:
    n = int(math.floor(abs(x) + 0.5))
    return -n if x < 0 else n


def payclc(hours, rate, igrade, gross, tax):
    gross = 0.0
    tax = 0.0
    if igrade < 1 or igrade > 3:
        return gross, tax, 1
    otmult = {1: 1.0, 2: 1.5, 3: 2.0}[igrade]
    reg = min(hours, 40.0) * rate
    ovt = 0.0
    if hours - 40.0 > 0:
        ovt = (hours - 40.0) * rate * otmult
    gross = reg + ovt
    tax = float(_TAXTAB[0, igrade - 1]) * (gross - float(_TAXTAB[1, igrade - 1]))
    if tax < 0:
        tax = 0.0
    return gross, tax, 0


def splitc(amt, isg, idol, icent):
    ict = _nint(amt * 100.0)
    isg = -1 if ict < 0 else 1
    idol = _idiv(ict, 100)
    icent = abs(_mod(ict, 100))
    return isg, idol, icent


def main():
    nemp = 5
    ids = [101, 102, 103, 104, 105]
    igrade = [1, 2, 3, 7, 2]
    hours = [40.0, 44.0, 42.0, 40.0, 8.0]
    rate = [12.5, 20.0, 25.0, 10.0, 9.75]
    npaid = 0
    nrej = 0
    tgross = ttax = tnet = best = 0.0
    print(" EMP" + " " * 5 + "GROSS" + " " * 7 + "TAX" + " " * 7 + "NET")
    for i in range(nemp):
        gross, tax, alt = payclc(hours[i], rate[i], igrade[i], 0.0, 0.0)
        if alt == 1:
            nrej += 1
            print(f" {ids[i]:3d}  BAD GRADE {igrade[i]:2d}")
            continue
        net = gross - tax
        npaid += 1
        tgross += gross
        ttax += tax
        tnet += net
        if net > best:
            best = net
        print(f" {ids[i]:3d}{gross:10.2f}{tax:10.2f}{net:10.2f}")
    print(f" TOT{tgross:10.2f}{ttax:10.2f}{tnet:10.2f}")
    print(f" PAID{npaid:3d}  REJECTED{nrej:3d}")
    if npaid == 0:
        # STOP; unreachable with the built-in data, so no f77rt dependency is needed
        return
    avg = tnet / float(npaid)
    _, idol, icent = splitc(avg, 0, 0, 0)
    print(f" AVG NET {idol:6d}.{icent:02d}")
    _, idol, icent = splitc(best, 0, 0, 0)
    print(f" TOP NET {idol:6d}.{icent:02d}")


if __name__ == "__main__":
    main()
