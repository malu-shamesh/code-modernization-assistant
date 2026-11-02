"""Reference port of legacy.f (small LCG with SAVE state)."""

from __future__ import annotations

_IA = 75
_IC = 74
_IM = 65537

# SAVE IX, NCALL with DATA IX /1/, NCALL /0/
_ix = 1
_ncall = 0


def lcgint(mode, iseed):
    global _ix, _ncall
    if mode == 0:
        _ix = abs(iseed) % _IM  # operands are non-negative, so % equals Fortran MOD
        _ncall = 0
        return _ix
    if mode == 1:
        _ix = (_IA * _ix + _IC) % _IM
        _ncall += 1
        return _ix
    if mode == 2:
        return _ncall
    return -1


def lcguni():
    return float(lcgint(1, 0)) / 65537.0


def lcgrng(n):
    # (draw * n) / 65537 is a truncating integer division; both operands are non-negative here
    return 1 + (lcgint(1, 0) * n) // 65537


def lcgfil(n, x):
    for i in range(1, n + 1):
        x[i - 1] = lcguni()
