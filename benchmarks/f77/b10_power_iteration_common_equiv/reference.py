"""Python port of the power-iteration package with COMMON blocks and EQUIVALENCE.

COMMON /PWRCTL/ becomes the module-level ``_ctl`` object (initialised the way BLOCK DATA PWRBD
does).  COMMON /PWRWRK/ is the pair of 100 element arrays ``_V`` and ``_W``; R is EQUIVALENCEd
onto W, so ``_R`` is the very same array object as ``_W``.
"""

from __future__ import annotations

import math

import numpy as np


class _Control:
    """COMMON /PWRCTL/"""

    def __init__(self) -> None:
        self.maxits = 50
        self.tolp = 1.0e-10
        self.ncalls = 0
        self.nitot = 0
        self.nlast = 0


_ctl = _Control()
_V = np.zeros(100)
_W = np.zeros(100)
_R = _W


def pwrcfg(maxit: int, tol: float) -> None:
    if maxit > 0:
        _ctl.maxits = maxit
    if tol > 0.0:
        _ctl.tolp = tol


def pwritr(
    a: np.ndarray, lda: int, n: int, x0: np.ndarray, eigval: float, iters: int, ierr: int
) -> tuple[float, int, int]:
    _ctl.ncalls = _ctl.ncalls + 1
    eigval = 0.0
    iters = 0
    ierr = 0
    if n < 1 or n > 100 or lda < n:
        ierr = 2
        return eigval, iters, ierr
    s = 0.0
    for i in range(1, n + 1):
        s = s + x0[i - 1] * x0[i - 1]
    if s == 0.0:
        ierr = 3
        return eigval, iters, ierr
    for i in range(1, n + 1):
        _V[i - 1] = x0[i - 1] / math.sqrt(s)
    _ctl.nlast = n
    eigold = 0.0
    converged = False
    for k in range(1, _ctl.maxits + 1):
        for i in range(1, n + 1):
            _W[i - 1] = 0.0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                _W[i - 1] = _W[i - 1] + a[i - 1, j - 1] * _V[j - 1]
        eigval = 0.0
        s = 0.0
        for i in range(1, n + 1):
            eigval = eigval + _V[i - 1] * _W[i - 1]
            s = s + _W[i - 1] * _W[i - 1]
        wn = math.sqrt(s)
        if wn == 0.0:
            converged = True
            break
        if abs(eigval - eigold) <= _ctl.tolp * max(1.0, abs(eigval)):
            converged = True
            break
        if k < _ctl.maxits:
            for i in range(1, n + 1):
                _V[i - 1] = _W[i - 1] / wn
            eigold = eigval
    if converged:
        iters = k
    else:
        ierr = 1
        iters = _ctl.maxits
    for i in range(1, n + 1):
        _R[i - 1] = _W[i - 1] - eigval * _V[i - 1]
    _ctl.nitot = _ctl.nitot + iters
    return float(eigval), iters, ierr


def pwrget(which: int, i: int) -> float:
    result = 0.0
    if which == 3:
        result = float(_ctl.ncalls)
    if which == 4:
        result = float(_ctl.nitot)
    if i < 1 or i > _ctl.nlast:
        return result
    if which == 1:
        result = float(_V[i - 1])
    if which == 2:
        result = float(_R[i - 1])
    return result


def pwrrq(a: np.ndarray, lda: int, n: int) -> float:
    result = 0.0
    if n != _ctl.nlast or lda < n:
        return result
    for i in range(1, n + 1):
        _W[i - 1] = 0.0
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            _W[i - 1] = _W[i - 1] + a[i - 1, j - 1] * _V[j - 1]
    for i in range(1, n + 1):
        result = result + _V[i - 1] * _W[i - 1]
    return float(result)
