"""Reference port of legacy.f (Newton-Raphson and bisection root finding)."""

from __future__ import annotations

import math


def newton(f, df, x, tol, maxit, nit, ierr):
    ierr = 1
    done = False
    for nit in range(1, maxit + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1.0e-14:
            ierr = 2
            done = True
            break
        dx = fx / dfx
        x = x - dx
        if abs(dx) <= tol * max(1.0, abs(x)):
            ierr = 0
            done = True
            break
    if not done:
        nit = maxit
    return x, nit, ierr


def bisect(f, a, b, tol, maxit, root, nit):
    lo = a
    hi = b
    flo = f(lo)
    fhi = f(hi)
    nit = 0
    if flo * fhi > 0.0:
        return root, nit, 1
    mid = lo
    done = False
    for nit in range(1, maxit + 1):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0:
            done = True
            break
        if 0.5 * (hi - lo) < tol:
            done = True
            break
        if math.copysign(1.0, fmid) == math.copysign(1.0, flo):
            lo = mid
            flo = fmid
        else:
            hi = mid
    if not done:
        nit = maxit
    root = mid
    return root, nit, 0
