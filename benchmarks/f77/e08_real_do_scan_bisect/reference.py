"""Hand-written Python port of legacy.f (bracketing scan with a REAL DO variable)."""

from __future__ import annotations


def rootsc(f, a, b, h, tol, maxit, root, nit, xend):
    root = 0.0
    nit = 0
    xend = a
    if h == 0.0:
        return root, nit, xend, 1
    xl = a
    fl = f(xl)
    # DO X = A + H, B, H: start, limit and step are evaluated once; the trip count is
    # truncated, and X keeps start + trips * h after a normal exit.
    start = a + h
    trips = max(int((b - start + h) / h), 0)
    x = start
    target = 0
    for k in range(trips):
        x = start + k * h
        fr = f(x)
        prod = fl * fr
        if prod < 0.0:
            target = 30
            break
        if prod == 0.0:
            target = 20
            break
        xl = x
        fl = fr
    if target == 0:
        return root, nit, start + trips * h, 1
    xend = x
    if target == 20:
        root = xl if fl == 0.0 else x
        return root, nit, xend, 0
    xr = x
    while True:
        if nit >= maxit:
            return 0.5 * (xl + xr), nit, xend, 2
        xm = 0.5 * (xl + xr)
        fm = f(xm)
        nit += 1
        prod = fl * fm
        if prod == 0.0:
            return xm, nit, xend, 0
        if prod < 0.0:
            xr = xm
        else:
            xl = xm
            fl = fm
        if abs(xr - xl) <= tol:
            return 0.5 * (xl + xr), nit, xend, 0


def fixpt(g, x0, tol, maxit, xfix, nit):
    xfix = x0
    nit = 0
    while True:
        if nit >= maxit:
            return xfix, nit, 2
        xn = g(xfix)
        nit += 1
        if abs(xn) > 1.0e3:
            return xn, nit, 1
        if abs(xn - xfix) <= tol:
            return xn, nit, 0
        xfix = xn


def isteps(a, b, h):
    return max(int((b - a + h) / h), 0)
