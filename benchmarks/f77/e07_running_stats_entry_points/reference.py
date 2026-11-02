"""Hand-written Python port of legacy.f (RSTAT: one program unit, four entry points)."""

from __future__ import annotations


class _State:
    """Saved locals of the RSTRST program unit (DATA-initialised, shared by all entries)."""

    def __init__(self) -> None:
        self.n = 0
        self.wmean = 0.0
        self.wm2 = 0.0
        self.xlo = 0.0
        self.xhi = 0.0


_st = _State()


def rstrst():
    _st.n = 0
    _st.wmean = 0.0
    _st.wm2 = 0.0
    _st.xlo = 0.0
    _st.xhi = 0.0


def rstadd(x):
    _st.n += 1
    delta = x - _st.wmean
    _st.wmean = _st.wmean + delta / _st.n
    _st.wm2 = _st.wm2 + delta * (x - _st.wmean)
    if _st.n == 1:
        _st.xlo = x
        _st.xhi = x
        return
    if x < _st.xlo:
        _st.xlo = x
    if x > _st.xhi:
        _st.xhi = x


def rstget(cnt, xmean, xvar):
    cnt = _st.n
    xmean = _st.wmean
    xvar = 0.0
    if _st.n == 0:
        return cnt, xmean, xvar, 1
    if _st.n > 1:
        xvar = _st.wm2 / (_st.n - 1)
    return cnt, xmean, xvar, 0


def rstrng(xmin, xmax):
    xmin = _st.xlo
    xmax = _st.xhi
    if _st.n == 0:
        return xmin, xmax, 1
    return xmin, xmax, 0


def rstall(n, x):
    rstrst()
    for i in range(1, n + 1):
        rstadd(float(x[i - 1]))
