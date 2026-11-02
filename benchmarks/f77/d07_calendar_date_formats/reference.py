"""Python port of CALDAT (calendar arithmetic and date formatting)."""

from __future__ import annotations

_MDAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_T = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
_DAYNM = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
_MONNM = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _mod(a: int, p: int) -> int:
    """Fortran MOD: the result takes the sign of the dividend."""
    r = abs(a) % abs(p)
    return -r if a < 0 else r


def _idiv(a: int, b: int) -> int:
    """Fortran integer division: truncate toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _i_edit(n: int, w: int, m: int = 0) -> str:
    """Iw / Iw.m edit descriptor (asterisks on overflow)."""
    s = str(abs(n)).rjust(m, "0")
    if n < 0:
        s = "-" + s
    return "*" * w if len(s) > w else s.rjust(w)


def lastnb(s):
    for i in range(len(s), 0, -1):
        if s[i - 1] != " ":
            return i
    return 0


def isleap(iy):
    return (_mod(iy, 4) == 0 and _mod(iy, 100) != 0) or _mod(iy, 400) == 0


def ndays(iy, im):
    if im < 1 or im > 12:
        return 0
    if im == 2 and isleap(iy):
        return 29
    return _MDAYS[im - 1]


def valdat(iy, im, id):
    if iy < 1 or iy > 9999:
        return 1
    if im < 1 or im > 12:
        return 2
    if id < 1 or id > ndays(iy, im):
        return 3
    return 0


def idow(iy, im, id):
    y = iy
    if im < 3:
        y -= 1
    return _mod(y + _idiv(y, 4) - _idiv(y, 100) + _idiv(y, 400) + _T[im - 1] + id, 7)


def doy(iy, im, id):
    ntot = id
    for m in range(1, im):
        ntot += ndays(iy, m)
    return ntot


def nextdy(iy, im, id):
    id += 1
    if id > ndays(iy, im):
        id = 1
        im += 1
        if im > 12:
            im = 1
            iy += 1
    return iy, im, id


def fmtiso(iy, im, id, str, ierr):
    ierr = valdat(iy, im, id)
    if ierr != 0:
        return " " * 10, ierr
    return _i_edit(iy, 4, 4) + "-" + _i_edit(im, 2, 2) + "-" + _i_edit(id, 2, 2), ierr


def fmtlng(iy, im, id, str, ierr):
    ierr = valdat(iy, im, id)
    if ierr != 0:
        return " " * 40, ierr
    k = idow(iy, im, id)
    day = _i_edit(id, 1) if id < 10 else _i_edit(id, 2)
    text = _DAYNM[k] + ", " + _MONNM[im - 1] + " " + day + ", " + _i_edit(iy, 4)
    return text.ljust(40), ierr
