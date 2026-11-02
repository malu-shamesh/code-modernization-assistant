"""Reference port of legacy.f (token scanner over a line held in COMMON)."""

from __future__ import annotations

_BUFLEN = 40
_TOKLEN = 12


class _ScnBuf:
    """COMMON /SCNBUF/ BUF (CHARACTER*40)"""

    def __init__(self) -> None:
        self.buf = " " * _BUFLEN


class _ScnPtr:
    """COMMON /SCNPTR/ IPOS, ILEN"""

    def __init__(self) -> None:
        self.ipos = 0
        self.ilen = 0


_scnbuf = _ScnBuf()
_scnptr = _ScnPtr()


def _char_assign(value, n):
    """CHARACTER*n assignment: truncate or blank pad."""
    return value[:n].ljust(n)


def _is_digit(ic):
    return 48 <= ic <= 57


def _is_letter(ic):
    return 65 <= ic <= 90 or 97 <= ic <= 122


def scinit(line):
    _scnbuf.buf = _char_assign(line, _BUFLEN)
    ilen = _BUFLEN
    while ilen != 0 and _scnbuf.buf[ilen - 1] == " ":
        ilen = ilen - 1
    _scnptr.ilen = ilen
    _scnptr.ipos = 1


def scpos():
    return _scnptr.ipos


def scnext(tok, itype):
    tok = " " * _TOKLEN
    itype = 0
    buf = _scnbuf.buf
    # skip blanks (the arithmetic IF only asks whether any text is left)
    while _scnptr.ipos - _scnptr.ilen <= 0 and buf[_scnptr.ipos - 1] == " ":
        _scnptr.ipos = _scnptr.ipos + 1
    if _scnptr.ipos - _scnptr.ilen > 0:
        return tok, itype, 1
    ipos = _scnptr.ipos
    ic = ord(buf[ipos - 1])
    icls = 3
    if _is_digit(ic):
        icls = 1
    if _is_letter(ic):
        icls = 2
    j = ipos + 1
    if icls == 1:
        while j <= _scnptr.ilen and _is_digit(ord(buf[j - 1])):
            j = j + 1
    elif icls == 2:
        while j <= _scnptr.ilen and (_is_digit(ord(buf[j - 1])) or _is_letter(ord(buf[j - 1]))):
            j = j + 1
    tok = _char_assign(buf[ipos - 1 : j - 1], _TOKLEN)
    itype = icls
    _scnptr.ipos = j
    return tok, itype, 0


def scint(tok):
    n = 0
    for i in range(1, len(tok) + 1):
        if tok[i - 1] == " ":
            break
        ic = ord(tok[i - 1]) - 48
        if ic < 0 or ic > 9:
            return -1
        n = 10 * n + ic
    return n


def sctall(line, nnum, nid, nop):
    scinit(line)
    nnum = 0
    nid = 0
    nop = 0
    while True:
        _tok, itype, alt = scnext(" " * _TOKLEN, 0)
        if alt == 1:
            break
        if itype == 1:
            nnum = nnum + 1
        if itype == 2:
            nid = nid + 1
        if itype == 3:
            nop = nop + 1
    return nnum, nid, nop


def scsum(line, isum):
    scinit(line)
    isum = 0
    while True:
        tok, itype, alt = scnext(" " * _TOKLEN, 0)
        if alt == 1:
            break
        if itype == 1:
            isum = isum + scint(tok)
    return isum
