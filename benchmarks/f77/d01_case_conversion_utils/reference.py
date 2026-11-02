"""Python port of CASEUT (character case conversion utilities)."""

from __future__ import annotations

_WIDTH = 40


def _res(chars: list[str]) -> str:
    """Assign to CHARACTER*40: truncate, then pad with blanks."""
    return "".join(chars)[:_WIDTH].ljust(_WIDTH)


def touppr(text, res):
    out = []
    for ch in text[:_WIDTH]:
        ic = ord(ch)
        if 97 <= ic <= 122:
            ic -= 32
        out.append(chr(ic))
    return _res(out)


def tolowr(text, res):
    out = []
    for ch in text[:_WIDTH]:
        ic = ord(ch)
        if 65 <= ic <= 90:
            ic += 32
        out.append(chr(ic))
    return _res(out)


def swapcs(text, res):
    out = []
    for ch in text[:_WIDTH]:
        ic = ord(ch)
        if 97 <= ic <= 122:
            ic -= 32
        elif 65 <= ic <= 90:
            ic += 32
        out.append(chr(ic))
    return _res(out)


def nupper(text):
    ncount = 0
    for ch in text:
        if 65 <= ord(ch) <= 90:
            ncount += 1
    return ncount


def titlec(text, res):
    newwd = True
    out = []
    for ch in text[:_WIDTH]:
        ic = ord(ch)
        if ch == " ":
            newwd = True
        else:
            if newwd and 97 <= ic <= 122:
                ic -= 32
            if not newwd and 65 <= ic <= 90:
                ic += 32
            newwd = False
        out.append(chr(ic))
    return _res(out)
