"""Python port of TXTFMT (blank-padded comparison, justification, sorting, squeezing)."""

from __future__ import annotations


def order(a, b):
    # Fortran compares as if the shorter operand were padded with blanks
    n = max(len(a), len(b))
    a = a.ljust(n)
    b = b.ljust(n)
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def justfy(text, mode, width, res):
    res = [" "] * 40
    nw = max(1, min(width, 40))
    ifirst = 0
    ilast = 0
    for i in range(1, len(text) + 1):
        if text[i - 1] != " ":
            if ifirst == 0:
                ifirst = i
            ilast = i
    if ifirst == 0:
        return "".join(res)
    lc = min(ilast - ifirst + 1, nw)
    if mode == 2:
        is_ = nw - lc + 1
    elif mode == 3:
        is_ = (nw - lc) // 2 + 1  # operands are non-negative, so // truncates like Fortran
    else:
        is_ = 1
    res[is_ - 1 : is_ - 1 + lc] = text[ifirst - 1 : ifirst - 1 + lc]
    return "".join(res)


def sortnm(names, n):
    # CHARACTER*12 elements: compared blank padded, assigned padded or cut to 12
    for i in range(1, n):
        for j in range(1, n - i + 1):
            left = names[j - 1].ljust(12)[:12]
            right = names[j].ljust(12)[:12]
            if order(left, right) > 0:
                names[j - 1] = right
                names[j] = left


def squeez(text, res):
    buf = [" "] * 30  # BUF and its EQUIVALENCEd character array CH
    k = 0
    prevbl = True
    for ch in text:
        if ch == " ":
            if not prevbl and k < 30:
                k += 1
                buf[k - 1] = " "
            prevbl = True
        elif k < 30:
            k += 1
            buf[k - 1] = ch
            prevbl = False
    return "".join(buf)
