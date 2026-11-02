"""Python port of NUMSCN (free format number scanner).

SCANNM has two alternate returns, so its tuple ends with the number of the return taken
(0 for a normal RETURN): (value, nused, altret).
"""

from __future__ import annotations


def scannm(text, value, nused):
    value = 0.0
    nused = 0
    n = len(text)
    ifirst = 0
    for i in range(1, n + 1):
        if text[i - 1] != " ":
            ifirst = i
            break
    else:
        return value, nused, 1
    i = ifirst
    mant = 0.0
    nfrac = 0
    ndig = 0
    nedig = 0
    iexp = 0
    isign = 1
    jsign = 1
    lastcl = 0
    seenpt = False
    inexp = False
    while i <= n:
        c = text[i - 1]
        iclass = 5
        if "0" <= c <= "9":
            iclass = 1
        if c == "+" or c == "-":
            iclass = 2
        if c == ".":
            iclass = 3
        if c in "EDed":
            iclass = 4
        if iclass == 5:
            break
        if iclass == 1:
            k = ord(c) - ord("0")
            if inexp:
                iexp = min(10 * iexp + k, 9999)
                nedig += 1
            else:
                mant = 10.0 * mant + k
                ndig += 1
                if seenpt:
                    nfrac += 1
        elif iclass == 2:
            if inexp:
                if lastcl != 4:
                    break
                if c == "-":
                    jsign = -1
            else:
                if i != ifirst:
                    break
                if c == "-":
                    isign = -1
        elif iclass == 3:
            if inexp or seenpt:
                break
            seenpt = True
        else:
            if inexp or ndig == 0:
                break
            inexp = True
        if ndig > 0 and (not inexp or nedig > 0):
            nused = i
        lastcl = iclass
        i += 1
    if ndig == 0:
        return value, nused, 1
    ie = jsign * iexp - nfrac
    if abs(ie) > 300:
        return value, nused, 2
    if ie >= 0:
        value = mant * 10.0**ie
    else:
        value = mant / 10.0 ** (-ie)
    value = isign * value
    return value, nused, 0


def scanls(text, vals, maxv, nv, ierr, ibad):
    nv = 0
    ierr = 0
    ibad = 0
    ipos = 1
    n = len(text)
    while True:
        i = ipos
        while i <= n and text[i - 1] in " ,":
            i += 1
        if i > n:
            return nv, ierr, ibad
        if nv >= maxv:
            return nv, 3, i
        v, nu, alt = scannm(text[i - 1 : n], 0.0, 0)
        if alt == 1:
            return nv, 1, i
        if alt == 2:
            return nv, 2, i
        nv += 1
        vals[nv - 1] = v
        ipos = i + nu
