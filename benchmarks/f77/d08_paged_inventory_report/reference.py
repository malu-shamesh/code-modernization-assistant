"""Python port of PGRPT (paginated parts inventory report)."""

from __future__ import annotations

from types import SimpleNamespace

# COMMON /PGCTL/ LPP, NLINE, NPAGE
_pgctl = SimpleNamespace(lpp=0, nline=0, npage=0)


def _i_edit(n: int, w: int) -> str:
    s = str(n)
    return "*" * w if len(s) > w else s.rjust(w)


def _f_edit(x: float, w: int, d: int) -> str:
    """Fw.d edit descriptor: right justified, asterisks on overflow."""
    s = f"{x:.{d}f}"
    return "*" * w if len(s) > w else s.rjust(w)


def _e_edit(x: float, w: int, d: int) -> str:
    """Ew.d edit descriptor for |exponent| < 100: 0.dddE+ee."""
    if x == 0:
        mant, exp = "0" * d, 0
    else:
        digits, _, e = f"{abs(x):.{d - 1}e}".partition("e")
        mant = digits.replace(".", "")
        exp = int(e) + 1
    s = ("-" if x < 0 else "") + "0." + mant + ("E-" if exp < 0 else "E+") + f"{abs(exp):02d}"
    return s.rjust(w)


def pgdata():
    """BLOCK DATA PGDATA."""
    _pgctl.lpp = 55
    _pgctl.nline = 55
    _pgctl.npage = 0


pgdata()


def lastnb(s):
    for i in range(len(s), 0, -1):
        if s[i - 1] != " ":
            return i
    return 0


def pginit(lines):
    _pgctl.lpp = lines
    _pgctl.nline = lines
    _pgctl.npage = 0


def prtlin(text):
    if _pgctl.nline >= _pgctl.lpp:
        _pgctl.npage += 1
        print(" PARTS INVENTORY" + " " * 20 + "PAGE " + _i_edit(_pgctl.npage, 3))
        print(
            " "
            + "ITEM        ".rjust(12)
            + "  "
            + "QTY".rjust(5)
            + "  "
            + "PRICE".rjust(9)
            + "  "
            + "VALUE".rjust(11)
        )
        print(" " + "-" * 43)
        _pgctl.nline = 3
    n = max(lastnb(text), 1)
    print(" " + text[:n])
    _pgctl.nline += 1


def prtrow(name, iqty, price):
    name = name[:12].ljust(12)
    ext = iqty * price
    line = name + "  " + _i_edit(iqty, 5) + "  " + _f_edit(price, 9, 2) + "  " + _f_edit(ext, 11, 2)
    prtlin(line.ljust(60))


def prttot(iqtot, totval, nitem):
    lab = "TOTAL".ljust(12)
    line = lab + "  " + _i_edit(iqtot, 5) + " " * 13 + _f_edit(totval, 11, 2)
    prtlin(line.ljust(60))
    if nitem > 0:
        avg = totval / nitem
        line = "MEAN OVER" + _i_edit(nitem, 3) + " ITEMS" + " " * 3 + _e_edit(avg, 12, 4)
        prtlin(line.ljust(60))


def prtmat(a, lda, nr, nc):
    for i in range(1, nr + 1):
        values = [a[i - 1, j - 1] for j in range(1, nc + 1)]
        label = " ROW" + _i_edit(i, 3) + ":"
        # format reversion: the first record carries the label, later ones only the group
        first, rest = values[:6], values[6:]
        print(label + "".join(_f_edit(v, 9, 2) for v in first))
        while rest:
            print("".join(_f_edit(v, 9, 2) for v in rest[:6]))
            rest = rest[6:]


def main():
    name = ["BOLT M8", "WASHER 8", "NUT M8", "SPRING", "GASKET", "PLATE 200"]
    name = [s.ljust(12) for s in name]
    iqty = [120, 400, 150, 36, 12, 5]
    price = [0.35, 0.05, 0.12, 2.50, 18.75, 129.99]
    pginit(8)
    iqtot = 0
    totval = 0.0
    i = 1
    for _ in range(6):
        prtrow(name[i - 1], iqty[i - 1], price[i - 1])
        iqtot += iqty[i - 1]
        totval += iqty[i - 1] * price[i - 1]
        i += 1
    # the DO variable is 7 after the loop, so I - 1 rows were listed
    prttot(iqtot, totval, i - 1)


if __name__ == "__main__":
    main()
