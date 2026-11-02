"""Reference port of legacy.f (series sums and a report program)."""

from __future__ import annotations


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def harmon(n):
    s = 0.0
    for k in range(1, n + 1):
        s = s + 1.0 / float(k)
    return s


def isumsq(n):
    def sqr(j):
        return j * j

    s = 0
    for k in range(1, n + 1):
        s = s + sqr(k)
    return s


def geosum(a, r, n):
    s = 0.0
    t = a
    for _ in range(1, n + 1):
        s = s + t
        t = t * r
    return s


def isummu(n, m):
    k = _idiv(n, m)
    return _idiv(m * k * (k + 1), 2)


def main():
    print(" N     HARMONIC    SUMSQ    GEOM")
    for n in range(1, 6):
        print(f" {n:3d}   {harmon(n):10.6f}   {isumsq(n):6d}   {geosum(0.5, 0.5, n):8.5f}")
    print(f" MULTIPLES OF 7 UP TO 100: {isummu(100, 7):5d}")


if __name__ == "__main__":
    main()
