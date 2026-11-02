"""Hand-written Python port of legacy.f (accumulator calculator)."""

from __future__ import annotations


def calc(nops, iop, arg, acc, ierr):
    acc = 0.0
    ierr = 0
    for i in range(1, nops + 1):
        op = int(iop[i - 1])
        if op < 1 or op > 6:
            ierr = 1
            break
        if op == 1:
            acc = acc + arg[i - 1]
        elif op == 2:
            acc = acc - arg[i - 1]
        elif op == 3:
            acc = acc * arg[i - 1]
        elif op == 4:
            if arg[i - 1] == 0:
                ierr = 2
                break
            acc = acc / arg[i - 1]
        elif op == 5:
            acc = -acc
        else:
            acc = abs(acc)
    return float(acc), ierr


def ksign(x, eps):
    if abs(x) - eps <= 0:
        return 0
    if x <= 0:
        return -1
    return 1


def tally(n, x, eps, nneg, nzero, npos):
    nneg = nzero = npos = 0
    for i in range(1, n + 1):
        k = ksign(x[i - 1], eps) + 2
        if k == 1:
            nneg += 1
        elif k == 2:
            nzero += 1
        else:
            npos += 1
    return nneg, nzero, npos
