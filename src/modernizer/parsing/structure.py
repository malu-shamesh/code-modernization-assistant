"""Block structure of a program unit: DO/IF nesting, shared DO terminals, cyclomatic counts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .source import EXECUTABLE_KINDS, Statement
from .stmts import Do, Goto

_AND_OR = re.compile(r"\.(?:AND|OR)\.", re.IGNORECASE)


@dataclass(slots=True)
class Structure:
    max_depth: int = 0
    # (outermost DO statement, terminal statement, number of loops sharing the terminal)
    shared_terminals: list[tuple[Statement, Statement, int]] = field(default_factory=list)
    # unlabelled DO / DO WHILE loops with the END DO that closes them (None when unmatched)
    block_loops: list[tuple[Statement, Statement | None]] = field(default_factory=list)
    unbalanced: bool = False  # an IF/DO block that is never closed, or a closer with no opener


def scan_structure(statements: list[Statement]) -> Structure:
    result = Structure()
    stack: list[tuple[str, str | None, Statement]] = []
    for stmt in statements:
        kind = stmt.kind
        depth = len(stack)
        if kind in {"do", "do_while"}:
            depth += 1
        elif kind == "if_then":
            depth += 1
        result.max_depth = max(result.max_depth, depth)
        if stmt.label:
            closed = []
            while stack and stack[-1][0] == "do" and stack[-1][1] == stmt.label:
                closed.append(stack.pop())
            if len(closed) > 1:
                result.shared_terminals.append((closed[-1][2], stmt, len(closed)))
        if kind in {"do", "do_while"}:
            info = stmt.info
            label = info.label if isinstance(info, Do) else None
            stack.append(("do", label, stmt))
        elif kind == "if_then":
            stack.append(("if", None, stmt))
        elif kind in {"else", "else_if"}:
            result.unbalanced |= not any(entry[0] == "if" for entry in stack)
        elif kind == "end_if":
            result.unbalanced |= _pop_nearest(stack, "if") is None
        elif kind == "end_do":
            entry = _pop_nearest(stack, "do", unlabeled=True)
            if entry is None:
                result.unbalanced = True
            else:
                result.block_loops.append((entry[2], stmt))
    for entry in stack:
        result.unbalanced = True
        if entry[0] == "do" and entry[1] is None:
            result.block_loops.append((entry[2], None))
    return result


def _pop_nearest(
    stack: list[tuple[str, str | None, Statement]], what: str, unlabeled: bool = False
) -> tuple[str, str | None, Statement] | None:
    for i in range(len(stack) - 1, -1, -1):
        entry = stack[i]
        if entry[0] == what and (not unlabeled or entry[1] is None):
            del stack[i:]
            return entry
    return None


def cyclomatic_branches(statements: list[Statement]) -> int:
    """Number of branch points: IF/ELSE IF/DO/logical IF/.AND./.OR./arithmetic IF/computed GOTO."""
    total = 0
    for stmt in statements:
        if stmt.kind not in EXECUTABLE_KINDS:
            continue
        for part in (stmt, stmt.inner) if stmt.inner is not None else (stmt,):
            kind = part.kind
            if kind in {"if_then", "else_if", "do", "do_while", "if_logical"}:
                total += 1
            elif kind == "if_arith":
                total += 2
            elif kind == "goto_computed" and isinstance(part.info, Goto):
                total += max(0, len(part.info.labels) - 1)
        total += len(_AND_OR.findall(stmt.masked))
    return total
