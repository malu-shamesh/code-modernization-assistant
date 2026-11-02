"""What a statement reads, writes and calls: the shared basis of the API and detector passes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from .exprs import Expr, ImpliedDo, Kw, Name, Range, Ref, Star, children
from .source import Statement
from .stmts import (
    Assign,
    AssignLabel,
    Call,
    Common,
    Condition,
    Decl,
    Dimension,
    Do,
    Goto,
    Io,
    Parameter,
    Return,
)
from .symbols import Scope, StatementFunction, expr_type, is_intrinsic_call

IO_STATUS_OUTPUTS = frozenset({"iostat", "size"})
INQUIRE_OUTPUTS = frozenset(
    {
        "iostat",
        "exist",
        "opened",
        "number",
        "named",
        "name",
        "access",
        "sequential",
        "direct",
        "form",
        "formatted",
        "unformatted",
        "recl",
        "nextrec",
        "blank",
        "size",
    }
)


@dataclass(slots=True)
class CallSite:
    """A procedure reference: ``CALL name(args)`` or ``name(args)`` inside an expression."""

    callee: str
    args: tuple[Expr, ...]
    statement: Statement


@dataclass(slots=True)
class Effects:
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    sites: list[CallSite] = field(default_factory=list)


def is_user_call(ref: Ref, scope: Scope) -> bool:
    """True for ``name(args)`` that is a function reference (not array/intrinsic/stmt function)."""
    if scope.is_array(ref.name) or ref.name in scope.stmt_functions:
        return False
    if ref.sub is not None or any(isinstance(a, Range) for a in ref.args):
        return False
    return not is_intrinsic_call(ref, scope)


def statement_exprs(info: object) -> Iterator[Expr]:
    """Top-level expressions of a statement record (conditions, values, subscripts, bounds)."""
    if isinstance(info, Assign):
        yield info.target
        yield info.value
    elif isinstance(info, Call):
        yield from info.args
    elif isinstance(info, Condition):
        yield info.cond
    elif isinstance(info, Do):
        for e in (info.lo, info.hi, info.step, info.cond):
            if e is not None:
                yield e
    elif isinstance(info, Goto):
        if info.expr is not None:
            yield info.expr
    elif isinstance(info, Return):
        if info.expr is not None:
            yield info.expr
    elif isinstance(info, Io):
        if info.unit is not None:
            yield info.unit
        if info.fmt is not None:
            yield info.fmt
        yield from info.controls.values()
        yield from info.items
    elif isinstance(info, Parameter):
        for _, value in info.items:
            yield value
    elif isinstance(info, StatementFunction):
        yield info.body
    elif isinstance(info, Decl | Dimension | Common):
        for entity in _entities(info):
            for dim in entity.dims:
                for bound in (dim.lower, dim.upper):
                    if bound is not None:
                        yield bound


def _entities(info: Decl | Dimension | Common) -> list:
    if isinstance(info, Common):
        return [e for _, entities in info.blocks for e in entities]
    return info.entities


class _Scanner:
    def __init__(self, scope: Scope, statement: Statement) -> None:
        self.scope = scope
        self.statement = statement
        self.effects = Effects()

    def read(self, expr: Expr) -> None:
        """Record every name read by ``expr``; user calls become call sites."""
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Name):
                self.effects.reads.append(node.name)
            elif isinstance(node, Ref):
                if is_user_call(node, self.scope):
                    self.site(node.name, node.args)
                else:
                    if self.scope.is_array(node.name):
                        self.effects.reads.append(node.name)
                    stack.extend(reversed(node.args + (node.sub or ())))
            elif isinstance(node, Kw):
                stack.append(node.value)
            elif isinstance(node, Star):
                continue
            else:
                stack.extend(reversed(children(node)))

    def site(self, callee: str, args: tuple[Expr, ...]) -> None:
        self.effects.sites.append(CallSite(callee, args, self.statement))
        for arg in args:
            if isinstance(arg, Name):
                continue  # passed by reference: the callee decides whether it is read/written
            if isinstance(arg, Ref) and self.scope.is_array(arg.name):
                for sub in arg.args:
                    self.read(sub)
                continue
            self.read(arg)

    def write_target(self, expr: Expr) -> None:
        if isinstance(expr, Name):
            self.effects.writes.append(expr.name)
        elif isinstance(expr, Ref):
            self.effects.writes.append(expr.name)
            if expr.sub is not None or any(isinstance(a, Range) for a in expr.args):
                # S(I:J) = ...: the rest of the string keeps its old value, so S is also read
                self.effects.reads.append(expr.name)
            for sub in expr.args + (expr.sub or ()):
                self.read(sub)
        elif isinstance(expr, ImpliedDo):
            for item in expr.items:
                self.write_target(item)
            self.read(expr.lo)
            self.read(expr.hi)
            if expr.step is not None:
                self.read(expr.step)
        else:
            self.read(expr)


def statement_effects(part: Statement, scope: Scope) -> Effects:
    """Direct reads, writes and call sites of one (effective) statement (cached on it)."""
    if isinstance(part.effects, Effects):
        return part.effects
    scanner = _Scanner(scope, part)
    info = part.info
    if isinstance(info, Assign):
        scanner.read(info.value)
        scanner.write_target(info.target)
    elif isinstance(info, Call):
        scanner.site(info.name, info.args)
    elif isinstance(info, Do):
        for e in (info.lo, info.hi, info.step, info.cond):
            if e is not None:
                scanner.read(e)
        if info.var:
            scanner.effects.writes.append(info.var)
    elif isinstance(info, Io):
        _io_effects(scanner, info, scope)
    elif isinstance(info, AssignLabel):
        scanner.effects.writes.append(info.var)
    elif isinstance(info, Goto) and info.var:
        scanner.effects.reads.append(info.var)
    else:
        for expr in statement_exprs(info):
            scanner.read(expr)
    part.effects = scanner.effects
    return scanner.effects


def _io_effects(scanner: _Scanner, info: Io, scope: Scope) -> None:
    op = info.op
    # Specifier keywords that receive a value; for INQUIRE almost all of them do.
    outputs = INQUIRE_OUTPUTS if op == "inquire" else IO_STATUS_OUTPUTS
    if info.unit is not None:
        internal = isinstance(info.unit, Name | Ref) and expr_type(info.unit, scope) == "character"
        if internal and op == "write":
            scanner.write_target(info.unit)
        else:
            scanner.read(info.unit)
    if info.fmt is not None:
        scanner.read(info.fmt)
    for key, value in info.controls.items():
        if key in {"unit", "fmt"}:
            continue
        if key in outputs and isinstance(value, Name | Ref):
            scanner.write_target(value)
        else:
            scanner.read(value)
    for item in info.items:
        if op == "read":
            scanner.write_target(item)
        else:
            scanner.read(item)
