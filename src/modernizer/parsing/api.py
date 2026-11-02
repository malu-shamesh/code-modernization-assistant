"""Public API of each program unit: dummy arguments, intents and written scalars.

A scalar dummy argument is *written* when it is an assignment target, a DO index, a READ target
(or I/O status variable), or an actual argument of a call whose callee is defined in the same file
and itself writes the matching dummy. The last rule is computed interprocedurally to a fixpoint.
Unknown external callees are assumed not to write their arguments.

Intent of a written scalar is ``out`` when its first access (in text order) is a pure write that
is certain to run whenever the unit is entered, and ``inout`` otherwise: read first, read in the
same statement, a partial (substring) assignment, or a write that can be skipped because it sits
inside a block IF / DO / logical IF or after an earlier jump or RETURN (the caller's value then
flows back out). Array dummies are ``inout`` when an element is written (directly or by a callee)
and ``in`` otherwise.

An ENTRY statement is an extra entry point of its unit with its own dummy arguments. It gets an
ApiSymbol of its own (same kind as the host unit, listed right after it); its accesses are taken
from the whole host unit, which over-approximates the statements the entry can reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ApiParam, ApiSymbol
from .effects import statement_effects
from .exprs import Name, Ref, Star
from .source import Statement
from .stmts import Call, Do, Io
from .symbols import Scope

_MAX_PASSES = 12
_ESCAPE_KINDS = frozenset(
    {"goto", "goto_computed", "goto_assigned", "if_arith", "return", "stop", "exit_cycle"}
)
_BRANCHING_IO_KEYS = frozenset({"end", "err", "eor"})


@dataclass(slots=True)
class _Event:
    kind: str  # "r" read, "w" write, "a" actual argument of a same-file candidate callee
    name: str
    callee: str = ""
    index: int = 0
    certain: bool = True  # runs on every path through the unit (not conditional, not skippable)


@dataclass(slots=True)
class _Entry:
    """A callable name of a unit: the unit itself or one of its ENTRY points."""

    scope: Scope
    name: str
    kind: str
    dummies: list[str]
    events: list[_Event] = field(default_factory=list)
    called: set[str] = field(default_factory=set)  # names used as procedures


def effective_parts(stmt: Statement) -> list[Statement]:
    return [stmt, stmt.inner] if stmt.inner is not None else [stmt]


class _Flow:
    """Tracks whether the next statement of a unit is certain to execute once the unit is entered."""

    def __init__(self) -> None:
        self.blocks: list[tuple[str, str | None]] = []  # ("if", None) or ("do", label)
        self.escaped = False  # an earlier statement may have jumped over the rest

    @property
    def certain(self) -> bool:
        return not self.blocks and not self.escaped

    def advance(self, stmt: Statement) -> None:
        """Update the state after ``stmt`` (its own certainty was read before the call)."""
        blocks = self.blocks
        if stmt.label:
            while blocks and blocks[-1][0] == "do" and blocks[-1][1] == stmt.label:
                blocks.pop()
        kind = stmt.kind
        if kind in {"do", "do_while"}:
            info = stmt.info
            blocks.append(("do", info.label if isinstance(info, Do) else None))
        elif kind == "if_then":
            blocks.append(("if", None))
        elif kind == "end_if":
            self._close("if", False)
        elif kind == "end_do":
            self._close("do", True)
        if _may_jump(stmt):
            self.escaped = True

    def _close(self, what: str, unlabeled: bool) -> None:
        for i in range(len(self.blocks) - 1, -1, -1):
            entry = self.blocks[i]
            if entry[0] == what and (not unlabeled or entry[1] is None):
                del self.blocks[i:]
                return


def _may_jump(stmt: Statement) -> bool:
    for part in effective_parts(stmt):
        if part.kind in _ESCAPE_KINDS:
            return True
        info = part.info
        if isinstance(info, Io) and _BRANCHING_IO_KEYS & info.controls.keys():
            return True
        if isinstance(info, Call) and any(isinstance(a, Star) and a.label for a in info.args):
            return True
    return False


def _collect(scope: Scope) -> tuple[list[_Event], set[str]]:
    events: list[_Event] = []
    called: set[str] = set()
    flow = _Flow()
    for stmt in scope.unit.body:
        here = flow.certain
        for part in effective_parts(stmt):
            if part.invalid or part.info is None:
                continue
            certain = here and part is stmt
            eff = statement_effects(part, scope)
            reads, writes = eff.reads, eff.writes
            if part.kind == "stmt_function":
                # F(X) = ...: the formals shadow unit variables and nothing is assigned
                formals = set(scope.stmt_functions[part.info.target.name].dummies)
                reads = [n for n in reads if n not in formals]
                writes = []
            events.extend(_Event("r", n) for n in reads)
            for site in eff.sites:
                called.add(site.callee)
                for i, arg in enumerate(site.args):
                    if isinstance(arg, Name) or (isinstance(arg, Ref) and scope.is_array(arg.name)):
                        events.append(_Event("a", arg.name, site.callee, i, certain))
            events.extend(_Event("w", n, certain=certain) for n in writes)
        flow.advance(stmt)
    return events, called


def _dummy_kind(scope: Scope, name: str, called: set[str]) -> str:
    sym = scope.symbols.get(name)
    if sym is not None and sym.dims:
        return "array"
    if (sym is not None and sym.is_external) or name in called:
        return "external"
    return "scalar"


def _entries(scopes: list[Scope]) -> list[_Entry]:
    result: list[_Entry] = []
    for scope in scopes:
        events, called = _collect(scope)
        result.append(_Entry(scope, scope.name, scope.kind, scope.dummies, events, called))
        if scope.kind in {"subroutine", "function"}:
            for name, dummies in scope.entry_points.items():
                result.append(_Entry(scope, name, scope.kind, dummies, events, called))
    return result


def compute_api(scopes: list[Scope]) -> tuple[ApiSymbol, ...]:
    entries = _entries(scopes)
    by_name: dict[str, _Entry] = {}
    for e in entries:
        if e.kind in {"subroutine", "function"}:
            by_name.setdefault(e.name, e)

    kinds = {
        id(e): {d: _dummy_kind(e.scope, d, e.called) for d in e.dummies if d != "*"}
        for e in entries
    }
    written: dict[int, set[str]] = {
        id(e): {x.name for x in e.events if x.kind == "w" and x.name in kinds[id(e)]}
        for e in entries
    }
    for _ in range(_MAX_PASSES * len(entries) + 1):
        changed = False
        for e in entries:
            mine = kinds[id(e)]
            for x in e.events:
                if x.kind != "a" or x.name not in mine or x.name in written[id(e)]:
                    continue
                callee = by_name.get(x.callee)
                if callee is None or x.index >= len(callee.dummies):
                    continue
                if callee.dummies[x.index] in written[id(callee)]:
                    written[id(e)].add(x.name)
                    changed = True
        if not changed:
            break

    intents = _intents(entries, by_name, kinds, written)
    return tuple(_symbol(e, kinds[id(e)], intents[id(e)]) for e in entries)


def _intents(
    entries: list[_Entry],
    by_name: dict[str, _Entry],
    kinds: dict[int, dict[str, str]],
    written: dict[int, set[str]],
) -> dict[int, dict[str, str]]:
    intents: dict[int, dict[str, str]] = {
        id(e): {d: ("inout" if d in written[id(e)] else "in") for d in kinds[id(e)]}
        for e in entries
    }
    for _ in range(_MAX_PASSES):
        changed = False
        for e in entries:
            for name, kind in kinds[id(e)].items():
                if kind != "scalar" or name not in written[id(e)]:
                    continue
                new = "inout" if _read_first(e, name, by_name, written, intents) else "out"
                if intents[id(e)][name] != new:
                    intents[id(e)][name] = new
                    changed = True
        if not changed:
            break
    return intents


def _read_first(
    entry: _Entry,
    name: str,
    by_name: dict[str, _Entry],
    written: dict[int, set[str]],
    intents: dict[int, dict[str, str]],
) -> bool:
    for e in entry.events:
        if e.name != name:
            continue
        if e.kind == "r":
            return True
        if e.kind == "w":
            return not e.certain
        callee = by_name.get(e.callee)
        if callee is None or e.index >= len(callee.dummies):
            return True
        dummy = callee.dummies[e.index]
        if dummy not in written[id(callee)]:
            return True  # only read by the callee
        return not e.certain or intents[id(callee)].get(dummy, "inout") == "inout"
    return False


def _symbol(entry: _Entry, kinds: dict[str, str], intents: dict[str, str]) -> ApiSymbol:
    scope = entry.scope
    params: list[ApiParam] = []
    returns: list[str] = []
    for name in entry.dummies:
        if name == "*":
            continue
        kind = kinds[name]
        sym = scope.symbols.get(name)
        params.append(
            ApiParam(
                name=name,
                type="external" if kind == "external" else (scope.type_of_name(name) or "unknown"),
                rank=len(sym.dims) if sym is not None else 0,
                intent=intents[name],
            )
        )
        if kind == "scalar" and intents[name] != "in":
            returns.append(name)
    result_type = None
    if entry.kind == "function":
        result_type = _result_type(scope, entry.name)
    return ApiSymbol(
        name=entry.name,
        kind=entry.kind,
        params=tuple(params),
        result_type=result_type,
        returns=tuple(returns),
        alt_returns=entry.dummies.count("*"),
    )


def _result_type(scope: Scope, name: str) -> str:
    if name == scope.name and scope.result_type:
        return scope.result_type
    return scope.type_of_name(name) or scope.implicit_type(name) or "unknown"
