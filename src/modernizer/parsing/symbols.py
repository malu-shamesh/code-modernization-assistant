"""Per-unit symbol tables, implicit typing rules and expression type inference."""

from __future__ import annotations

from dataclasses import dataclass, field

from .exprs import (
    Bin,
    Cplx,
    Expr,
    Logical,
    Name,
    Num,
    Paren,
    Range,
    Ref,
    Str,
    Un,
    target_names,
)
from .intrinsics import ALL_INTRINSICS, intrinsic_type, promote
from .source import EXECUTABLE_KINDS, Statement
from .stmts import (
    Assign,
    Common,
    Data,
    Decl,
    Dim,
    Dimension,
    Entity,
    Entry,
    Equivalence,
    Header,
    Implicit,
    NameList,
    Parameter,
)
from .units import UnitBlock

_RELATIONAL_OPS = frozenset(
    {".EQ.", ".NE.", ".LT.", ".LE.", ".GT.", ".GE.", "==", "/=", "<", "<=", ">", ">="}
)
_LOGICAL_OPS = frozenset({".AND.", ".OR.", ".EQV.", ".NEQV.", ".XOR."})
_ARITHMETIC_OPS = frozenset({"+", "-", "*", "/", "**"})


@dataclass(slots=True)
class Symbol:
    name: str
    type: str | None = None  # integer real double complex logical character (explicit only)
    dims: tuple[Dim, ...] = ()
    char_len: str | None = None
    is_dummy: bool = False
    is_external: bool = False
    is_intrinsic: bool = False
    is_parameter: bool = False
    value: Expr | None = None  # PARAMETER value
    common: str | None = None
    saved: bool = False
    data_init: bool = False
    typed: bool = False  # has an explicit type declaration
    declared_line: int = 0

    @property
    def rank(self) -> int:
        return len(self.dims)


@dataclass(slots=True)
class StatementFunction:
    name: str
    dummies: tuple[str, ...]
    body: Expr
    statement: Statement


@dataclass(slots=True)
class Scope:
    """Declared names and typing rules of one program unit."""

    unit: UnitBlock
    kind: str
    name: str
    dummies: list[str] = field(default_factory=list)  # '*' marks alternate-return dummies
    result_name: str | None = None
    result_type: str | None = None
    implicit_none: bool = False
    implicit: dict[str, str] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    stmt_functions: dict[str, StatementFunction] = field(default_factory=dict)
    entry_points: dict[str, list[str]] = field(default_factory=dict)  # ENTRY name -> dummies
    save_all: bool = False
    unit_types: dict[str, str | None] = field(default_factory=dict)  # same-file function results

    def symbol(self, name: str) -> Symbol:
        sym = self.symbols.get(name)
        if sym is None:
            sym = self.symbols[name] = Symbol(name)
        return sym

    def implicit_type(self, name: str) -> str | None:
        if self.implicit_none or not name:
            return None
        letter = name[0].lower()
        if letter in self.implicit:
            return self.implicit[letter]
        if "a" <= letter <= "z":
            return "integer" if "i" <= letter <= "n" else "real"
        return None

    def type_of_name(self, name: str) -> str | None:
        sym = self.symbols.get(name)
        if sym is not None and sym.type:
            return sym.type
        if name == self.result_name and self.result_type:
            return self.result_type
        return self.implicit_type(name)

    def is_array(self, name: str) -> bool:
        sym = self.symbols.get(name)
        return sym is not None and bool(sym.dims)


def _apply_entity(scope: Scope, entity: Entity, line: int) -> Symbol:
    sym = scope.symbol(entity.name)
    if entity.dims:
        sym.dims = entity.dims
    if entity.char_len is not None:
        sym.char_len = entity.char_len
    if not sym.declared_line:
        sym.declared_line = line
    return sym


def _build_implicit(scope: Scope, statements: list[Statement]) -> None:
    for stmt in statements:
        info = stmt.info
        if isinstance(info, Implicit):
            if info.none:
                scope.implicit_none = True
            for spec, ranges in info.rules:
                for lo, hi in ranges:
                    for code in range(ord(lo), ord(hi) + 1):
                        scope.implicit[chr(code)] = spec.api_type


def build_scope(unit: UnitBlock) -> Scope:
    """Walk the specification statements of ``unit`` and record every declared name."""
    header = unit.header.info if unit.header is not None else None
    scope = Scope(unit, unit.kind, unit.name)
    if isinstance(header, Header):
        scope.dummies = list(header.args)
        if unit.kind == "function":
            scope.result_name = header.result or header.name
            if header.type_spec is not None:
                scope.result_type = header.type_spec.api_type
        for dummy in header.args:
            if dummy != "*":
                scope.symbol(dummy).is_dummy = True
    _build_implicit(scope, unit.statements)
    seen_executable = False
    for stmt in unit.body:
        info = stmt.info
        line = stmt.line
        if isinstance(info, Decl):
            for entity in info.entities:
                if info.attr_dims and not entity.dims:
                    entity = Entity(entity.name, info.attr_dims, entity.char_len)
                sym = _apply_entity(scope, entity, line)
                sym.type = info.spec.api_type
                sym.typed = True
                if info.spec.base == "character" and sym.char_len is None:
                    sym.char_len = info.spec.star
                if "parameter" in info.attrs:
                    sym.is_parameter = True
                if "external" in info.attrs:
                    sym.is_external = True
                if "save" in info.attrs:
                    sym.saved = True
                if entity.name == scope.result_name and scope.kind == "function":
                    scope.result_type = sym.type
        elif isinstance(info, Dimension):
            for entity in info.entities:
                _apply_entity(scope, entity, line)
        elif isinstance(info, Common):
            for block, entities in info.blocks:
                for entity in entities:
                    sym = _apply_entity(scope, entity, line)
                    sym.common = block
        elif isinstance(info, Parameter):
            for name, value in info.items:
                sym = scope.symbol(name)
                sym.is_parameter = True
                sym.value = value
                sym.declared_line = sym.declared_line or line
        elif isinstance(info, NameList) and stmt.kind in {"external", "intrinsic", "save"}:
            _apply_name_list(scope, stmt.kind, info)
        elif isinstance(info, Data):
            for target in info.targets:
                for name in target_names(target):
                    scope.symbol(name).data_init = True
        elif isinstance(info, Equivalence):
            pass
        elif isinstance(info, Entry):
            scope.entry_points[info.name] = info.args
            for dummy in info.args:
                if dummy != "*":
                    scope.symbol(dummy).is_dummy = True
        elif stmt.kind == "assign" and not seen_executable and _is_statement_function(scope, stmt):
            assert isinstance(info, Assign) and isinstance(info.target, Ref)
            stmt.kind = "stmt_function"
            scope.stmt_functions[info.target.name] = StatementFunction(
                info.target.name,
                tuple(a.name for a in info.target.args if isinstance(a, Name)),
                info.value,
                stmt,
            )
        elif stmt.kind in EXECUTABLE_KINDS:
            seen_executable = True
    return scope


def _apply_name_list(scope: Scope, kind: str, info: NameList) -> None:
    if kind == "save":
        if not info.names and not info.blocks:
            scope.save_all = True
        for name in info.names:
            scope.symbol(name).saved = True
        return
    for name in info.names:
        sym = scope.symbol(name)
        if kind == "external":
            sym.is_external = True
        else:
            sym.is_intrinsic = True


def _is_statement_function(scope: Scope, stmt: Statement) -> bool:
    info = stmt.info
    if not isinstance(info, Assign) or not isinstance(info.target, Ref):
        return False
    target = info.target
    if scope.is_array(target.name) or target.sub is not None or not target.args:
        return False
    if not all(isinstance(a, Name) for a in target.args):
        return False
    sym = scope.symbols.get(target.name)
    return not (sym is not None and (sym.is_dummy or sym.common is not None))


def expr_type(expr: Expr, scope: Scope) -> str | None:
    """Static type of ``expr`` (integer real double complex logical character) or ``None``."""
    try:
        return _expr_type(expr, scope)
    except RecursionError:  # absurdly deep expression: unknown rather than a crash
        return None


def _expr_type(expr: Expr, scope: Scope) -> str | None:
    if isinstance(expr, Num):
        return expr.kind
    if isinstance(expr, Str):
        return "character"
    if isinstance(expr, Logical):
        return "logical"
    if isinstance(expr, Cplx):
        return "complex"
    if isinstance(expr, Name):
        return scope.type_of_name(expr.name)
    if isinstance(expr, Paren):
        return _expr_type(expr.inner, scope)
    if isinstance(expr, Un):
        if expr.op == ".NOT.":
            return "logical"
        return _expr_type(expr.operand, scope)
    if isinstance(expr, Bin):
        if expr.op in _RELATIONAL_OPS or expr.op in _LOGICAL_OPS:
            return "logical"
        if expr.op == "//":
            return "character"
        if expr.op in _ARITHMETIC_OPS:
            return promote(_expr_type(expr.left, scope), _expr_type(expr.right, scope))
        return None
    if isinstance(expr, Ref):
        return _ref_type(expr, scope)
    return None


def is_intrinsic_call(ref: Ref, scope: Scope) -> bool:
    """True when ``ref`` is a call to an intrinsic function rather than an array or user function."""
    if ref.name not in ALL_INTRINSICS or scope.is_array(ref.name):
        return False
    if ref.name in scope.stmt_functions or ref.name in scope.unit_types:
        return False
    sym = scope.symbols.get(ref.name)
    return not (sym is not None and (sym.is_external or sym.is_dummy or sym.typed and sym.dims))


def _ref_type(ref: Ref, scope: Scope) -> str | None:
    if ref.sub is not None or any(isinstance(a, Range) for a in ref.args):
        return "character"
    if scope.is_array(ref.name) or ref.name in scope.stmt_functions:
        return scope.type_of_name(ref.name)
    if is_intrinsic_call(ref, scope):
        return intrinsic_type(ref.name, [_expr_type(a, scope) for a in ref.args])
    sym = scope.symbols.get(ref.name)
    if sym is not None and sym.type:
        return sym.type
    if ref.name in scope.unit_types and scope.unit_types[ref.name]:
        return scope.unit_types[ref.name]
    return scope.implicit_type(ref.name)
