"""Detectors: one rule per id in :data:`modernizer.features.FEATURES`.

Detection runs over the logical statements and the per-unit symbol table, not over raw text, so
strings, comments and declarations cannot cause false positives by construction. The heuristic
rules are:

``integer_division``      ``a / b`` whose operands are both integer typed (symbol table, implicit
                          rules, integer literals, integer intrinsics); any real/unknown operand
                          means no hit.
``implicit_typing``       first use of a name that has no type declaration, in a unit without
                          IMPLICIT NONE (function results, intrinsic and procedure names excluded).
``intrinsic_specific``    call of a type-specific name (DABS, AMAX1, IFIX ...).
``intrinsic_semantics``   call of SIGN/MOD/NINT/INT/AINT/ANINT/DIM or their specific variants.
``sequence_association``  array element (or an array of another rank) passed to an array dummy of
                          a callee defined in the same file.
``machine_constants``     call of D1MACH/R1MACH/I1MACH/SLAMCH/DLAMCH.
``character_star``        CHARACTER*n / CHARACTER*(*) / ``name*n`` length specs (not LEN=).
``save_statement``        SAVE, or DATA initialising a local variable of a SUBROUTINE/FUNCTION.
``implicit_statement``    IMPLICIT with type rules (IMPLICIT NONE is the modern form: no hit).
``fixed_form_source``     one hit on the first fixed-form column marker of a fixed-form file.
``vendor_extension``      INCLUDE, NAMELIST, STRUCTURE/RECORD/UNION/MAP, BYTE, DO WHILE, END DO
                          loops, EXIT/CYCLE, DEC TYPE/ACCEPT I/O and Fortran 90 constructs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..features import FEATURES
from ..models import FeatureHit
from .api import effective_parts
from .effects import CallSite, Effects, statement_effects, statement_exprs
from .exprs import Bin, Cplx, Name, Num, Range, Ref, Star, target_names, walk
from .intrinsics import (
    ALL_INTRINSICS,
    BIT_INTRINSICS,
    COMPLEX_INTRINSICS,
    MACHINE_CONSTANTS,
    SEMANTIC_INTRINSICS,
    SPECIFIC_INTRINSICS,
)
from .source import FIXED, SourceModel, Statement
from .stmts import (
    Assign,
    Common,
    Data,
    Decl,
    Dimension,
    Do,
    Entity,
    Entry,
    Equivalence,
    Header,
    Implicit,
    Io,
    NameList,
    Parameter,
    Return,
    TypeSpec,
)
from .structure import scan_structure
from .symbols import Scope, expr_type, is_intrinsic_call

_SNIPPET_LIMIT = 120
_DOTTED = re.compile(r"\.(?:EQ|NE|LT|LE|GT|GE|AND|OR|NOT|EQV|NEQV|XOR)\.", re.IGNORECASE)
_D_EXPONENT = re.compile(r"(?<![\w.$])(?:\d+\.?\d*|\.\d+)[dD][+-]?\d+(?![\w])")
_VENDOR_PERCENT = re.compile(r"%\s*(?:VAL|LOC|REF|DESCR)\s*\(", re.IGNORECASE)
_VENDOR_KINDS = frozenset(
    {"include", "namelist", "vendor_block", "vendor_decl", "exit_cycle", "f90", "contains"}
)
_FORMAT_FREE_KINDS = frozenset({"format"})


def snippet_of(stmt: Statement) -> str:
    text = " ".join(stmt.raw.split())
    if len(text) > _SNIPPET_LIMIT:
        text = text[: _SNIPPET_LIMIT - 3] + "..."
    return text


@dataclass(slots=True)
class ModuleContext:
    scopes: list[Scope]
    by_name: dict[str, Scope]  # callable units (subroutine/function) by lower-case name
    unit_names: frozenset[str]


def build_context(scopes: list[Scope]) -> ModuleContext:
    by_name: dict[str, Scope] = {}
    for scope in scopes:
        if scope.kind in {"subroutine", "function"}:
            by_name.setdefault(scope.name, scope)
    unit_types = {
        s.name: s.result_type or s.implicit_type(s.name)
        for s in by_name.values()
        if s.kind == "function"
    }
    for scope in scopes:
        scope.unit_types = unit_types
    return ModuleContext(scopes, by_name, frozenset(by_name))


class _UnitDetector:
    def __init__(self, ctx: ModuleContext, scope: Scope) -> None:
        self.ctx = ctx
        self.scope = scope
        self.hits: list[FeatureHit] = []
        self._seen: set[tuple[str, int, int]] = set()

    def _effects(self, part: Statement) -> Effects:
        return statement_effects(part, self.scope)

    # -- output ---------------------------------------------------------------------------
    def add(
        self, feature: str, stmt: Statement, *, end_line: int | None = None, line: int | None = None
    ) -> None:
        assert feature in FEATURES, feature
        start = line if line is not None else stmt.line
        end = end_line if end_line is not None else stmt.end_line
        key = (feature, start, end)
        if key in self._seen:
            return
        self._seen.add(key)
        self.hits.append(FeatureHit(feature, start, end, snippet_of(stmt)))

    # -- driver ---------------------------------------------------------------------------
    def run(self) -> list[FeatureHit]:
        scope = self.scope
        unit = scope.unit
        header = unit.header
        if header is not None:
            self._header(header)
        structure = scan_structure(unit.body)
        for first, terminal, _ in structure.shared_terminals:
            self.add("do_shared_terminal", first, end_line=terminal.end_line)
        for loop, end in structure.block_loops:
            self.add("vendor_extension", loop, end_line=(end or loop).end_line)
        for stmt in unit.statements:
            if stmt is unit.end:
                continue
            self._text_level(stmt)
            for part in effective_parts(stmt):
                self._dispatch(part, stmt)
                if part.info is not None and not part.invalid:
                    self._expressions(part, stmt)
        self._implicit_typing()
        self._arg_output()
        return self.hits

    # -- statement level ------------------------------------------------------------------
    def _header(self, stmt: Statement) -> None:
        info = stmt.info
        if stmt.kind == "block_data":
            self.add("block_data", stmt)
        if "*" in self.scope.dummies:
            self.add("alternate_return", stmt)
        if isinstance(info, Header) and info.type_spec is not None:
            self._type_features(info.type_spec, stmt, [])

    def _text_level(self, stmt: Statement) -> None:
        if stmt.n_hollerith:
            self.add("hollerith", stmt)
        if stmt.kind in _FORMAT_FREE_KINDS:
            return
        masked = stmt.masked
        if _VENDOR_PERCENT.search(masked):
            self.add("vendor_extension", stmt)
        if _DOTTED.search(masked):
            self.add("logical_dotted", stmt)
        if _D_EXPONENT.search(masked):
            self.add("exponent_literal_d", stmt)

    def _dispatch(self, part: Statement, stmt: Statement) -> None:
        kind = part.kind
        info = part.info
        add = self.add
        if kind == "goto":
            add("goto_unconditional", stmt)
        elif kind == "goto_computed":
            add("goto_computed", stmt)
        elif kind in {"goto_assigned", "assign_label"}:
            add("goto_assigned", stmt)
        elif kind == "if_arith":
            add("if_arithmetic", stmt)
        elif kind == "if_logical":
            add("if_logical_single", stmt)
        elif kind == "do":
            self._do(part, stmt)
        elif kind == "do_while":
            pass  # reported with its END DO by the block pass
        elif kind in {"stop", "pause"}:
            add("stop_pause", stmt)
        elif kind == "return":
            if isinstance(info, Return) and info.expr is not None:
                add("alternate_return", stmt)
        elif kind == "entry":
            add("entry_statement", stmt)
            if isinstance(info, Entry) and "*" in info.args:
                add("alternate_return", stmt)
        elif kind == "common":
            add("common_block", stmt)
            if isinstance(info, Common):
                for _, entities in info.blocks:
                    self._entities(entities, stmt)
        elif kind == "equivalence":
            add("equivalence", stmt)
        elif kind == "data":
            self._data(part, stmt)
        elif kind == "save":
            add("save_statement", stmt)
        elif kind == "implicit":
            if isinstance(info, Implicit) and not info.none:
                add("implicit_statement", stmt)
                for spec, _ in info.rules:
                    self._type_features(spec, stmt, [])
        elif kind == "parameter":
            add("parameter_statement", stmt)
        elif kind == "declaration":
            if isinstance(info, Decl):
                self._declaration(info, stmt)
        elif kind == "dimension":
            if isinstance(info, Dimension):
                self._entities(info.entities, stmt)
        elif kind in {"external", "intrinsic"}:
            add("external_intrinsic_stmt", stmt)
            if kind == "external" and isinstance(info, NameList):
                for name in info.names:
                    sym = self.scope.symbols.get(name)
                    if sym is not None and sym.is_dummy:
                        add("procedure_argument", stmt)
        elif kind == "stmt_function":
            add("statement_function", stmt)
        elif kind == "format":
            add("format_statement", stmt)
        elif kind in {"read", "write", "print"}:
            if isinstance(info, Io) and info.fmt is not None:
                add("io_list_directed" if isinstance(info.fmt, Star) else "io_formatted", stmt)
            if part.text[:6].upper().startswith(("TYPE", "ACCEPT")):
                add("vendor_extension", stmt)  # DEC spellings of PRINT and READ
        elif kind == "io_file":
            add("io_file_control", stmt)
        if kind in _VENDOR_KINDS:
            add("vendor_extension", stmt)

    def _do(self, part: Statement, stmt: Statement) -> None:
        info = part.info
        if not isinstance(info, Do):
            return
        if info.label is not None:
            self.add("do_labeled", stmt)
        if info.var and self.scope.type_of_name(info.var) in {"real", "double"}:
            self.add("do_real_index", stmt)

    def _data(self, part: Statement, stmt: Statement) -> None:
        self.add("data_statement", stmt)
        info = part.info
        if not isinstance(info, Data) or self.scope.kind not in {"subroutine", "function"}:
            return
        for target in info.targets:
            for name in target_names(target):
                sym = self.scope.symbols.get(name)
                if sym is None or (sym.common is None and not sym.is_dummy):
                    self.add("save_statement", stmt)
                    return

    # -- declarations ---------------------------------------------------------------------
    def _declaration(self, info: Decl, stmt: Statement) -> None:
        self._type_features(info.spec, stmt, info.entities)
        if "parameter" in info.attrs:
            self.add("parameter_statement", stmt)
        if "external" in info.attrs or "intrinsic" in info.attrs:
            self.add("external_intrinsic_stmt", stmt)
        if "save" in info.attrs:
            self.add("save_statement", stmt)
        self._entities(info.entities, stmt)

    def _type_features(self, spec: TypeSpec, stmt: Statement, entities: list[Entity]) -> None:
        base = spec.base
        has_star = "*" in spec.text
        if base == "double":
            self.add("double_precision", stmt)
        elif base == "doublecomplex":
            self.add("complex_type", stmt)
            self.add("real_kind_star", stmt)
        elif base == "character":
            if has_star or any(e.char_len for e in entities):
                self.add("character_star", stmt)
        elif base in {"byte", "derived"}:
            self.add("vendor_extension", stmt)
        else:
            if base == "complex":
                self.add("complex_type", stmt)
            if spec.star is not None:
                self.add("real_kind_star", stmt)

    def _entities(self, entities: list[Entity], stmt: Statement) -> None:
        for entity in entities:
            dims = entity.dims
            if not dims:
                continue
            if len(dims) >= 2:
                self.add("array_multidim", stmt)
            for dim in dims:
                if dim.lower is not None and not _is_one(dim.lower):
                    self.add("array_custom_bounds", stmt)
            if dims[-1].upper is None:
                self.add("array_assumed_size", stmt)
            sym = self.scope.symbols.get(entity.name)
            if sym is not None and sym.is_dummy and self._variable_bounds(dims):
                self.add("array_adjustable", stmt)

    def _variable_bounds(self, dims: tuple) -> bool:
        for dim in dims:
            for bound in (dim.lower, dim.upper):
                if bound is None:
                    continue
                for node in walk(bound):
                    if isinstance(node, Name):
                        sym = self.scope.symbols.get(node.name)
                        if sym is None or not sym.is_parameter:
                            return True
        return False

    # -- expression level -----------------------------------------------------------------
    def _expressions(self, part: Statement, stmt: Statement) -> None:
        for top in statement_exprs(part.info):
            for node in walk(top):
                if isinstance(node, Bin):
                    self._binary(node, stmt)
                elif isinstance(node, Cplx):
                    self.add("complex_type", stmt)
                elif isinstance(node, Ref):
                    self._reference(node, stmt)
        for site in self._effects(part).sites:
            self._call_site(site, stmt)

    def _binary(self, node: Bin, stmt: Statement) -> None:
        if node.op == "/":
            left = expr_type(node.left, self.scope)
            right = expr_type(node.right, self.scope)
            if left == "integer" and right == "integer":
                self.add("integer_division", stmt)
        elif node.op == "//":
            self.add("character_substring", stmt)

    def _reference(self, ref: Ref, stmt: Statement) -> None:
        scope = self.scope
        name = ref.name
        has_range = any(isinstance(a, Range) for a in ref.args)
        if ref.sub is not None or has_range:
            if scope.type_of_name(name) in {"character", None} or not scope.is_array(name):
                self.add("character_substring", stmt)
        if scope.is_array(name):
            return
        if name in MACHINE_CONSTANTS:
            self.add("machine_constants", stmt)
        if not is_intrinsic_call(ref, scope):
            return
        if name in SPECIFIC_INTRINSICS:
            self.add("intrinsic_specific", stmt)
        if name in SEMANTIC_INTRINSICS:
            self.add("intrinsic_semantics", stmt)
        if name in BIT_INTRINSICS:
            self.add("bit_intrinsics", stmt)
        if name in COMPLEX_INTRINSICS:
            self.add("complex_type", stmt)

    def _call_site(self, site: CallSite, stmt: Statement) -> None:
        scope = self.scope
        callee = self.ctx.by_name.get(site.callee)
        for i, arg in enumerate(site.args):
            if isinstance(arg, Star) and arg.label:
                self.add("alternate_return", stmt)
            if isinstance(arg, Name) and self._is_procedure(arg.name):
                self.add("procedure_argument", stmt)
            if callee is None or i >= len(callee.dummies):
                continue
            dummy = callee.symbols.get(callee.dummies[i])
            dummy_rank = dummy.rank if dummy is not None else 0
            if dummy_rank == 0:
                continue
            if isinstance(arg, Ref) and scope.is_array(arg.name) and arg.sub is None:
                if not any(isinstance(a, Range) for a in arg.args):
                    self.add("sequence_association", stmt)
            elif isinstance(arg, Name) and scope.is_array(arg.name):
                actual = scope.symbols[arg.name]
                if actual.rank != dummy_rank:
                    self.add("sequence_association", stmt)

    def _is_procedure(self, name: str) -> bool:
        return is_procedure_name(self.ctx, self.scope, name)

    # -- whole-unit passes ----------------------------------------------------------------
    def _arg_output(self) -> None:
        scope = self.scope
        pending = {d for d in scope.dummies if d != "*"}
        for entry_dummies in scope.entry_points.values():
            pending.update(d for d in entry_dummies if d != "*")
        for stmt in scope.unit.body:
            for part in effective_parts(stmt):
                if part.info is None or part.invalid or part.kind == "stmt_function":
                    continue
                for name in self._effects(part).writes:
                    sym = scope.symbols.get(name)
                    if name in pending and not (sym is not None and sym.is_external):
                        pending.discard(name)
                        self.add("arg_output", stmt)

    def _implicit_typing(self) -> None:
        scope = self.scope
        if scope.implicit_none:
            return
        assigned = {
            n
            for stmt in scope.unit.body
            for part in effective_parts(stmt)
            if part.info is not None and not part.invalid
            for n in self._effects(part).writes
        }
        seen: set[str] = set()
        for stmt in scope.unit.body:
            for part in effective_parts(stmt):
                if part.info is None or part.invalid:
                    continue
                for name in self._used_names(part):
                    if name in seen:
                        continue
                    seen.add(name)
                    if self._relies_on_implicit(name, assigned):
                        self.add("implicit_typing", stmt)

    def _used_names(self, part: Statement) -> list[str]:
        scope = self.scope
        info = part.info
        kind = part.kind
        if kind in {"external", "intrinsic", "save", "implicit"}:
            return []
        if isinstance(info, Decl | Dimension | Common):
            entities = (
                [e for _, ents in info.blocks for e in ents]
                if isinstance(info, Common)
                else info.entities
            )
            names = [] if isinstance(info, Decl) else [e.name for e in entities]
            for bound in statement_exprs(info):
                names.extend(n.name for n in walk(bound) if isinstance(n, Name))
            return names
        if isinstance(info, Parameter):
            return [name for name, _ in info.items]
        if isinstance(info, Equivalence):
            return [
                n.name for s in info.sets for e in s for n in walk(e) if isinstance(n, Name | Ref)
            ]
        if isinstance(info, Data):
            return [n for t in info.targets for n in target_names(t)]
        eff = self._effects(part)
        names = list(eff.reads)
        for site in eff.sites:
            names.extend(a.name for a in site.args if isinstance(a, Name))
        names.extend(eff.writes)
        if kind == "stmt_function" and isinstance(info, Assign) and isinstance(info.target, Ref):
            dummies = scope.stmt_functions[info.target.name].dummies
            return [n for n in names if n not in dummies and n != info.target.name]
        return names

    def _relies_on_implicit(self, name: str, assigned: set[str]) -> bool:
        scope = self.scope
        sym = scope.symbols.get(name)
        if sym is not None and (sym.typed or sym.is_external or sym.is_intrinsic):
            return False
        if name in {scope.name, scope.result_name} or name in scope.stmt_functions:
            return False
        if name in self.ctx.unit_names:
            return False
        is_variable = sym is not None and (
            sym.dims or sym.common or sym.is_dummy or sym.is_parameter
        )
        return not (name in ALL_INTRINSICS and name not in assigned and not is_variable)


def is_procedure_name(ctx: ModuleContext, scope: Scope, name: str) -> bool:
    """True when the bare name ``name`` denotes a procedure rather than a variable."""
    sym = scope.symbols.get(name)
    if sym is not None and (sym.is_external or sym.is_intrinsic):
        return True
    if name == scope.name or name not in ctx.unit_names:
        return False
    return not (sym is not None and (sym.typed or sym.dims or sym.is_dummy))


def unit_references(ctx: ModuleContext, scope: Scope) -> tuple[str, ...]:
    """Procedures called or passed as arguments by the unit (first-seen order, no self)."""
    names: dict[str, None] = {}
    dummies = set(scope.dummies)
    for stmt in scope.unit.body:
        for part in effective_parts(stmt):
            if part.info is None or part.invalid:
                continue
            for site in statement_effects(part, scope).sites:
                if site.callee not in dummies:
                    names[site.callee] = None
                for arg in site.args:
                    if isinstance(arg, Name) and is_procedure_name(ctx, scope, arg.name):
                        names[arg.name] = None
    names.pop(scope.name, None)
    return tuple(names)


def _is_one(expr: object) -> bool:
    return isinstance(expr, Num) and expr.kind == "integer" and int(expr.text) == 1


def detect_module(
    model: SourceModel, ctx: ModuleContext, extra_scopes: list[Scope] | None = None
) -> list[FeatureHit]:
    """All hits of the module; ``extra_scopes`` hold statements that sit outside any unit."""
    hits: list[FeatureHit] = []
    for scope in [*ctx.scopes, *(extra_scopes or [])]:
        hits.extend(_UnitDetector(ctx, scope).run())
    hits.extend(_fixed_form_hit(model))
    hits.sort(key=lambda h: (h.line, h.feature_id, h.end_line))
    return hits


def _fixed_form_hit(model: SourceModel) -> list[FeatureHit]:
    if model.form != FIXED or not model.statements:
        return []
    line = model.marker_lines[0] if model.marker_lines else model.statements[0].line
    text = model.lines[line - 1].strip() if 0 < line <= len(model.lines) else ""
    return [FeatureHit("fixed_form_source", line, line, text[:_SNIPPET_LIMIT])]


__all__ = [
    "ModuleContext",
    "build_context",
    "detect_module",
    "is_procedure_name",
    "snippet_of",
    "unit_references",
]
