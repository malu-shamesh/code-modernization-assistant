"""Entry points: Fortran source -> units, feature hits, metrics and API symbols."""

from __future__ import annotations

import logging

from ..features import FEATURES
from ..models import ApiSymbol, CodeUnit, FeatureHit, ModuleMetrics, ParsedModule
from .api import compute_api, effective_parts
from .detectors import ModuleContext, build_context, detect_module, unit_references
from .effects import statement_effects
from .grammar import error_spans
from .source import SourceModel, Statement, normalize_model, read_source
from .stmts import Common, parse_statement
from .structure import cyclomatic_branches, scan_structure
from .symbols import Scope, build_scope
from .units import UnitBlock, split_units

_LOG = logging.getLogger(__name__)


class _Analysis:
    """Everything derived from one source text (immutable after construction)."""

    def __init__(self, source: str) -> None:
        self.model: SourceModel = read_source(source)
        for stmt in self.model.statements:
            parse_statement(stmt)
        self.blocks: list[UnitBlock]
        self.blocks, orphans = split_units(self.model.statements)
        self.scopes: list[Scope] = [build_scope(block) for block in self.blocks]
        self.ctx: ModuleContext = build_context(self.scopes)
        extra = [build_scope(UnitBlock("program", "", None, orphans))] if orphans else []
        self.hits: list[FeatureHit] = detect_module(self.model, self.ctx, extra)
        self.api: tuple[ApiSymbol, ...] = compute_api(self.scopes)
        self._syntax_errors: bool | None = None

    # -- syntax verdict ---------------------------------------------------------------------
    def has_syntax_errors(self) -> bool:
        if self._syntax_errors is None:
            self._syntax_errors = self._compute_syntax_errors()
        return self._syntax_errors

    def _compute_syntax_errors(self) -> bool:
        statements = self.model.statements
        if any(scan_structure(b.body).unbalanced for b in self.blocks):
            return True
        text = normalize_model(self.model).text
        for first, last in error_spans(text):
            rows = statements[first : last + 1]
            if not rows:
                return True
            # A tree-sitter error is explained when the engine fully understands every
            # statement in its span (a known legacy form the grammar lacks).
            if any(_broken(stmt) for stmt in rows):
                return True
        return any(_unknown(stmt) for stmt in statements)

    # -- module construction ----------------------------------------------------------------
    def code_units(self) -> list[CodeUnit]:
        units = []
        for block, scope in zip(self.blocks, self.scopes, strict=True):
            start, end = block.start_line, block.end_line
            units.append(
                CodeUnit(
                    name=block.name,
                    kind=block.kind,
                    source="\n".join(self.model.lines[start - 1 : end]),
                    start_line=start,
                    end_line=end,
                    defines=_defines(scope),
                    references=unit_references(self.ctx, scope),
                    hits=[h for h in self.hits if start <= h.line <= end],
                )
            )
        return units

    def metrics(self) -> ModuleMetrics:
        model = self.model
        blocks = {name for scope in self.scopes for name in _common_blocks(scope)}
        distinct = {h.feature_id for h in self.hits}
        callable_units = {s.name for s in self.scopes}
        callable_units.update(e for s in self.scopes for e in s.entry_points)
        calls: dict[str, None] = {}
        for scope in self.scopes:
            for name in unit_references(self.ctx, scope):
                sym = scope.symbols.get(name)
                if name in callable_units or name in scope.dummies:
                    continue
                if not (sym is not None and sym.is_intrinsic):
                    calls[name] = None
        return ModuleMetrics(
            loc=len(model.lines),
            sloc=len(model.code_lines),
            n_functions=sum(1 for b in self.blocks if b.kind in {"subroutine", "function"}),
            n_common_blocks=len(blocks),
            max_nesting=max((scan_structure(b.body).max_depth for b in self.blocks), default=0),
            cyclomatic=1 + sum(cyclomatic_branches(b.body) for b in self.blocks),
            n_feature_hits=len(self.hits),
            distinct_features=len(distinct),
            weighted_feature_score=sum(FEATURES[h.feature_id].weight for h in self.hits),
            semantic_hits=sum(
                1 for h in self.hits if FEATURES[h.feature_id].severity == "semantic"
            ),
            has_syntax_errors=self.has_syntax_errors(),
            calls=tuple(sorted(calls)),
        )


def _unknown(stmt: Statement) -> bool:
    """True when the statement is not any construct the engine knows."""
    return stmt.kind == "unknown" or (stmt.inner is not None and stmt.inner.kind == "unknown")


def _broken(stmt: Statement) -> bool:
    return (
        stmt.invalid
        or stmt.kind == "unknown"
        or (stmt.inner is not None and (stmt.inner.invalid or stmt.inner.kind == "unknown"))
    )


def _common_blocks(scope: Scope) -> set[str]:
    names: set[str] = set()
    for stmt in scope.unit.body:
        if isinstance(stmt.info, Common):
            names.update(name for name, _ in stmt.info.blocks)
    return names


def _defines(scope: Scope) -> tuple[str, ...]:
    names: dict[str, None] = {}
    for name, sym in scope.symbols.items():
        if (sym.typed or sym.dims or sym.common is not None or sym.is_parameter) and not (
            sym.is_dummy
        ):
            names[name] = None
    for stmt in scope.unit.body:
        for part in effective_parts(stmt):
            if part.info is None or part.invalid or part.kind == "stmt_function":
                continue
            for name in statement_effects(part, scope).writes:
                if name not in scope.dummies:
                    names[name] = None
    for name in (*scope.stmt_functions, *scope.entry_points):
        names[name] = None
    return tuple(names)


def symbol_tables(source: str) -> dict[str, Scope]:
    """Symbol table of every program unit by lower-case unit name (first definition wins).

    Meant for tooling and tests; ``Scope`` exposes declared names, types, implicit rules,
    arrays, PARAMETER constants, COMMON members and statement functions.
    """
    analysis = _safe(source)
    if analysis is None:
        return {}
    tables: dict[str, Scope] = {}
    for scope in analysis.scopes:
        tables.setdefault(scope.name, scope)
    return tables


def _empty(path: str, source: object) -> ParsedModule:
    text = source if isinstance(source, str) else ""
    return ParsedModule(path, text, [], [], ModuleMetrics(has_syntax_errors=True))


def _safe(source: object) -> _Analysis | None:
    if not isinstance(source, str):
        return None
    try:
        return _Analysis(source)
    except Exception:  # noqa: BLE001 - garbage input must never crash the caller
        _LOG.debug("fortran analysis failed", exc_info=True)
        return None


def analyze(path: str, source: str) -> ParsedModule:
    """Parse ``source`` and describe it. Never raises: undecodable input gives a module whose
    metrics say ``has_syntax_errors``."""
    analysis = _safe(source)
    if analysis is None:
        return _empty(path, source)
    try:
        return ParsedModule(
            path=path,
            source=source,
            units=analysis.code_units(),
            hits=list(analysis.hits),
            metrics=analysis.metrics(),
            api=analysis.api,
        )
    except Exception:  # noqa: BLE001
        _LOG.debug("fortran analysis failed", exc_info=True)
        return _empty(path, source)


def detect_features(source: str) -> list[FeatureHit]:
    """Feature hits sorted by (line, feature id), original line numbers. Never raises."""
    analysis = _safe(source)
    return list(analysis.hits) if analysis is not None else []


def extract_api(source: str) -> tuple[ApiSymbol, ...]:
    """Per program unit contract (lower-case names). Never raises."""
    analysis = _safe(source)
    return analysis.api if analysis is not None else ()
