"""Splitting a large module into slices of program units, and extracting its shared state."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..models import ApiSymbol, CodeUnit, FeatureHit, ParsedModule
from .prompts import STATE_FEATURES, format_signatures

UNIT_SEPARATOR = "\n\n"

# Presence of either means several units may share storage, so a prelude is worth a call.
_SHARED_STATE_FEATURES = frozenset({"common_block", "block_data"})
_STATE_FEATURE_SET = frozenset(STATE_FEATURES)

# Statement keywords (blanks removed, upper case) that declare storage or initial values.
_DECLARATION_PREFIXES = (
    "COMMON",
    "SAVE",
    "EQUIVALENCE",
    "DIMENSION",
    "IMPLICIT",
    "PARAMETER",
    "INTEGER",
    "REAL",
    "DOUBLEPRECISION",
    "COMPLEX",
    "LOGICAL",
    "CHARACTER",
)
_STATE_PREFIXES = ("COMMON", "SAVE", "EQUIVALENCE")
_HEADER_PREFIXES = ("SUBROUTINE", "FUNCTION", "PROGRAM", "BLOCKDATA")
_TYPED_FUNCTION = re.compile(
    r"^(INTEGER|REAL|DOUBLEPRECISION|COMPLEX|LOGICAL|CHARACTER)(\*\d+|\*\(.*?\))?FUNCTION"
)
_COMMENT_COLUMN1 = "CcDd*!"


@dataclass(frozen=True)
class Chunk:
    """A group of program units converted in one call, with the hits that fall inside it."""

    module: ParsedModule
    names: frozenset[str]

    @property
    def source(self) -> str:
        return self.module.source


# --------------------------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------------------------


def pack_units(module: ParsedModule, max_chars: int) -> list[list[int]]:
    """Group unit indexes, in source order, into chunks of at most ``max_chars`` characters.

    A single unit larger than the limit ends up alone in its own chunk.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, unit in enumerate(module.units):
        added = len(unit.source) + (len(UNIT_SEPARATOR) if current else 0)
        if current and size + added > max_chars:
            groups.append(current)
            current, size, added = [], 0, len(unit.source)
        current.append(index)
        size += added
    if current:
        groups.append(current)
    return groups


def _unit_for_hit(units: Sequence[CodeUnit], hit: FeatureHit) -> int:
    """Index of the unit containing ``hit``, else the next unit after it, else the last one."""
    for index, unit in enumerate(units):
        if unit.start_line <= hit.line <= unit.end_line:
            return index
    for index, unit in enumerate(units):
        if unit.start_line > hit.line:
            return index
    return len(units) - 1


def _unit_names(unit: CodeUnit) -> set[str]:
    return {unit.name.lower(), *(d.lower() for d in unit.defines)}


def build_chunks(module: ParsedModule, groups: Sequence[Sequence[int]]) -> list[Chunk]:
    """Join the units of each group into one source string.

    Hits keep their ORIGINAL line numbers (the prompt gutter numbers the slice the same way).
    """
    owner = {index: gi for gi, group in enumerate(groups) for index in group}
    hits_by_group: dict[int, list[FeatureHit]] = {gi: [] for gi in range(len(groups))}
    if module.units:
        for hit in module.hits:
            hits_by_group[owner[_unit_for_hit(module.units, hit)]].append(hit)

    chunks: list[Chunk] = []
    for gi, group in enumerate(groups):
        units = [module.units[i] for i in group]
        names: set[str] = set()
        for unit in units:
            names |= _unit_names(unit)
        chunk_module = replace(
            module,
            source=UNIT_SEPARATOR.join(u.source.rstrip("\n") for u in units),
            units=units,
            hits=hits_by_group[gi],
        )
        chunks.append(Chunk(chunk_module, frozenset(names)))
    return chunks


def chunk_api(module: ParsedModule, chunk: Chunk) -> tuple[ApiSymbol, ...]:
    """API symbols of the units in ``chunk`` (matched by unit name or ENTRY name)."""
    source = chunk.source
    selected: list[ApiSymbol] = []
    for symbol in module.api:
        name = symbol.name.lower()
        if name in chunk.names or re.search(
            rf"^\s*(?:\d+\s+)?ENTRY\s+{re.escape(name)}\b", source, re.IGNORECASE | re.MULTILINE
        ):
            selected.append(symbol)
    return tuple(selected)


def other_chunks_note(module: ParsedModule, chunks: Sequence[Chunk], index: int) -> str:
    """Python signatures of every unit defined in the chunks other than ``chunks[index]``."""
    lines: list[str] = []
    for other_index, other in enumerate(chunks):
        if other_index == index:
            continue
        symbols = chunk_api(module, other)
        if symbols:
            lines.append(format_signatures(symbols))
        else:
            lines.extend(
                f"- {u.kind} {u.name}" for u in other.module.units if u.kind != "statements"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------------------------


def has_shared_state(module: ParsedModule) -> bool:
    """True when COMMON or BLOCK DATA may couple the program units of ``module``."""
    return any(h.feature_id in _SHARED_STATE_FEATURES for h in module.hits) or any(
        u.kind == "block_data" for u in module.units
    )


def _strip_comment(text: str) -> str:
    """Drop a trailing '!' comment that is not inside a character constant."""
    quote = ""
    for index, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return text[:index]
    return text


def _has_toplevel_equals(code: str) -> bool:
    depth = 0
    quote = ""
    for ch in code:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "=" and depth == 0:
            return True
    return False


def _statements(lines: Sequence[str]) -> list[tuple[list[int], str]]:
    """Group fixed-form lines into statements: ``([line indexes], statement text)``."""
    statements: list[tuple[list[int], str]] = []
    for index, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip() or line[0] in _COMMENT_COLUMN1:
            continue
        is_continuation = len(line) > 5 and line[5] not in " 0" and not line[:5].strip()
        text = line[6:] if len(line) > 6 else ""
        if is_continuation and statements:
            statements[-1][0].append(index)
            statements[-1] = (statements[-1][0], statements[-1][1] + "\n" + text)
        else:
            statements.append(([index], text))
    return statements


def _squash(text: str) -> str:
    return "".join(_strip_comment(text).split()).upper()


def _is_header(code: str) -> bool:
    return code.startswith(_HEADER_PREFIXES) or bool(_TYPED_FUNCTION.match(code))


def _is_state_statement(code: str) -> bool:
    """COMMON, SAVE, EQUIVALENCE, DATA or BLOCK DATA: statements that create shared/kept state."""
    if _has_toplevel_equals(code):
        return False
    if code.startswith("DATA"):
        return "/" in code
    return code.startswith(("BLOCKDATA", *_STATE_PREFIXES))


def _is_declaration(code: str) -> bool:
    return code.startswith(_DECLARATION_PREFIXES) and not _has_toplevel_equals(code)


def state_lines(module: ParsedModule) -> list[tuple[int, str]]:
    """Original ``(line_number, text)`` of the state-related declarations of every unit.

    A unit contributes when it has a state statement (COMMON, DATA, SAVE, EQUIVALENCE, BLOCK
    DATA) or a state feature hit. Then its state statements, its type/DIMENSION/PARAMETER
    declarations (they give the shapes) and its header are kept. Comments and executable code
    are dropped, which keeps the prelude prompt small for very large modules.
    """
    hit_ranges = [
        (h.line, max(h.end_line, h.line)) for h in module.hits if h.feature_id in _STATE_FEATURE_SET
    ]
    result: list[tuple[int, str]] = []
    for unit in module.units:
        lines = unit.source.rstrip("\n").split("\n")
        statements = _statements(lines)
        header: tuple[list[int], str] | None = None
        kept: list[tuple[list[int], str]] = []
        anchored = False
        for statement in statements:
            indexes, text = statement
            code = _squash(text)
            first, last = unit.start_line + indexes[0], unit.start_line + indexes[-1]
            on_hit = any(lo <= last and first <= hi for lo, hi in hit_ranges)
            anchored = anchored or on_hit or _is_state_statement(code)
            if statement is statements[0] and _is_header(code):
                header = statement
            elif on_hit or _is_state_statement(code) or _is_declaration(code):
                kept.append(statement)
        if not anchored or not (kept or header):
            continue
        for indexes, _ in ([header] if header else []) + kept:
            result.extend((unit.start_line + i, lines[i]) for i in indexes)
    return result
