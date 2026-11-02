"""Statement-level parsing: turns a classified :class:`Statement` into a typed ``info`` record.

Every parse function works on the normalised statement text (keywords already separated, see
``source.py``). Anything that cannot be understood marks the statement ``invalid`` instead of
raising, so a damaged statement never takes the rest of the module down with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .exprs import (
    Expr,
    ExprError,
    Kw,
    Name,
    Ref,
    Star,
    parse_args,
    parse_expr,
    parse_expr_list,
    split_top,
)
from .source import Statement, mask_strings

_IDENT = r"[A-Za-z_][A-Za-z0-9_$]*"
_TYPE_WORD = re.compile(
    r"\s*(DOUBLE\s*PRECISION|DOUBLE\s*COMPLEX|INTEGER|REAL|COMPLEX|LOGICAL|CHARACTER|BYTE)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class TypeSpec:
    base: str  # integer real double doublecomplex complex logical character byte
    star: str | None = None  # "8" for REAL*8, "(*)" for CHARACTER*(*), "(LEN=8)" ...
    text: str = ""

    @property
    def api_type(self) -> str:
        base = self.base
        if base == "derived":
            return "unknown"
        if base == "real" and self.star is not None:
            digits = re.sub(r"\D", "", self.star)
            if digits in {"8", "16"}:
                return "double"
        if base in {"doublecomplex", "complex"}:
            return "complex"
        if base == "byte":
            return "integer"
        return base


@dataclass(slots=True)
class Dim:
    lower: Expr | None  # None means the default lower bound 1
    upper: Expr | None  # None means assumed size ('*')


@dataclass(slots=True)
class Entity:
    name: str
    dims: tuple[Dim, ...] = ()
    char_len: str | None = None


@dataclass(slots=True)
class Header:
    kind: str
    name: str
    args: list[str] = field(default_factory=list)  # '*' marks an alternate-return dummy
    type_spec: TypeSpec | None = None
    result: str | None = None


@dataclass(slots=True)
class Decl:
    spec: TypeSpec
    entities: list[Entity]
    attrs: frozenset[str] = frozenset()
    attr_dims: tuple[Dim, ...] = ()  # from a DIMENSION(...) attribute (Fortran 90 style)


@dataclass(slots=True)
class Implicit:
    none: bool
    rules: list[tuple[TypeSpec, list[tuple[str, str]]]] = field(default_factory=list)


@dataclass(slots=True)
class Dimension:
    entities: list[Entity]


@dataclass(slots=True)
class Common:
    blocks: list[tuple[str, list[Entity]]]


@dataclass(slots=True)
class Equivalence:
    sets: list[list[Expr]]


@dataclass(slots=True)
class Data:
    targets: list[Expr]
    values: str


@dataclass(slots=True)
class Parameter:
    items: list[tuple[str, Expr]]


@dataclass(slots=True)
class NameList:
    names: list[str]
    blocks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Assign:
    target: Expr
    value: Expr


@dataclass(slots=True)
class Call:
    name: str
    args: tuple[Expr, ...]


@dataclass(slots=True)
class Condition:
    cond: Expr
    labels: tuple[str, ...] = ()


@dataclass(slots=True)
class Do:
    label: str | None
    var: str | None
    lo: Expr | None = None
    hi: Expr | None = None
    step: Expr | None = None
    cond: Expr | None = None  # DO WHILE


@dataclass(slots=True)
class Goto:
    labels: tuple[str, ...] = ()
    expr: Expr | None = None
    var: str | None = None


@dataclass(slots=True)
class AssignLabel:
    label: str
    var: str


@dataclass(slots=True)
class Return:
    expr: Expr | None


@dataclass(slots=True)
class Io:
    op: str  # read write print open close inquire rewind backspace endfile
    unit: Expr | None
    fmt: Expr | None  # Star() for list-directed, None when absent
    controls: dict[str, Expr]
    items: list[Expr]


@dataclass(slots=True)
class Entry:
    name: str
    args: list[str]


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def match_paren(text: str, start: int) -> int:
    """Index of the ``)`` matching the ``(`` at ``text[start]`` (strings are skipped), or -1."""
    depth = 0
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_type_prefix(text: str) -> tuple[TypeSpec, str] | None:
    """Split a leading type specifier off ``text``; returns (spec, remaining text)."""
    derived = re.match(r"\s*(?:TYPE|CLASS)\s*\(", text, re.IGNORECASE)
    if derived:
        close = match_paren(text, derived.end() - 1)
        if close < 0:
            return None
        return TypeSpec("derived", None, text[: close + 1].strip()), text[close + 1 :].lstrip(" ,")
    match = _TYPE_WORD.match(text)
    if not match:
        return None
    word = re.sub(r"\s+", "", match.group(1)).upper()
    base = {"DOUBLEPRECISION": "double", "DOUBLECOMPLEX": "doublecomplex"}.get(word, word.lower())
    rest = text[match.end() :].lstrip()
    star: str | None = None
    if rest.startswith("*"):
        rest = rest[1:].lstrip()
        if rest.startswith("("):
            close = match_paren(rest, 0)
            if close < 0:
                return None
            star, rest = rest[: close + 1], rest[close + 1 :]
        else:
            digits = re.match(r"\d+", rest)
            if not digits:
                return None
            star, rest = digits.group(), rest[digits.end() :]
    elif rest.startswith("(") and base not in {"double", "doublecomplex"}:
        close = match_paren(rest, 0)
        inner = rest[1:close] if close > 0 else ""
        if close > 0 and re.match(r"\s*(?:KIND|LEN)\s*=|\s*\d|\s*\*", inner, re.IGNORECASE):
            star, rest = rest[: close + 1], rest[close + 1 :]
            kind = re.match(r"\s*(?:KIND\s*=)?\s*(\d+)\s*$", inner, re.IGNORECASE)
            if base != "character" and kind:
                star = kind.group(1)
    return TypeSpec(base, star, text[: len(text) - len(rest)].strip()), rest.lstrip()


def _bound(text: str) -> Expr | None:
    text = text.strip()
    return None if not text or text == "*" else parse_expr(text)


def parse_dims(inner: str) -> tuple[Dim, ...]:
    dims: list[Dim] = []
    for part in split_top(inner):
        pieces = split_top(part, ":")
        if len(pieces) == 2:
            dims.append(Dim(_bound(pieces[0]), _bound(pieces[1])))
        else:
            dims.append(Dim(None, _bound(part)))
    return tuple(dims)


def parse_entities(text: str) -> list[Entity]:
    """Declarators: ``A``, ``A(10,N)``, ``S*8``, ``S(3)*8`` (F90 initialisers are skipped)."""
    entities: list[Entity] = []
    for part in split_top(text):
        part = part.strip()
        if not part:
            continue
        match = re.match(rf"({_IDENT})\s*", part)
        if not match:
            raise ExprError(f"bad declarator {part!r}")
        name = match.group(1).lower()
        rest = part[match.end() :]
        dims: tuple[Dim, ...] = ()
        char_len = None
        if rest.startswith("("):
            close = match_paren(rest, 0)
            if close < 0:
                raise ExprError("unbalanced declarator")
            dims = parse_dims(rest[1:close])
            rest = rest[close + 1 :].lstrip()
        if rest.startswith("*"):
            length = re.match(r"\*\s*(\d+|\([^)]*\))", rest)
            if length:
                char_len = length.group(1)
        entities.append(Entity(name, dims, char_len))
    return entities


def _rest_after(text: str, pattern: str) -> str:
    match = re.match(pattern, text, re.IGNORECASE)
    if not match:
        raise ExprError("statement head not recognised")
    return text[match.end() :]


def _arg_names(text: str) -> list[str]:
    names = []
    for part in split_top(text):
        part = part.strip()
        if part:
            names.append("*" if part == "*" else part.lower())
    return names


def _split_assignment(text: str) -> tuple[str, str]:
    masked = mask_strings(text)
    depth = 0
    for i, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "=" and depth == 0:
            return text[:i], text[i + 1 :]
    raise ExprError("no assignment operator")


def _cond_and_tail(text: str) -> tuple[Expr, str]:
    start = text.index("(")
    close = match_paren(text, start)
    if close < 0:
        raise ExprError("unbalanced condition")
    return parse_expr(text[start + 1 : close]), text[close + 1 :].strip()


# --------------------------------------------------------------------------------------------
# Per-kind parsers
# --------------------------------------------------------------------------------------------


def _parse_header(kind: str, text: str) -> Header:
    spec = None
    body = text
    if kind == "function":
        prefix = parse_type_prefix(text)
        if prefix is not None:
            spec, body = prefix
        body = _rest_after(body, r"\s*FUNCTION\s+")
    elif kind == "subroutine":
        body = _rest_after(text, r"\s*SUBROUTINE\s+")
    elif kind == "program":
        body = _rest_after(text, r"\s*PROGRAM\s+")
    else:
        body = _rest_after(text, r"\s*BLOCK\s*DATA\s*")
    name_match = re.match(rf"\s*({_IDENT})?\s*(\*\s*\d+)?\s*", body)
    name = (name_match.group(1) or "").lower() if name_match else ""
    rest = body[name_match.end() :] if name_match else body
    args: list[str] = []
    if rest.startswith("("):
        close = match_paren(rest, 0)
        if close < 0:
            raise ExprError("unbalanced dummy argument list")
        args = _arg_names(rest[1:close])
        rest = rest[close + 1 :]
    result = re.search(rf"RESULT\s*\(\s*({_IDENT})\s*\)", rest, re.IGNORECASE)
    return Header(kind, name, args, spec, result.group(1).lower() if result else None)


def _parse_declaration(text: str) -> Decl:
    prefix = parse_type_prefix(text)
    if prefix is None:
        raise ExprError("bad type declaration")
    spec, rest = prefix
    attrs: set[str] = set()
    attr_dims: tuple[Dim, ...] = ()
    masked = mask_strings(rest)
    if "::" in masked:
        head, tail = rest[: masked.index("::")], rest[masked.index("::") + 2 :]
        for part in split_top(head.lstrip(", ")):
            word = re.match(r"\s*([A-Za-z]+)\s*(\(.*\))?\s*$", part)
            if word:
                attrs.add(word.group(1).lower())
                if word.group(1).lower() == "dimension" and word.group(2):
                    attr_dims = parse_dims(word.group(2)[1:-1])
        rest = tail
    return Decl(spec, parse_entities(rest), frozenset(attrs), attr_dims)


def _parse_implicit(text: str) -> Implicit:
    rest = _rest_after(text, r"\s*IMPLICIT\s*").strip()
    if rest.upper().startswith(("NONE", "UNDEFINED")):
        return Implicit(True)
    rules: list[tuple[TypeSpec, list[tuple[str, str]]]] = []
    while rest:
        prefix = parse_type_prefix(rest)
        if prefix is None or not prefix[1].startswith("("):
            raise ExprError("bad IMPLICIT statement")
        spec, after = prefix
        close = match_paren(after, 0)
        if close < 0:
            raise ExprError("bad IMPLICIT range")
        ranges = []
        for part in split_top(after[1:close]):
            letters = part.strip().lower()
            if len(letters) == 1:
                ranges.append((letters, letters))
            elif len(letters) == 3 and letters[1] == "-":
                ranges.append((letters[0], letters[2]))
            else:
                raise ExprError("bad IMPLICIT range")
        rules.append((spec, ranges))
        rest = after[close + 1 :].lstrip(" ,")
    return Implicit(False, rules)


def _parse_common(text: str) -> Common:
    rest = _rest_after(text, r"\s*COMMON\s*")
    pieces = re.split(r"(/[^/]*/)", rest)
    blocks: list[tuple[str, list[Entity]]] = []
    name = ""
    for piece in pieces:
        if piece.startswith("/") and piece.endswith("/") and len(piece) >= 2:
            name = piece[1:-1].strip().lower()
        elif piece.strip(" ,"):
            blocks.append((name, parse_entities(piece.strip(" ,"))))
    return Common(blocks)


def _parse_equivalence(text: str) -> Equivalence:
    rest = _rest_after(text, r"\s*EQUIVALENCE\s*")
    sets: list[list[Expr]] = []
    for part in split_top(rest):
        part = part.strip()
        if not (part.startswith("(") and part.endswith(")")):
            raise ExprError("bad EQUIVALENCE set")
        sets.append(parse_expr_list(part[1:-1]))
    return Equivalence(sets)


def _parse_data(text: str) -> Data:
    rest = _rest_after(text, r"\s*DATA\s*")
    parts = split_top(rest, "/")
    targets: list[Expr] = []
    values: list[str] = []
    for k, part in enumerate(parts):
        if k % 2 == 0:
            cleaned = part.strip(" ,")
            if cleaned:
                targets.extend(parse_expr_list(cleaned))
        else:
            values.append(part)
    return Data(targets, "/".join(values))


def _parse_parameter(text: str) -> Parameter:
    rest = _rest_after(text, r"\s*PARAMETER\s*").strip()
    if not (rest.startswith("(") and rest.endswith(")")):
        raise ExprError("bad PARAMETER statement")
    items = []
    for part in split_top(rest[1:-1]):
        lhs, rhs = _split_assignment(part)
        items.append((lhs.strip().lower(), parse_expr(rhs)))
    return Parameter(items)


def _parse_name_list(text: str, keyword: str) -> NameList:
    rest = _rest_after(text, rf"\s*{keyword}\s*").strip()
    names: list[str] = []
    blocks: list[str] = []
    for part in split_top(rest.lstrip(":")):
        part = part.strip()
        if part.startswith("/") and part.endswith("/") and len(part) >= 2:
            blocks.append(part[1:-1].strip().lower())
        elif part:
            names.append(part.lower())
    return NameList(names, blocks)


def _parse_io(kind: str, text: str) -> Io:
    match = re.match(
        r"\s*(READ|WRITE|PRINT|ACCEPT|TYPE|OPEN|CLOSE|INQUIRE|REWIND|BACKSPACE|ENDFILE)\s*",
        text,
        re.IGNORECASE,
    )
    if not match:
        raise ExprError("bad I/O statement")
    op = match.group(1).lower()
    op = {"accept": "read", "type": "print"}.get(op, op)
    rest = text[match.end() :]
    unit: Expr | None = None
    fmt: Expr | None = None
    controls: dict[str, Expr] = {}
    if rest.startswith("("):
        close = match_paren(rest, 0)
        if close < 0:
            raise ExprError("unbalanced I/O control list")
        positional: list[Expr] = []
        for arg in parse_args(rest[1:close]):
            if isinstance(arg, Kw):
                controls[arg.key] = arg.value
            else:
                positional.append(arg)
        unit = controls.get("unit", positional[0] if positional else None)
        if op in {"read", "write"}:
            index = 0 if "unit" in controls else 1
            fmt = controls.get("fmt", positional[index] if len(positional) > index else None)
        items_text = rest[close + 1 :].strip().lstrip(",")
    elif op in {"print", "read"}:
        parts = split_top(rest, ",")
        head = parts[0].strip()
        fmt = Star() if head == "*" else parse_expr(head)
        items_text = ",".join(parts[1:])
    else:
        unit = parse_expr(rest.strip())
        items_text = ""
    items = parse_expr_list(items_text)
    return Io(op, unit, fmt, controls, items)


def _label(text: str | None) -> str | None:
    return None if text is None else (text.lstrip("0") or "0")


def _parse_do(text: str) -> Do:
    while_match = re.match(r"\s*DO\s*(\d+)?\s*,?\s*WHILE\s*(\(.*)$", text, re.IGNORECASE)
    if while_match:
        cond, _ = _cond_and_tail(while_match.group(2))
        return Do(_label(while_match.group(1)), None, cond=cond)
    match = re.match(rf"\s*DO\s*(\d+)?\s*,?\s*({_IDENT})\s*=\s*(.*)$", text, re.IGNORECASE)
    if not match:
        if re.fullmatch(r"\s*DO\s*", text, re.IGNORECASE):
            return Do(None, None)
        raise ExprError("bad DO statement")
    parts = split_top(match.group(3))
    if len(parts) not in {2, 3}:
        raise ExprError("bad DO bounds")
    step = parse_expr(parts[2]) if len(parts) == 3 else None
    return Do(
        _label(match.group(1)),
        match.group(2).lower(),
        parse_expr(parts[0]),
        parse_expr(parts[1]),
        step,
    )


def _parse_goto(kind: str, text: str) -> Goto:
    rest = _rest_after(text, r"\s*GO\s*TO\s*").strip()
    if kind == "goto":
        return Goto((rest.split()[0].lstrip("0") or "0",))
    if kind == "goto_computed":
        close = match_paren(rest, 0)
        if close < 0:
            raise ExprError("unbalanced computed GOTO")
        labels = tuple(p.strip().lstrip("0") or "0" for p in split_top(rest[1:close]))
        selector = rest[close + 1 :].strip().lstrip(",")
        return Goto(labels, parse_expr(selector))
    name = re.match(_IDENT, rest)
    if not name:
        raise ExprError("bad assigned GOTO")
    labels: tuple[str, ...] = ()
    tail = rest[name.end() :].strip().lstrip(",").strip()
    if tail.startswith("("):
        close = match_paren(tail, 0)
        labels = tuple(p.strip() for p in split_top(tail[1:close]))
    return Goto(labels, None, name.group().lower())


def parse_statement(stmt: Statement) -> None:
    """Fill ``stmt.info`` (and ``stmt.inner.info``); sets ``stmt.invalid`` on failure."""
    try:
        stmt.info = _parse(stmt)
        if stmt.inner is not None:
            stmt.inner.info = _parse(stmt.inner)
            if stmt.inner.kind in {"unknown", "if_then", "else_if", "else", "end_if"}:
                stmt.invalid = True
    except (ExprError, ValueError, IndexError, AttributeError, RecursionError):
        stmt.invalid = True
        stmt.info = None
    if stmt.kind == "unknown":
        stmt.invalid = True


def _parse(stmt: Statement) -> object | None:
    kind, text = stmt.kind, stmt.text
    if kind in {"subroutine", "function", "program", "block_data"}:
        return _parse_header(kind, text)
    if kind == "assign":
        lhs, rhs = _split_assignment(text)
        target = parse_expr(lhs)
        if not isinstance(target, Name | Ref):
            raise ExprError("bad assignment target")
        return Assign(target, parse_expr(rhs))
    if kind == "call":
        match = re.match(rf"\s*CALL\s+({_IDENT})\s*(.*)$", text, re.IGNORECASE)
        if not match:
            raise ExprError("bad CALL")
        tail = match.group(2).strip()
        if not tail:
            return Call(match.group(1).lower(), ())
        if not (tail.startswith("(") and tail.endswith(")")):
            raise ExprError("bad CALL arguments")
        return Call(match.group(1).lower(), parse_args(tail[1:-1]))
    if kind in {"if_then", "else_if"}:
        return Condition(_cond_and_tail(text)[0])
    if kind == "if_logical":
        return Condition(_cond_and_tail(text)[0])
    if kind == "if_arith":
        cond, tail = _cond_and_tail(text)
        return Condition(cond, tuple(p.strip().lstrip("0") or "0" for p in tail.split(",")))
    if kind in {"do", "do_while"}:
        return _parse_do(text)
    if kind in {"goto", "goto_computed", "goto_assigned"}:
        return _parse_goto(kind, text)
    if kind == "assign_label":
        match = re.match(rf"\s*ASSIGN\s+(\d+)\s+TO\s+({_IDENT})", text, re.IGNORECASE)
        if not match:
            raise ExprError("bad ASSIGN")
        return AssignLabel(match.group(1).lstrip("0") or "0", match.group(2).lower())
    if kind == "return":
        rest = _rest_after(text, r"\s*RETURN\s*").strip()
        return Return(parse_expr(rest) if rest else None)
    if kind == "declaration":
        return _parse_declaration(text)
    if kind == "implicit":
        return _parse_implicit(text)
    if kind == "dimension":
        return Dimension(parse_entities(_rest_after(text, r"\s*DIMENSION\s*")))
    if kind == "common":
        return _parse_common(text)
    if kind == "equivalence":
        return _parse_equivalence(text)
    if kind == "data":
        return _parse_data(text)
    if kind == "parameter":
        return _parse_parameter(text)
    if kind in {"external", "intrinsic"}:
        return _parse_name_list(text, kind.upper())
    if kind == "save":
        return _parse_name_list(text, "SAVE")
    if kind in {"read", "write", "print", "io_file"}:
        return _parse_io(kind, text)
    if kind == "entry":
        header = _parse_header("subroutine", re.sub(r"(?i)^\s*ENTRY", "SUBROUTINE", text))
        return Entry(header.name, header.args)
    return None
