"""Expression parsing for the statement-level analysis.

A small recursive-descent parser for Fortran 77 expressions with the operator precedence of the
standard (``**`` > ``* /`` > ``+ -`` > ``//`` > relational > ``.NOT.`` > ``.AND.`` > ``.OR.`` >
``.EQV.``). ``name(args)`` is parsed as a :class:`Ref` because only the symbol table can tell an
array element from a function call. Implied-DO lists, complex literals, substring ranges,
alternate-return actuals (``*10``) and keyword arguments (``END=10``) are understood.

Parse failures raise :class:`ExprError`; callers mark the statement invalid instead of crashing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

DOTTED_WORDS = "EQ|NE|LT|LE|GT|GE|AND|OR|NOT|EQV|NEQV|XOR|TRUE|FALSE"
_TOKEN = re.compile(
    rf"""
    (?P<ws>\s+)
  | (?P<dot>\.(?:{DOTTED_WORDS})\.)
  | (?P<num>(?:\d+(?:\.(?!(?:{DOTTED_WORDS})\.)\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<str>'(?:[^']|'')*'|"(?:[^"]|"")*")
  | (?P<op>\*\*|//|==|/=|<=|>=|[-+*/(),:=<>%&])
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)
_RELATIONAL = {".EQ.", ".NE.", ".LT.", ".LE.", ".GT.", ".GE.", "==", "/=", "<", "<=", ">", ">="}


class ExprError(ValueError):
    """The text is not a Fortran expression this parser understands."""


@dataclass(frozen=True, slots=True)
class Name:
    name: str  # lower case


@dataclass(frozen=True, slots=True)
class Num:
    text: str
    kind: str  # "integer" | "real" | "double"


@dataclass(frozen=True, slots=True)
class Str:
    value: str


@dataclass(frozen=True, slots=True)
class Logical:
    value: bool


@dataclass(frozen=True, slots=True)
class Ref:
    """``name(args)``: an array element, a function reference or a substring."""

    name: str
    args: tuple[Expr, ...]
    sub: tuple[Expr, ...] | None = None  # second group: A(I)(2:3)


@dataclass(frozen=True, slots=True)
class Bin:
    op: str  # upper case; dotted operators keep their dots
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Un:
    op: str
    operand: Expr


@dataclass(frozen=True, slots=True)
class Paren:
    inner: Expr


@dataclass(frozen=True, slots=True)
class Cplx:
    real: Expr
    imag: Expr


@dataclass(frozen=True, slots=True)
class Range:
    lo: Expr | None
    hi: Expr | None


@dataclass(frozen=True, slots=True)
class Star:
    """A bare ``*`` (list-directed unit/format) or ``*label`` (alternate-return actual)."""

    label: str = ""


@dataclass(frozen=True, slots=True)
class Kw:
    key: str  # lower case
    value: Expr


@dataclass(frozen=True, slots=True)
class ImpliedDo:
    items: tuple[Expr, ...]
    var: str
    lo: Expr
    hi: Expr
    step: Expr | None = None


Expr = Name | Num | Str | Logical | Ref | Bin | Un | Paren | Cplx | Range | Star | Kw | ImpliedDo


def children(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, Ref):
        return expr.args + (expr.sub or ())
    if isinstance(expr, Bin):
        return (expr.left, expr.right)
    if isinstance(expr, Un):
        return (expr.operand,)
    if isinstance(expr, Paren):
        return (expr.inner,)
    if isinstance(expr, Cplx):
        return (expr.real, expr.imag)
    if isinstance(expr, Range):
        return tuple(e for e in (expr.lo, expr.hi) if e is not None)
    if isinstance(expr, Kw):
        return (expr.value,)
    if isinstance(expr, ImpliedDo):
        extra = (expr.lo, expr.hi) + ((expr.step,) if expr.step is not None else ())
        return expr.items + extra
    return ()


def walk(expr: Expr) -> Iterator[Expr]:
    """Pre-order traversal (iterative, so deep nesting cannot hit the recursion limit)."""
    stack = [expr]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(children(node)))


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        match = _TOKEN.match(text, i)
        if not match:
            raise ExprError(f"unexpected character {text[i]!r}")
        i = match.end()
        kind = match.lastgroup or ""
        if kind != "ws":
            tokens.append((kind, match.group()))
    return tokens


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split ``text`` on ``sep`` outside parentheses and strings."""
    parts: list[str] = []
    depth = 0
    quote = ""
    start = 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _number_kind(text: str) -> str:
    if re.fullmatch("[0-9]+", text):
        return "integer"
    return "double" if re.search(r"[dD][+-]?\d+$", text) else "real"


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.i = 0
        self.depth = 0

    # -- token helpers --------------------------------------------------------------------
    def peek(self, offset: int = 0) -> tuple[str, str]:
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else ("end", "")

    def at(self, value: str) -> bool:
        kind, text = self.peek()
        return kind in {"op", "dot"} and text.upper() == value

    def take(self) -> tuple[str, str]:
        token = self.peek()
        self.i += 1
        return token

    def expect(self, value: str) -> None:
        if not self.at(value):
            raise ExprError(f"expected {value!r}, found {self.peek()[1]!r}")
        self.i += 1

    def done(self) -> bool:
        return self.i >= len(self.tokens)

    # -- grammar --------------------------------------------------------------------------
    def expr(self) -> Expr:
        self.depth += 1
        if self.depth > 200:
            raise ExprError("expression nested too deeply")
        try:
            left = self._or()
            while self.peek()[0] == "dot" and self.peek()[1].upper() in {
                ".EQV.",
                ".NEQV.",
                ".XOR.",
            }:
                op = self.take()[1].upper()
                left = Bin(op, left, self._or())
            return left
        finally:
            self.depth -= 1

    def _or(self) -> Expr:
        left = self._and()
        while self.at(".OR."):
            self.take()
            left = Bin(".OR.", left, self._and())
        return left

    def _and(self) -> Expr:
        left = self._not()
        while self.at(".AND."):
            self.take()
            left = Bin(".AND.", left, self._not())
        return left

    def _not(self) -> Expr:
        if self.at(".NOT."):
            self.take()
            return Un(".NOT.", self._not())
        return self._relational()

    def _relational(self) -> Expr:
        left = self._concat()
        kind, text = self.peek()
        if kind in {"op", "dot"} and text.upper() in _RELATIONAL:
            self.take()
            return Bin(text.upper(), left, self._concat())
        return left

    def _concat(self) -> Expr:
        left = self._add()
        while self.at("//"):
            self.take()
            left = Bin("//", left, self._add())
        return left

    def _add(self) -> Expr:
        if self.at("+") or self.at("-"):
            op = self.take()[1]
            left: Expr = Un(op, self._term())
        else:
            left = self._term()
        while self.at("+") or self.at("-"):
            op = self.take()[1]
            left = Bin(op, left, self._term())
        return left

    def _term(self) -> Expr:
        left = self._factor()
        while self.at("*") or self.at("/"):
            op = self.take()[1]
            left = Bin(op, left, self._factor())
        return left

    def _factor(self) -> Expr:
        if self.at("+") or self.at("-"):
            op = self.take()[1]
            return Un(op, self._factor())
        base = self._primary()
        if self.at("**"):
            self.take()
            return Bin("**", base, self._factor())
        return base

    def _primary(self) -> Expr:
        kind, text = self.take()
        if kind == "num":
            return Num(text, _number_kind(text))
        if kind == "str":
            return Str(text[1:-1].replace(text[0] * 2, text[0]))
        if kind == "dot" and text.upper() in {".TRUE.", ".FALSE."}:
            return Logical(text.upper() == ".TRUE.")
        if kind == "name":
            name = text.lower()
            groups = []
            while True:
                while self.at("("):
                    groups.append(self.arguments())
                if not self.at("%"):
                    break
                self.take()  # derived-type member: the base variable stands for the whole
                if self.peek()[0] != "name":
                    raise ExprError("expected a component name after '%'")
                self.take()
            if not groups:
                return Name(name)
            return Ref(name, groups[0], groups[1] if len(groups) > 1 else None)
        if kind == "op" and text == "(":
            return self._parenthesised()
        if kind == "op" and text == "%" and self.peek()[0] == "name":
            self.take()  # vendor %VAL(x) / %LOC(x) / %REF(x): the argument stands for the whole
            args = self.arguments()
            return Paren(args[0]) if len(args) == 1 else Paren(Name("%descr"))
        raise ExprError(f"unexpected token {text!r}")

    def _parenthesised(self) -> Expr:
        items = [self.expr()]
        control: tuple[str, Expr, Expr, Expr | None] | None = None
        while self.at(","):
            self.take()
            if self.peek()[0] == "name" and self.peek(1) == ("op", "="):
                control = self._do_control()
                break
            items.append(self.expr())
        self.expect(")")
        if control is not None:
            var, lo, hi, step = control
            return ImpliedDo(tuple(items), var, lo, hi, step)
        if len(items) == 1:
            return Paren(items[0])
        if len(items) == 2 and all(_is_real_constant(e) for e in items):
            return Cplx(items[0], items[1])
        raise ExprError("parenthesised list is not an expression")

    def _do_control(self) -> tuple[str, Expr, Expr, Expr | None]:
        var = self.take()[1].lower()
        self.expect("=")
        lo = self.expr()
        self.expect(",")
        hi = self.expr()
        step = None
        if self.at(","):
            self.take()
            step = self.expr()
        return var, lo, hi, step

    def arguments(self) -> tuple[Expr, ...]:
        self.expect("(")
        args: list[Expr] = []
        if self.at(")"):
            self.take()
            return ()
        while True:
            args.append(self._argument())
            if self.at(","):
                self.take()
                continue
            self.expect(")")
            return tuple(args)

    def _argument(self) -> Expr:
        kind, text = self.peek()
        if kind == "op" and text == "*":
            self.take()
            nxt = self.peek()
            if nxt[0] == "num":
                self.take()
                return Star(nxt[1])
            return Star()
        if kind == "name" and self.peek(1) == ("op", "="):
            self.take()
            self.take()
            return Kw(text.lower(), self.expr() if not self.at("*") else self._argument())
        if self.at(":"):
            self.take()
            hi = None if self.at(",") or self.at(")") else self.expr()
            return Range(None, hi)
        first = self.expr()
        if self.at(":"):
            self.take()
            hi = None if self.at(",") or self.at(")") else self.expr()
            return Range(first, hi)
        return first


def _is_real_constant(expr: Expr) -> bool:
    if isinstance(expr, Un):
        expr = expr.operand
    return isinstance(expr, Num)


def parse_expr(text: str) -> Expr:
    parser = _Parser(tokenize(text))
    result = parser.expr()
    if not parser.done():
        raise ExprError(f"trailing text after expression: {parser.peek()[1]!r}")
    return result


def parse_expr_list(text: str) -> list[Expr]:
    """Comma separated expressions / implied-DO items (an I/O list). Empty text gives ``[]``."""
    if not text.strip():
        return []
    parser = _Parser(tokenize(text))
    items = [parser.expr()]
    while parser.at(","):
        parser.take()
        items.append(parser.expr())
    if not parser.done():
        raise ExprError(f"trailing text after list: {parser.peek()[1]!r}")
    return items


def parse_args(text: str) -> tuple[Expr, ...]:
    """Parse the inside of a parenthesised argument/control list."""
    parser = _Parser(tokenize("(" + text + ")"))
    args = parser.arguments()
    if not parser.done():
        raise ExprError("trailing text after argument list")
    return args


def target_names(expr: Expr) -> list[str]:
    """Names of the variables a target (or I/O item) writes: scalars, array elements, implied-DO bodies."""
    names: list[str] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Name | Ref):
            names.append(node.name)
        elif isinstance(node, ImpliedDo):
            stack.extend(node.items)
        elif isinstance(node, Paren):
            stack.append(node.inner)
    return names
