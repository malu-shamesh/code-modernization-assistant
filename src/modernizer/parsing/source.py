"""Fortran source reading: fixed form, free form, Hollerith constants, blank-insensitive keywords.

Real FORTRAN 77 is column based. This module turns raw text into *logical statements*: one
statement per entry with its label, the original line span and a normalised text in which

* comments, sequence numbers and continuation markers are gone,
* Hollerith constants (``5HHELLO``) are ordinary quoted strings,
* keywords are separated from what follows (``GOTO10`` -> ``GOTO 10``, ``DO10I=1,N`` ->
  ``DO 10 I=1,N``, ``IF(X.GT.0)GOTO10`` -> ``IF(X.GT.0) GOTO 10``).

Each statement is also classified (``kind``). Classification is what makes the keyword splitter
safe: ``DO10I=1.5`` is an assignment while ``DO10I=1,5`` is a loop, and ``REALX`` is a declaration
but ``REALX=1`` is an assignment, exactly as the blank-insensitive F77 rules say.

Handled blank-insensitive forms: every statement-leading keyword (type names, DO, IF, GOTO, CALL,
READ/WRITE/PRINT, ASSIGN, DATA, COMMON, ...) and the closers ``ENDIF``/``ENDDO``/``ELSEIF``.
NOT handled: blanks inside names or numbers (``AB C = 1``), keywords split by blanks in the
middle of operands (``X .GT.0`` is fine, ``X. GT. 0`` is not), and Hollerith counts separated
from the ``H`` by blanks.

``tree_text`` is a rewritten twin of each statement for tree-sitter-fortran, which does not know
a few legacy forms (alternate-return ``*`` dummies and actuals, ``COMMON //``, plain assigned
``GOTO k``, ``TYPE *``); the engine itself always works on ``text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FIXED = "fixed"
FREE = "free"

FIXED_WIDTH = 72
_FIELD_WIDTH = FIXED_WIDTH - 6
_LINE_SPLIT = re.compile(r"\r\n|\r|\n")
_NAME = re.compile(r"[A-Z_][A-Z0-9_$]*")
_SEQUENCE_TAIL = re.compile(r"\s*(?:\d+|[A-Za-z]{1,6}\d{3,})?\s*")
_LABEL_PREFIX = re.compile(r"\s*(\d{1,5})(?:\s+|(?=[A-Za-z(]))")
_DO_HEAD = re.compile(r"DO(\d*),?([A-Z_][A-Z0-9_$]*)=")
_DO_WHILE = re.compile(r"DO(\d*),?WHILE\(")
_ARITH_IF_TAIL = re.compile(r"\d+,\d+,\d+")
_ASSIGN_LABEL = re.compile(r"ASSIGN(\d+)TO([A-Z_][A-Z0-9_$]*)$")
_ENDS_WITH_DOTTED_OP = re.compile(r"\.[A-Za-z]{2,4}\.\s*$")

# Statement kinds -------------------------------------------------------------------------------
EXECUTABLE_KINDS = frozenset(
    {
        "assign",
        "call",
        "if_then",
        "else_if",
        "else",
        "end_if",
        "if_logical",
        "if_arith",
        "do",
        "do_while",
        "end_do",
        "continue",
        "goto",
        "goto_computed",
        "goto_assigned",
        "assign_label",
        "return",
        "stop",
        "pause",
        "read",
        "write",
        "print",
        "io_file",
        "exit_cycle",
    }
)
HEADER_KINDS = frozenset({"subroutine", "function", "program", "block_data"})

_TYPE_WORDS = (
    "DOUBLEPRECISION",
    "DOUBLECOMPLEX",
    "INTEGER",
    "REAL",
    "COMPLEX",
    "LOGICAL",
    "CHARACTER",
    "BYTE",
)

_SIMPLE_KEYWORDS: dict[str, str] = {
    "SUBROUTINE": "subroutine",
    "FUNCTION": "function",
    "PROGRAM": "program",
    "ENTRY": "entry",
    "COMMON": "common",
    "EQUIVALENCE": "equivalence",
    "DATA": "data",
    "DIMENSION": "dimension",
    "PARAMETER": "parameter",
    "EXTERNAL": "external",
    "INTRINSIC": "intrinsic",
    "SAVE": "save",
    "FORMAT": "format",
    "CALL": "call",
    "RETURN": "return",
    "STOP": "stop",
    "PAUSE": "pause",
    "CONTINUE": "continue",
    "READ": "read",
    "WRITE": "write",
    "PRINT": "print",
    "ACCEPT": "read",
    "OPEN": "io_file",
    "CLOSE": "io_file",
    "INQUIRE": "io_file",
    "REWIND": "io_file",
    "BACKSPACE": "io_file",
    "ENDFILE": "io_file",
    "INCLUDE": "include",
    "NAMELIST": "namelist",
    "IMPLICIT": "implicit",
    "EXIT": "exit_cycle",
    "CYCLE": "exit_cycle",
    "STRUCTURE": "vendor_block",
    "ENDSTRUCTURE": "vendor_block",
    "RECORD": "vendor_block",
    "UNION": "vendor_block",
    "ENDUNION": "vendor_block",
    "MAP": "vendor_block",
    "ENDMAP": "vendor_block",
    "ENCODE": "vendor_block",
    "DECODE": "vendor_block",
    "POINTER": "vendor_decl",
    "AUTOMATIC": "vendor_decl",
    "STATIC": "vendor_decl",
    "VOLATILE": "vendor_decl",
    "USE": "f90",
    "MODULE": "f90",
    "CONTAINS": "contains",
    "INTERFACE": "f90",
    "ENDINTERFACE": "f90",
    "ALLOCATE": "f90",
    "DEALLOCATE": "f90",
    "NULLIFY": "f90",
    "SELECTCASE": "f90",
    "CASE": "f90",
    "ENDSELECT": "f90",
    "WHERE": "f90",
    "ENDWHERE": "f90",
    "SEQUENCE": "f90",
    "PRIVATE": "f90",
    "PUBLIC": "f90",
    "PROTECTED": "f90",
    "PROCEDURE": "f90",
    "CLASS": "f90",
    "ABSTRACT": "f90",
    "IMPORT": "f90",
    "ENUM": "f90",
    "ENDENUM": "f90",
    "GENERIC": "f90",
    "FINAL": "f90",
    "ASSOCIATE": "f90",
    "ENDASSOCIATE": "f90",
    "CRITICAL": "f90",
    "ENDCRITICAL": "f90",
    "FORALL": "f90",
    "ENDFORALL": "f90",
    "ELSEWHERE": "f90",
    "SUBMODULE": "f90",
    "ENDSUBMODULE": "f90",
    "INTENT": "f90",
    "OPTIONAL": "f90",
    "TARGET": "f90",
    "ALLOCATABLE": "f90",
    "ASYNCHRONOUS": "f90",
    "VALUE": "f90",
    "BIND": "f90",
    "BLOCK": "f90",
    "ENDBLOCK": "f90",
    "FLUSH": "f90",
}
_KEYWORDS_BY_LENGTH = sorted(_SIMPLE_KEYWORDS, key=len, reverse=True)
_END_UNIT_WORDS = ("PROGRAM", "SUBROUTINE", "FUNCTION", "BLOCKDATA", "MODULE")

# Kinds the engine understands but tree-sitter-fortran cannot parse: kept out of the tree.
_TREE_COMMENTED_KINDS = frozenset({"vendor_block", "vendor_decl"})


@dataclass(frozen=True)
class NormalizedSource:
    """Free-form text for tree-sitter plus the map back to the original lines.

    ``text`` has exactly one line per logical statement. ``line_map[i]`` is the original 1-based
    line on which normalized line ``i`` (0-based) starts; ``end_line_map[i]`` is where it ends
    (they differ for continued statements).
    """

    text: str
    line_map: list[int]
    form: str
    end_line_map: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Statement:
    """One logical statement (or, for a logical IF, the whole IF with its ``inner`` statement)."""

    index: int
    label: str | None
    text: str
    raw: str
    kind: str
    line: int
    end_line: int
    masked: str
    tree_text: str = ""
    n_hollerith: int = 0
    inner: Statement | None = None
    info: object | None = None
    effects: object | None = None  # cached by effects.statement_effects
    invalid: bool = False

    @property
    def effective(self) -> Statement:
        """The statement that actually executes (the inner one of a logical IF)."""
        return self.inner if self.inner is not None else self


@dataclass(slots=True)
class SourceModel:
    form: str
    lines: list[str]
    statements: list[Statement]
    code_lines: frozenset[int]  # physical lines that carry code (statement or continuation)
    marker_lines: tuple[int, ...] = ()  # lines showing fixed-form syntax (labels, C comments ...)


@dataclass(slots=True)
class Classified:
    text: str
    kind: str
    inner: Classified | None = None
    tree_text: str = ""


# --------------------------------------------------------------------------------------------
# Lines and source form
# --------------------------------------------------------------------------------------------


def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def split_lines(source: object) -> list[str]:
    """Physical lines of ``source`` (never raises; non-text input gives no lines)."""
    if not isinstance(source, str):
        return []
    text = source.replace("\x00", " ")
    text = text.removeprefix("\ufeff")
    lines = _LINE_SPLIT.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _is_fixed_comment(line: str) -> bool:
    first = line[0]
    if first in "Cc*!#":
        return True
    # Debug lines ("D" in column 1 with a blank label/continuation field) are comments.
    return first in "Dd" and not line[1:6].strip()


def _quote_aware_comment_start(line: str) -> int:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return i
    return -1


def detect_source_form(source: str) -> str:
    """Guess ``"fixed"`` or ``"free"`` from the text alone.

    Free form is chosen when a line ends with a free-form continuation ``&`` or uses ``::``, or
    when at least 40% of the code lines have text in columns 1-5 (impossible in fixed form,
    where that field holds only a label). Everything else, including empty input, is fixed.
    """
    code = invalid = 0
    for line in split_lines(source):
        if not line.strip() or _is_fixed_comment(line):
            continue
        cut = _quote_aware_comment_start(line)
        body = line if cut < 0 else line[:cut]
        code += 1
        shown = mask_strings(body)
        if shown.rstrip().endswith("&") or "::" in shown:
            return FREE
        if line[0] == "\t":
            continue
        tab = line.find("\t")
        head = line[:tab] if 0 <= tab < 6 else line[:5]
        if head.strip(" 0123456789"):
            invalid += 1
    return FREE if code and invalid * 10 >= code * 4 else FIXED


# --------------------------------------------------------------------------------------------
# Physical line -> logical statement groups
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Group:
    label: str
    pieces: list[tuple[str, int]]  # (statement-field text, physical line number)


def _statement_field(line: str) -> str:
    text = line[6:FIXED_WIDTH]
    tail = line[FIXED_WIDTH:]
    # Text after column 72 is dropped when it looks like a sequence number, kept otherwise.
    return text if _SEQUENCE_TAIL.fullmatch(tail) else line[6:]


def _sloppy(line: str) -> tuple[str, str, str]:
    stripped = line.strip()
    match = _LABEL_PREFIX.match(stripped)
    if match:
        return "initial", match.group(1), stripped[match.end() :]
    return "initial", "", stripped


def _classify_fixed(line: str) -> tuple[str, str, str]:
    """Return (kind, label, body) for one physical line of fixed-form source."""
    if not line.strip():
        return "blank", "", ""
    if _is_fixed_comment(line):
        return "comment", "", ""
    tab = line.find("\t")
    if 0 <= tab < 6 and not line[:tab].strip(" 0123456789"):
        label, rest = line[:tab].strip(), line[tab + 1 :]
        if not label and rest[:1] in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            return "continuation", "", rest[1:]
        return "initial", label, rest
    head = line[:5]
    if head.strip(" 0123456789"):
        return _sloppy(line)
    marker = line[5] if len(line) > 5 else " "
    label = head.strip()
    if marker not in " 0":
        if label:
            return _sloppy(line)
        return "continuation", "", _statement_field(line)
    return "initial", label, _statement_field(line)


def _blank_or_comment(body: str) -> bool:
    stripped = body.strip()
    return not stripped or stripped.startswith("!")


def _read_fixed(lines: list[str]) -> tuple[list[_Group], list[int]]:
    groups: list[_Group] = []
    markers: list[int] = []
    current: _Group | None = None
    for number, line in enumerate(lines, 1):
        kind, label, body = _classify_fixed(line)
        if kind == "comment" and line[:1] in "CcDd*":
            markers.append(number)
        if kind in {"blank", "comment"}:
            continue
        if kind == "continuation" and current is not None:
            current.pieces.append((body, number))
            markers.append(number)
            continue
        if label or line[:1] == "\t":
            markers.append(number)
        if _blank_or_comment(body):
            continue
        current = _Group(label, [(body, number)])
        groups.append(current)
    return groups, markers


def _read_free(lines: list[str]) -> list[_Group]:
    groups: list[_Group] = []
    current: _Group | None = None
    open_continuation = False
    for number, line in enumerate(lines, 1):
        cut = _quote_aware_comment_start(line)
        code = (line if cut < 0 else line[:cut]).rstrip()
        if not code.strip():
            continue
        continues = code.endswith("&")
        if continues:
            code = code[:-1]
        if open_continuation and current is not None:
            lead = code.lstrip()
            current.pieces.append((lead[1:] if lead.startswith("&") else code, number))
        else:
            label = ""
            match = _LABEL_PREFIX.match(code)
            if match:
                if not code[match.end() :].strip():
                    continue
                label, code = match.group(1), code[match.end() :]
            current = _Group(label, [(code, number)])
            groups.append(current)
        open_continuation = continues
    return groups


# --------------------------------------------------------------------------------------------
# Statement text scanning: strings, Hollerith constants, inline comments
# --------------------------------------------------------------------------------------------


class _Scanner:
    """Walks the pieces of one logical statement, converting Hollerith constants to strings."""

    def __init__(self) -> None:
        self.out: list[str] = []
        self.raw: list[str] = []
        self.quote = ""
        self.hollerith_left = 0
        self.n_hollerith = 0
        self.last_significant = ""

    def _head(self) -> str:
        return "".join(self.out).replace(" ", "")[:4].upper()

    def _hollerith_allowed(self, i: int, piece: str) -> int:
        """Length of an ``nH`` prefix starting at ``piece[i]`` (0 when it is not Hollerith)."""
        j = i
        while j < len(piece) and _is_digit(piece[j]):
            j += 1
        if j == i or j - i > 4 or j >= len(piece) or piece[j] not in "hH":
            return 0
        previous = self.last_significant
        allowed = previous in {"(", ",", "/", "="} or (previous == "*" and self._head() == "DATA")
        if previous == ".":  # IF (C .EQ. 1HA): after a dotted operator, never after a decimal point
            allowed = bool(_ENDS_WITH_DOTTED_OP.search("".join(self.out)))
        if not allowed or (i > 0 and piece[i - 1].isalnum()):
            return 0
        return j - i

    def feed(self, piece: str, pad_to: int | None, last: bool) -> None:
        i, n = 0, len(piece)
        cut = n
        while i < n:
            ch = piece[i]
            if self.hollerith_left:
                take = piece[i : i + self.hollerith_left]
                self.out.append(take.replace("'", "''"))
                self.hollerith_left -= len(take)
                i += len(take)
                if not self.hollerith_left:
                    self.out.append("'")
                continue
            if self.quote:
                self.out.append(ch)
                if ch == self.quote:
                    if i + 1 < n and piece[i + 1] == self.quote:
                        self.out.append(ch)
                        i += 1
                    else:
                        self.quote = ""
                i += 1
                continue
            if ch in "'\"":
                self.quote = ch
                self.out.append(ch)
                self.last_significant = ch
            elif ch == "!":
                cut = i
                break
            elif _is_digit(ch):
                width = self._hollerith_allowed(i, piece)
                if width:
                    count = int(piece[i : i + width])
                    self.n_hollerith += 1
                    self.out.append("'")
                    self.hollerith_left = count
                    self.last_significant = "'"
                    i += width + 1
                    continue
                self.out.append(ch)
                self.last_significant = ch
            else:
                self.out.append(ch)
                if not ch.isspace():
                    self.last_significant = ch
            i += 1
        self.raw.append(piece[:cut])
        if not last and pad_to is not None and (self.quote or self.hollerith_left):
            # Fixed-form lines are blank-padded to the full width inside character context.
            pad = max(0, pad_to - len(piece))
            if self.quote:
                self.out.append(" " * pad)
            else:
                take = min(pad, self.hollerith_left)
                self.out.append(" " * take)
                self.hollerith_left -= take
                if not self.hollerith_left:
                    self.out.append("'")

    def finish(self) -> tuple[str, str]:
        if self.hollerith_left:
            self.out.append(" " * self.hollerith_left + "'")
            self.hollerith_left = 0
        raw = " ".join(p.strip() for p in self.raw if p.strip())
        return "".join(self.out), raw


def mask_strings(text: str, fill: str = " ") -> str:
    """``text`` with the contents of quoted strings replaced by ``fill`` (same length)."""
    out: list[str] = []
    quote = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < n and text[i + 1] == quote:
                    out.append(fill * 2)
                    i += 2
                    continue
                quote = ""
                out.append(ch)
            else:
                out.append(fill)
        else:
            if ch in "'\"":
                quote = ch
            out.append(ch)
        i += 1
    return "".join(out)


def _split_semicolons(code: str) -> list[str]:
    masked = mask_strings(code)
    if ";" not in masked:
        return [code]
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(masked):
        if ch == ";":
            parts.append(code[start:i])
            start = i + 1
    parts.append(code[start:])
    return [p for p in parts if p.strip()]


# --------------------------------------------------------------------------------------------
# Classification and keyword splitting
# --------------------------------------------------------------------------------------------


def _compact(code: str) -> tuple[str, str, list[int]]:
    """Blank-free text, its string-masked twin, and the original index of each kept char."""
    chars: list[str] = []
    masked: list[str] = []
    pos: list[int] = []
    quote = ""
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        if quote:
            chars.append(ch)
            pos.append(i)
            if ch == quote:
                if i + 1 < n and code[i + 1] == quote:
                    masked.append("\x00")
                    chars.append(ch)
                    pos.append(i + 1)
                    masked.append("\x00")
                    i += 2
                    continue
                quote = ""
                masked.append(ch)
            else:
                masked.append("\x00")
        elif ch in "'\"":
            quote = ch
            chars.append(ch)
            masked.append(ch)
            pos.append(i)
        elif not ch.isspace():
            chars.append(ch)
            masked.append(ch)
            pos.append(i)
        i += 1
    return "".join(chars), "".join(masked), pos


def _match(u: str, start: int) -> int:
    depth = 0
    for j in range(start, len(u)):
        if u[j] == "(":
            depth += 1
        elif u[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _has_top_level_comma(u: str, start: int) -> bool:
    depth = 0
    for ch in u[start:]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
    return False


def _assignment_equals(u: str) -> int:
    """Index of the ``=`` when ``u`` is ``name[(..)]...= expr``, else -1."""
    match = _NAME.match(u)
    if not match:
        return -1
    i = match.end()
    while True:
        while i < len(u) and u[i] == "(":
            j = _match(u, i)
            if j < 0:
                return -1
            i = j + 1
        member = _NAME.match(u, i + 1) if u[i : i + 1] == "%" else None
        if member is None:
            break
        i = member.end()
    if i < len(u) and u[i] == "=" and u[i + 1 : i + 2] != "=":
        return i
    return -1


def _blank_outside_parentheses(text: str) -> bool:
    depth = 0
    for ch in mask_strings(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch.isspace() and depth == 0:
            return True
    return False


def _respace(code: str, pos: list[int], bounds: list[int]) -> str:
    inserts = set()
    for b in bounds:
        if 0 < b < len(pos):
            at = pos[b]
            if at > 0 and not code[at - 1].isspace():
                inserts.add(at)
    if not inserts:
        return code.strip()
    out: list[str] = []
    for i, ch in enumerate(code):
        if i in inserts:
            out.append(" ")
        out.append(ch)
    return "".join(out).strip()


def _type_spec(u: str, start: int) -> tuple[int, list[int]] | None:
    """End index (exclusive) of a type specifier at ``u[start:]`` and token boundaries."""
    for word in _TYPE_WORDS:
        if not u.startswith(word, start):
            continue
        end = start + len(word)
        bounds = [start + 6] if word.startswith("DOUBLE") else []
        if u[end : end + 1] == "*":
            j = end + 1
            if u[j : j + 1] == "(":
                close = _match(u, j)
                if close < 0:
                    return None
                end = close + 1
            else:
                while j < len(u) and _is_digit(u[j]):
                    j += 1
                end = j
        elif u[end : end + 1] == "(" and word not in {"DOUBLEPRECISION", "DOUBLECOMPLEX"}:
            close = _match(u, end)
            inner = u[end + 1 : close] if close > 0 else ""
            if close > 0 and re.match(r"(KIND|LEN)=|\d|\*", inner):
                end = close + 1
        return end, bounds
    return None


def classify(code: str, nested: bool = False) -> Classified:
    """Classify one statement (label removed, comments stripped) and normalise its spacing.

    ``nested`` marks the statement after a logical IF, which may not be another IF.
    """
    c, m, pos = _compact(code)
    u = m.upper()
    if not u:
        return Classified("", "empty")
    text = code.strip()

    def done(kind: str, bounds: list[int], inner: Classified | None = None) -> Classified:
        spaced = _respace(code, pos, bounds)
        return Classified(spaced, kind, inner, _tree_rewrite(kind, spaced))

    eq = _assignment_equals(u)
    if u.startswith("DO"):
        do = _DO_HEAD.match(u)
        if do and _has_top_level_comma(u, do.end()):
            return done("do", [2, do.start(2)])
        if u == "DO":
            return done("do", [])
    if eq >= 0:
        if _blank_outside_parentheses(code[: pos[eq]].strip()):
            # blanks are not significant: "DO 50 I = 1.5" assigns to the variable DO50I
            glued = f"{c[:eq]}{code[pos[eq] :]}".strip()
            return Classified(glued, "assign", None, _tree_rewrite("assign", glued))
        return done("assign", [])
    if u.startswith("DO"):
        wh = _DO_WHILE.match(u)
        if wh:
            return done("do_while", [2, 2 + len(wh.group(1)), wh.end() - 6])

    if u.startswith("IF("):
        close = _match(u, 2)
        if close > 0 and not nested:
            tail = u[close + 1 :]
            if tail == "THEN":
                return done("if_then", [close + 1])
            if _ARITH_IF_TAIL.fullmatch(tail):
                return done("if_arith", [close + 1])
            if tail:
                inner = classify(code[pos[close + 1] :], nested=True)
                head = code[: pos[close + 1]].rstrip()
                return Classified(
                    f"{head} {inner.text}", "if_logical", inner, f"{head} {inner.tree_text}"
                )
        return done("unknown", [])
    if u.startswith("ELSEIF("):
        close = _match(u, 6)
        if close > 0 and u[close + 1 :] == "THEN":
            return done("else_if", [4, close + 1])
        return done("unknown", [])
    if u == "ELSE":
        return done("else", [])
    if u == "ENDIF":
        return done("end_if", [3])
    if u == "ENDDO":
        return done("end_do", [3])
    if u == "END":
        return done("end", [])
    for word in _END_UNIT_WORDS:
        if u.startswith("END" + word):
            return done("end", [3, 3 + len(word)] if word != "BLOCKDATA" else [3, 8])

    if u.startswith(("TYPE(", "CLASS(")):
        close = _match(u, u.index("("))
        if close > 0:
            return done("declaration", [close + 1])
    typed = _type_spec(u, 0)
    if typed is not None:
        end, bounds = typed
        rest = u[end:]
        if rest.startswith("FUNCTION") and len(rest) > 8:
            return done("function", [*bounds, end, end + 8])
        return done("declaration", [*bounds, end])

    if u.startswith("GOTO") and len(u) > 4:
        rest = u[4:]
        if _is_digit(rest[0]):
            return done("goto", [4])
        if rest[0] == "(":
            close = _match(u, 4)
            bounds = [4]
            if close > 0 and u[close + 1 : close + 2] != ",":
                bounds.append(close + 1)
            return done("goto_computed", bounds)
        return done("goto_assigned", [4])
    if u.startswith("ASSIGN"):
        label = _ASSIGN_LABEL.match(u)
        if label:
            return done("assign_label", [6, 6 + len(label.group(1)), 8 + len(label.group(1))])
    if u.startswith("BLOCKDATA"):
        return done("block_data", [5, 9])
    if u.startswith("IMPLICIT"):
        bounds = [8]
        spec = _type_spec(u, 8)
        if spec is not None:
            bounds.extend([*spec[1], spec[0]])
        return done("implicit", bounds)
    for word in _KEYWORDS_BY_LENGTH:
        if not u.startswith(word):
            continue
        kind = _SIMPLE_KEYWORDS[word]
        rest = u[len(word) :]
        if word == "TYPE" or (word == "MODULE" and rest.startswith("PROCEDURE")):
            kind = "f90"
        bounds = [len(word)]
        if kind in {"read", "write"} and rest.startswith("("):
            close = _match(u, len(word))
            if close > 0:
                bounds.append(close + 1)
        if kind == "format":
            bounds = []
        return done(kind, bounds)
    if u.startswith("TYPE") and len(u) > 4 and (u[4] == "*" or _is_digit(u[4])):
        return done("print", [4])
    if re.match(r"(TYPE|ENDTYPE)", u):
        return done("f90", [])
    return Classified(text, "unknown", None, text)


def _sub_masked(pattern: re.Pattern[str], repl: str, text: str, masked: str) -> str:
    """``pattern.sub`` on ``text`` that only fires where ``masked`` (strings blanked) matches."""
    out: list[str] = []
    last = 0
    for match in pattern.finditer(masked):
        out.append(text[last : match.start()])
        out.append(pattern.sub(repl, text[match.start() : match.end()], count=1))
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def _tree_rewrite(kind: str, text: str) -> str:
    """Rewrite ``text`` into a form tree-sitter-fortran parses (same statement, same line)."""
    masked = mask_strings(text)
    upper = masked.upper()
    if kind in _TREE_COMMENTED_KINDS:
        return "! " + text
    if kind in {"subroutine", "entry"}:
        open_at = masked.find("(")
        if open_at < 0:
            return text
        counter = 0

        def star(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            return f"{match.group(1)}star_{counter}"

        return text[:open_at] + re.sub(r"([(,]\s*)\*(?=\s*[,)])", star, text[open_at:])
    if kind == "call":
        return _sub_masked(re.compile(r"([(,]\s*)\*(\d)"), r"\g<1>\g<2>", text, masked)
    if kind == "common":
        return re.sub(r"(?i)^(COMMON)\s*//", r"\1 ", text) if "//" in masked else text
    if kind == "goto_assigned":
        return text if "(" in masked else f"{text}, (99999)"
    if kind == "read" and upper.startswith("ACCEPT"):
        return "READ" + text[6:]
    if kind == "print" and upper.startswith("TYPE"):
        return "PRINT" + text[4:]
    if kind == "function":
        return re.sub(r"(?i)(FUNCTION\s+[A-Z_][A-Z0-9_$]*)\s*\*\s*\d+", r"\1", text)
    return text


# --------------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------------


def _build_statements(groups: list[_Group], fixed: bool) -> list[Statement]:
    statements: list[Statement] = []
    for group in groups:
        scanner = _Scanner()
        last_piece = len(group.pieces) - 1
        for k, (text, _) in enumerate(group.pieces):
            scanner.feed(text, _FIELD_WIDTH if fixed else None, k == last_piece)
        code, raw = scanner.finish()
        first, last = group.pieces[0][1], group.pieces[-1][1]
        label = group.label.lstrip("0") or ("0" if group.label else "")
        for k, segment in enumerate(_split_semicolons(code)):
            classified = classify(segment)
            if classified.kind == "empty":
                continue
            stmt = Statement(
                index=len(statements),
                label=(label or None) if k == 0 else None,
                text=classified.text,
                raw=raw,
                kind=classified.kind,
                line=first,
                end_line=last,
                masked=mask_strings(classified.text),
                tree_text=classified.tree_text,
                n_hollerith=scanner.n_hollerith if k == 0 else 0,
            )
            if classified.inner is not None:
                inner = classified.inner
                stmt.inner = Statement(
                    index=stmt.index,
                    label=None,
                    text=inner.text,
                    raw=raw,
                    kind=inner.kind,
                    line=first,
                    end_line=last,
                    masked=mask_strings(inner.text),
                    tree_text=inner.tree_text,
                )
            statements.append(stmt)
    return statements


def read_source(source: str) -> SourceModel:
    """Read ``source`` into logical statements. Never raises on malformed text."""
    lines = split_lines(source)
    form = detect_source_form(source)
    markers: list[int] = []
    if form == FIXED:
        groups, markers = _read_fixed(lines)
    else:
        groups = _read_free(lines)
    statements = _build_statements(groups, form == FIXED)
    code_lines = frozenset(n for g in groups for _, n in g.pieces)
    return SourceModel(form, lines, statements, code_lines, tuple(sorted(set(markers))))


def normalize_source(source: str) -> NormalizedSource:
    """Free-form, one-statement-per-line text for tree-sitter with a line map to the original."""
    model = read_source(source)
    return normalize_model(model)


def normalize_model(model: SourceModel) -> NormalizedSource:
    rows: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for stmt in model.statements:
        text = stmt.tree_text or stmt.text
        rows.append(f"{stmt.label} {text}" if stmt.label and not text.startswith("!") else text)
        starts.append(stmt.line)
        ends.append(stmt.end_line)
    return NormalizedSource("\n".join(rows), starts, model.form, ends)
