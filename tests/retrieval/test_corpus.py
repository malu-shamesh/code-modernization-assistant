"""Validation of the curated FORTRAN 77 -> Python migration corpus.

The corpus is plain data, so these tests read it with ``json`` only and never execute snippets:
``modern`` examples are checked with ``compile`` to guarantee they are valid Python 3 syntax, and
``legacy`` examples are checked for a plausible fixed-form column layout.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from modernizer.features import FEATURE_IDS, FEATURES, is_semantic
from modernizer.runtime import public_names

CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "modernizer"
    / "retrieval"
    / "data"
    / "f77_to_python.jsonl"
)

EXPECTED_KEYS = ("id", "title", "feature_ids", "legacy", "modern", "notes")
KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
EM_DASH = chr(0x2014)

MAX_SNIPPET_CHARS = 1500
MAX_NOTES_CHARS = 700
MIN_ENTRIES = 110
MAX_ENTRIES = 150
SEMANTIC_MIN_ENTRIES = 2

# The runtime helper API that converted code may import from f77rt. Hard-coded on purpose: the
# corpus is ground truth for the model, so it must not drift when the runtime changes silently.
HELPER_API = frozenset(
    """
    idiv fmod sign nint anint aint int_ dim ipow ichar char_ index_ len_trim lge lgt lle llt
    str_eq str_cmp char_assign substr set_substr do_trips do_range do_final FortranStop
    write_formatted write_list read_formatted read_list format_record open_unit close_unit
    rewind_unit backspace_unit endfile_unit write_internal read_internal iand ior ieor ishft
    not_ d1mach r1mach i1mach cmplx zeros_f
    """.split()
)

# Fortran intrinsic names that have no Python or runtime equivalent of the same name: a bare call
# to one of them in a ``modern`` snippet is a leftover that would fail at run time (math.sqrt is
# an attribute call and is fine; a bare sqrt(x) is not, even when the legacy text also says SQRT).
FORTRAN_ONLY_NAMES = frozenset(
    """
    dabs dsqrt dexp dlog dlog10 dsin dcos dtan datan datan2 dmax1 dmin1 amax1 amin1 amax0 amin0
    max0 min0 max1 min1 iabs idint ifix dble sngl dfloat alog alog10 amod dmod isign idim dsign
    cabs csqrt cexp clog conjg aimag btest ibset ibclr ibits ishftc dprod dnint idnint mod
    sqrt exp log log10 sin cos tan asin acos atan atan2 sinh cosh tanh real dreal dimag dconjg
    """.split()
)

COMMENT_MARKERS = ("C", "c", "*", "!")


def _load_lines() -> list[str]:
    # Tolerate CRLF so a Windows checkout with core.autocrlf does not break the suite; the
    # loader strips each line, so the data stays valid either way.
    raw = CORPUS_PATH.read_bytes()
    return raw.decode("utf-8").replace("\r\n", "\n").split("\n")


def _load_entries() -> list[dict[str, Any]]:
    return [json.loads(line) for line in _load_lines() if line]


ENTRIES = _load_entries()


def _entry_ids() -> list[str]:
    return [entry["id"] for entry in ENTRIES]


def test_file_layout() -> None:
    lines = _load_lines()
    assert lines[-1] == "", "file must end with a newline"
    assert all(line for line in lines[:-1]), "blank lines are not allowed"
    assert not CORPUS_PATH.read_bytes().startswith(b"\xef\xbb\xbf"), "no BOM"
    assert not any("\r" in line for line in lines), "stray carriage return inside a record"


def test_entry_count_in_target_range() -> None:
    assert MIN_ENTRIES <= len(ENTRIES) <= MAX_ENTRIES


def test_ascii_only_and_no_em_dash() -> None:
    raw = CORPUS_PATH.read_bytes()
    assert raw.isascii(), "corpus file must be pure ASCII"
    for entry in ENTRIES:
        for key in ("id", "title", "legacy", "modern", "notes"):
            assert EM_DASH not in entry[key], f"em dash in {entry['id']}.{key}"
            assert entry[key].isascii(), f"non-ASCII text in {entry['id']}.{key}"


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_entry_schema(entry: dict[str, Any]) -> None:
    assert tuple(entry) == EXPECTED_KEYS
    for key in ("id", "title", "legacy", "modern", "notes"):
        assert isinstance(entry[key], str), key
        assert entry[key].strip(), f"{key} must not be empty"
    # Fixed-form legacy code starts with significant blanks (columns 1-6), so only blank
    # lines at either end and trailing whitespace are rejected for it.
    assert entry["legacy"] == entry["legacy"].strip("\n").rstrip(), "legacy has stray whitespace"
    assert not entry["legacy"].startswith("\n")
    for key in ("id", "title", "modern", "notes"):
        assert entry[key] == entry[key].strip(), f"{key} has surrounding whitespace"
    assert KEBAB_CASE.match(entry["id"]), f"id {entry['id']!r} is not kebab-case"
    assert isinstance(entry["feature_ids"], list)
    assert entry["feature_ids"], "at least one feature id is required"
    assert all(isinstance(fid, str) for fid in entry["feature_ids"])
    assert len(set(entry["feature_ids"])) == len(entry["feature_ids"]), "duplicate feature ids"
    assert "\n" not in entry["notes"], "notes are a single paragraph"


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_entry_sizes(entry: dict[str, Any]) -> None:
    assert len(entry["legacy"]) <= MAX_SNIPPET_CHARS
    assert len(entry["modern"]) <= MAX_SNIPPET_CHARS
    assert len(entry["notes"]) <= MAX_NOTES_CHARS
    assert len(entry["title"]) <= 100


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_modern_snippet_compiles(entry: dict[str, Any]) -> None:
    source = textwrap.dedent(entry["modern"])
    compile(source, "<modern>", "exec")


def _fixed_form_problems(legacy: str) -> list[str]:
    problems: list[str] = []
    previous_was_statement = False
    for number, line in enumerate(legacy.split("\n"), start=1):
        if not line.strip():
            problems.append(f"line {number}: blank line")
            continue
        if line[0] in COMMENT_MARKERS:
            continue
        if "\t" in line:
            problems.append(f"line {number}: tab character")
        if line != line.rstrip():
            problems.append(f"line {number}: trailing blanks")
        text = line.rstrip()
        if len(text) > 72:
            problems.append(f"line {number}: {len(text)} columns, limit is 72")
        field = text[:5].ljust(5)
        if any(ch not in "0123456789 " for ch in field):
            problems.append(f"line {number}: columns 1-5 hold something other than a label")
        if len(text) < 7:
            problems.append(f"line {number}: no statement text after column 6")
            continue
        continuation = text[5] not in " 0"
        if continuation:
            if field.strip():
                problems.append(f"line {number}: continuation line with a label")
            if not previous_was_statement:
                problems.append(f"line {number}: continuation without a preceding statement")
        elif field.strip() and not field.strip().isdigit():
            problems.append(f"line {number}: malformed label {field!r}")
        previous_was_statement = True
    return problems


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_legacy_snippet_is_plausible_fixed_form(entry: dict[str, Any]) -> None:
    assert _fixed_form_problems(entry["legacy"]) == []


def test_fixed_form_check_detects_problems() -> None:
    bad = "\n".join(
        [
            "      X = 1.0",
            "   10X = 2.0",
            "     +     CONTINUED",
            " 1 0 CONTINUE",
            "      Y = " + "A" * 70,
            "  100+ CONTINUED WITH LABEL",
        ]
    )
    problems = _fixed_form_problems(bad)
    assert any("line 2" in p for p in problems)
    assert any("line 4" in p for p in problems)
    assert any("line 5" in p and "columns" in p for p in problems)
    assert any("line 6" in p for p in problems)
    assert _fixed_form_problems("C comment\n      X = 1.0\n     +  + 2.0") == []


class _NameCollector(ast.NodeVisitor):
    """Collects names bound and names called without a local binding in a snippet."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.called: set[str] = set()
        self.loaded: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        else:
            self.bound.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.bound.add(node.name)
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            self.bound.add(arg.arg)
        for arg in (args.vararg, args.kwarg):
            if arg is not None:
                self.bound.add(arg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            self.called.add(node.func.id)
        self.generic_visit(node)


def _collect(source: str) -> _NameCollector:
    collector = _NameCollector()
    collector.visit(ast.parse(source))
    return collector


def _unresolved_callees(entry: dict[str, Any]) -> set[str]:
    """Called names that are neither local, builtin, a runtime helper nor a routine named in the
    legacy snippet (a CALL target or function reference)."""
    collector = _collect(textwrap.dedent(entry["modern"]))
    legacy_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", entry["legacy"].upper()))
    unresolved = set()
    for name in collector.called - collector.bound:
        if hasattr(builtins, name) or name in HELPER_API:
            continue
        if name.upper() in legacy_tokens:
            continue
        unresolved.add(name)
    return unresolved


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_modern_snippet_uses_only_the_helper_api(entry: dict[str, Any]) -> None:
    tree = ast.parse(textwrap.dedent(entry["modern"]))
    imported: set[str] = set()
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "f77rt":
            imported.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "f77rt"
        ):
            attributes.add(node.attr)
    assert imported <= HELPER_API, f"unknown f77rt imports {sorted(imported - HELPER_API)}"
    assert attributes <= HELPER_API, f"unknown f77rt attributes {sorted(attributes - HELPER_API)}"
    assert _unresolved_callees(entry) == set(), "call to a name that is not in the f77rt API"
    collector = _collect(textwrap.dedent(entry["modern"]))
    leftovers = (collector.called - collector.bound) & FORTRAN_ONLY_NAMES
    assert not leftovers, f"Fortran-only intrinsic left in modern code: {sorted(leftovers)}"


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_modern_snippet_imports_only_at_the_top(entry: dict[str, Any]) -> None:
    tree = ast.parse(textwrap.dedent(entry["modern"]))
    seen_other = False
    for node in tree.body:
        is_import = isinstance(node, ast.Import | ast.ImportFrom)
        assert not (is_import and seen_other), "import after other code"
        seen_other = seen_other or not is_import
    nested = [
        n
        for top in tree.body
        if not isinstance(top, ast.Import | ast.ImportFrom)
        for n in ast.walk(top)
        if isinstance(n, ast.Import | ast.ImportFrom)
    ]
    assert not nested, "inline import inside a function or block"


def test_helper_api_is_exposed_by_the_runtime() -> None:
    missing = HELPER_API - public_names()
    assert not missing, f"runtime does not define {sorted(missing)}"


def test_name_scan_detects_leftovers() -> None:
    entry = {"modern": "y = dsqrt(x) + helper(x) + idiv(1, 2)", "legacy": "      Y = DSQRT(X)"}
    assert _unresolved_callees(entry) == {"helper"}
    collector = _collect(entry["modern"])
    assert (collector.called - collector.bound) & FORTRAN_ONLY_NAMES == {"dsqrt"}
    # A bare generic name resolves through the legacy text, so only the deny list catches it.
    bare = _collect("y = sqrt(x) + math.sin(x)")
    assert (bare.called - bare.bound) & FORTRAN_ONLY_NAMES == {"sqrt"}


def _statements(legacy: str) -> list[tuple[str, str]]:
    """(label, text) per statement of a fixed-form snippet, continuation lines joined."""
    statements: list[tuple[str, str]] = []
    for line in legacy.split("\n"):
        if line[0] in COMMENT_MARKERS:
            continue
        text = line.rstrip()
        if len(text) > 5 and text[5] not in " 0" and statements:
            label, previous = statements[-1]
            statements[-1] = (label, previous + text[6:])
        else:
            statements.append((text[:5].strip(), text[6:]))
    return statements


LABEL_REFERENCES = (
    re.compile(r"GOTO(\d+)"),
    re.compile(r"GOTO\(([\d,]+)\)"),
    re.compile(r"^(?:IF\(.*\))?DO(\d+)[A-Z]"),
    re.compile(r"^IF\(.*\)(\d+),(\d+),(\d+)$"),
    re.compile(r"END=(\d+)"),
    re.compile(r"ASSIGN(\d+)TO"),
    re.compile(r"^(?:IF\(.*\))?(?:WRITE|READ)\([^,)]*,(\d+)\)"),
)
# CALL f(*100, *200): the star-label actual arguments of an alternate-return call.
ALTERNATE_RETURN_ACTUAL = re.compile(r"[(,]\*(\d+)")


def _label_problems(legacy: str) -> list[str]:
    statements = _statements(legacy)
    labels = [label for label, _ in statements if label]
    problems = [
        f"label {label} defined twice" for label in sorted(set(labels)) if labels.count(label) > 1
    ]
    for _, text in statements:
        squeezed = re.sub(r"'[^']*'", "''", text).upper().replace(" ", "")
        patterns = [*LABEL_REFERENCES]
        if "CALL" in squeezed:
            patterns.append(ALTERNATE_RETURN_ACTUAL)
        for pattern in patterns:
            for match in pattern.finditer(squeezed):
                for group in match.groups():
                    for target in group.split(","):
                        if target not in labels:
                            problems.append(f"reference to undefined label {target}")
    return problems


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_ids())
def test_legacy_labels_are_defined_once_and_references_resolve(entry: dict[str, Any]) -> None:
    assert _label_problems(entry["legacy"]) == []


def test_label_check_detects_problems() -> None:
    bad = "\n".join(
        [
            "      GO TO 20",
            "   10 CONTINUE",
            "   10 CONTINUE",
            "      DO 30 I = 1, N",
            "      IF (X) 10, 40, 10",
            "      CALL CHECK(A, *10, *50)",
        ]
    )
    problems = _label_problems(bad)
    assert "label 10 defined twice" in problems
    assert {"reference to undefined label 20", "reference to undefined label 30"} <= set(problems)
    assert "reference to undefined label 40" in problems
    assert "reference to undefined label 50" in problems
    good = "      GO TO (10, 20), K\n   10 CONTINUE\n   20 X = 1.0"
    assert _label_problems(good) == []


def test_ids_are_unique() -> None:
    duplicates = [i for i, n in Counter(_entry_ids()).items() if n > 1]
    assert not duplicates, f"duplicate ids: {duplicates}"


def test_titles_are_unique() -> None:
    duplicates = [t for t, n in Counter(e["title"] for e in ENTRIES).items() if n > 1]
    assert not duplicates, f"duplicate titles: {duplicates}"


def test_feature_ids_are_valid() -> None:
    for entry in ENTRIES:
        unknown = [fid for fid in entry["feature_ids"] if fid not in FEATURES]
        assert not unknown, f"{entry['id']} references unknown features {unknown}"


def test_every_feature_is_covered() -> None:
    covered = {fid for entry in ENTRIES for fid in entry["feature_ids"]}
    missing = [fid for fid in FEATURE_IDS if fid not in covered]
    assert not missing, f"features without corpus entries: {missing}"


def test_semantic_features_have_several_entries() -> None:
    counts = Counter(fid for entry in ENTRIES for fid in entry["feature_ids"])
    thin = {
        fid: counts[fid]
        for fid in FEATURE_IDS
        if is_semantic(fid) and counts[fid] < SEMANTIC_MIN_ENTRIES
    }
    assert not thin, f"semantic features with fewer than two entries: {thin}"


def test_legacy_and_modern_differ() -> None:
    unchanged = [e["id"] for e in ENTRIES if e["legacy"] == e["modern"]]
    assert unchanged == []
