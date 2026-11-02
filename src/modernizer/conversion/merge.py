"""Merging separately converted slices into one module."""

from __future__ import annotations

import ast
import re
import threading
import warnings
from collections.abc import Sequence
from typing import cast

_PARSE_LOCK = threading.Lock()
_BLANK_RUNS = re.compile(r"\n{4,}")
_MAX_IMPORT_LINE = 88

_PlainKey = tuple[str, str, str | None]  # ("import", module, asname)
_FromKey = tuple[str, int, str | None]  # ("from", level, module)


def _parse(source: str) -> ast.Module:
    """``ast.parse`` without SyntaxWarnings (e.g. invalid escapes in regex strings).

    ``warnings.catch_warnings`` is not thread-safe, so concurrent merges are serialised.
    """
    with _PARSE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(source)


class _ImportTable:
    """Top-level imports in first-seen order; ``from m import a`` lines of one module merge."""

    def __init__(self) -> None:
        self._order: list[_PlainKey | _FromKey] = []
        self._from_names: dict[_FromKey, list[tuple[str, str | None]]] = {}

    def add(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                key: _PlainKey = ("import", alias.name, alias.asname)
                if key not in self._order:
                    self._order.append(key)
            return
        from_key: _FromKey = ("from", node.level, node.module)
        if from_key not in self._from_names:
            self._from_names[from_key] = []
            self._order.append(from_key)
        names = self._from_names[from_key]
        for alias in node.names:
            pair = (alias.name, alias.asname)
            if pair not in names:
                names.append(pair)

    def render(self) -> str:
        future = [k for k in self._order if k[0] == "from" and k[2] == "__future__"]
        rest = [k for k in self._order if k not in future]
        return "\n".join(self._render_one(k) for k in (*future, *rest))

    def _render_one(self, key: _PlainKey | _FromKey) -> str:
        if key[0] == "import":
            _, name, asname = key
            return f"import {name} as {asname}" if asname else f"import {name}"
        from_key = cast("_FromKey", key)
        _, level, module = from_key
        head = f"from {'.' * level}{module or ''} import"
        names = self._from_names[from_key]
        if any(name == "*" for name, _ in names):
            return f"{head} *"
        items = [f"{name} as {asname}" if asname else name for name, asname in names]
        line = f"{head} {', '.join(items)}"
        if len(line) <= _MAX_IMPORT_LINE:
            return line
        return f"{head} (\n" + "".join(f"    {item},\n" for item in items) + ")"


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _start_line(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in target.elts:
            names |= _target_names(element)
        return names
    return set()


def _defined_names(tree: ast.Module) -> set[str]:
    """Names bound by top-level defs, classes and plain assignments (not imports)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _target_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _target_names(node.target)
    return names


def _redefines(node: ast.stmt, names: set[str]) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name in names
    if isinstance(node, ast.Assign):
        bound: set[str] = set()
        for target in node.targets:
            bound |= _target_names(target)
        return bool(bound) and bound <= names
    if isinstance(node, ast.AnnAssign):
        bound = _target_names(node.target)
        return bool(bound) and bound <= names
    return False


def _is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` (either operand order)."""
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    operands = [test.left, *test.comparators]
    has_name = any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
    has_main = any(isinstance(o, ast.Constant) and o.value == "__main__" for o in operands)
    return has_name and has_main


def _span_lines(node: ast.stmt) -> range:
    return range(_start_line(node), (node.end_lineno or node.lineno) + 1)


def merge_sources(sources: Sequence[str], *, prelude: bool = False) -> str:
    """Merge converted slices (the first one is the state prelude when ``prelude`` is true).

    The module docstring of the first source stays on top, then come the hoisted and
    de-duplicated top-level imports (``__future__`` first, ``from m import a`` and
    ``from m import b`` merged), then the bodies in order. Top-level definitions of the
    prelude's names are dropped from later slices. ``if __name__ == "__main__":`` blocks are
    moved behind everything else, so a PROGRAM converted in an early slice can call the units
    of later slices. A source that does not parse is appended unchanged; the validator reports
    it later.
    """
    imports = _ImportTable()
    header = ""
    bodies: list[str] = []
    guards: list[str] = []
    prelude_names: set[str] = set()

    for position, source in enumerate(sources):
        try:
            tree = _parse(source)
        except SyntaxError:
            bodies.append(source.strip("\n"))
            continue
        lines = source.split("\n")
        drop: set[int] = set()
        spans = [(n.lineno, n.end_lineno or n.lineno) for n in tree.body]
        for index, node in enumerate(tree.body):
            lines_of_node = _span_lines(node)
            alone = (
                sum(1 for s, e in spans if s <= lines_of_node[-1] and lines_of_node[0] <= e) == 1
            )
            if index == 0 and _is_docstring(node):
                if position == 0:
                    header = "\n".join(lines[node.lineno - 1 : node.end_lineno]).strip("\n")
                drop.update(lines_of_node)
            elif isinstance(node, ast.Import | ast.ImportFrom) and alone:
                imports.add(node)
                drop.update(lines_of_node)
            elif alone and _is_main_guard(node):
                guard = "\n".join(lines[lines_of_node[0] - 1 : lines_of_node[-1]])
                if guard not in guards:
                    guards.append(guard)
                drop.update(lines_of_node)
            elif prelude and position > 0 and alone and _redefines(node, prelude_names):
                drop.update(lines_of_node)
        if prelude and position == 0:
            prelude_names = _defined_names(tree)
        body = "\n".join(ln for n, ln in enumerate(lines, start=1) if n not in drop)
        body = _BLANK_RUNS.sub("\n\n\n", body).strip("\n")
        if body.strip():
            bodies.append(body)

    top = "\n\n".join(block for block in (header, imports.render()) if block)
    return "\n\n\n".join(block for block in (top, *bodies, *guards) if block) + "\n"
