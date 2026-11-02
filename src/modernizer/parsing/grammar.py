"""tree-sitter-fortran front end: a syntax verdict on the normalised text.

The tree is built from :func:`modernizer.parsing.source.normalize_source` output (one statement
per line, so tree rows map straight back to logical statements). A fresh ``Parser`` is created
per call because parsers are not thread-safe; the ``Language`` is immutable and shared.

Pitfall (tree-sitter 0.26 on CPython 3.14): reading ``node.start_point.row`` can return a dangling
value for rows above ~256. Points are therefore always unpacked: ``row, _ = node.start_point``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tree_sitter_fortran
from tree_sitter import Language, Node, Parser

_LANGUAGE = Language(tree_sitter_fortran.language())
_LOG = logging.getLogger(__name__)


def error_spans(text: str) -> list[tuple[int, int]]:
    """Outermost ERROR/MISSING nodes as 0-based inclusive (first row, last row) pairs."""
    if not text.strip():
        return []
    try:
        tree = Parser(_LANGUAGE).parse((text + "\n").encode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - a parser failure must read as "cannot be parsed"
        _LOG.debug("tree-sitter failed", exc_info=True)
        return [(0, text.count("\n"))]
    root = tree.root_node
    if not root.has_error:
        return []
    spans: list[tuple[int, int]] = []
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            (first, _), (last, last_col) = node.start_point, node.end_point
            spans.append((first, last - 1 if last_col == 0 and last > first else last))
        elif node.has_error:
            stack.extend(node.children)
    spans.sort()
    return spans


_UNIT_NODES = frozenset({"subroutine", "function", "program", "block_data"})


@dataclass(frozen=True)
class TreeOutline:
    """Structure read from the tree: program units (kind, first row, last row) and the deepest
    DO / block-IF nesting. Rows are 0-based rows of the normalised text."""

    units: tuple[tuple[str, int, int], ...]
    max_depth: int


def tree_outline(text: str) -> TreeOutline:
    """Outline of ``text`` from the tree-sitter parse (used to cross-check the statement engine)."""
    if not text.strip():
        return TreeOutline((), 0)
    root = Parser(_LANGUAGE).parse((text + "\n").encode("utf-8", errors="replace")).root_node
    units: list[tuple[str, int, int]] = []
    max_depth = 0
    stack: list[tuple[Node, int, bool]] = [(root, 0, False)]
    while stack:
        node, depth, in_unit = stack.pop()
        kind = node.type
        if kind == "do_loop" or (
            kind == "if_statement" and any(c.type == "end_if_statement" for c in node.children)
        ):
            depth += 1
            max_depth = max(max_depth, depth)
        if kind in _UNIT_NODES and not in_unit:
            (first, _), (last, last_col) = node.start_point, node.end_point
            units.append((kind, first, last - 1 if last_col == 0 and last > first else last))
            in_unit = True
        stack.extend((child, depth, in_unit) for child in node.children)
    return TreeOutline(tuple(sorted(units, key=lambda u: u[1])), max_depth)
