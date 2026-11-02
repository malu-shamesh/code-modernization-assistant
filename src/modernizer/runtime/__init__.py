"""Runtime helpers that converted Fortran code imports as ``f77rt``.

``f77rt.py`` is deliberately a single standalone file (standard library and numpy only) so it can
be copied next to the converted modules and into the test sandbox.
"""

from __future__ import annotations

import ast
from importlib import resources

RUNTIME_FILENAME = "f77rt.py"


def runtime_source() -> str:
    """Source text of ``f77rt.py``, for writing next to converted code or into a test sandbox."""
    return resources.files(__package__).joinpath(RUNTIME_FILENAME).read_text(encoding="utf-8")


def public_names() -> frozenset[str]:
    """Names a converted module may import from ``f77rt`` (module-level, not underscore-prefixed)."""
    tree = ast.parse(runtime_source())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(n for n in names if not n.startswith("_"))


def api_doc() -> str:
    """One line per public function or class: signature and first docstring line."""
    tree = ast.parse(runtime_source())
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            summary = (ast.get_docstring(node) or "").strip().splitlines()
            args = ast.unparse(node.args)
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            lines.append(
                f"{node.name}({args}){returns}  # {summary[0] if summary else ''}".rstrip()
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            summary = (ast.get_docstring(node) or "").strip().splitlines()
            lines.append(f"class {node.name}  # {summary[0] if summary else ''}".rstrip())
    return "\n".join(lines)
