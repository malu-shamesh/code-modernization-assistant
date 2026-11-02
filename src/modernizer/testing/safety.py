"""Static screening of LLM-written test code.

``sanitize_tests`` is a best-effort guard that catches the obvious ways a generated test could
touch the network, spawn processes or delete files. It is NOT a security boundary: names can be
built dynamically (``getattr(os, "sys" + "tem")``) and no AST check can see everything. The real
isolation is the container the service runs in (Cloud Run: no credentials in the environment,
ephemeral filesystem, no inbound access) plus the scrubbed environment and process limits in
``modernizer.testing.runner``.
"""

from __future__ import annotations

import ast
import re
import sys

from ..conventions import RUNTIME_MODULE

TARGET_MODULE = "target_module"
STAR_IMPORT_OK: frozenset[str] = frozenset({RUNTIME_MODULE, TARGET_MODULE})
THIRD_PARTY_ALLOWED: frozenset[str] = frozenset({"numpy", "pytest", RUNTIME_MODULE, TARGET_MODULE})
ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(sys.stdlib_module_names) | THIRD_PARTY_ALLOWED

DENIED_MODULES: tuple[str, ...] = (
    "subprocess",
    "socket",
    "ctypes",
    "multiprocessing",
    "urllib.request",
    "requests",
    "http.client",
    "numpy.ctypeslib",
    "numpy.f2py",
    "numpy.distutils",
)
DENIED_CALLS: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "os.remove",
    "os.rmdir",
    "os.removedirs",
    "os.unlink",
    "os.fork",
    "os.kill",
    "shutil.rmtree",
)
# numpy functions whose first argument is an output path (or file) to be written.
NUMPY_WRITE_FUNCTIONS: frozenset[str] = frozenset(
    {"numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt"}
)
DENIED_CALL_PREFIXES: tuple[str, ...] = ("os.exec", "os.spawn", "os.posix_spawn")
DYNAMIC_CODE_FUNCTIONS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib.import_module",
        "builtins.eval",
        "builtins.exec",
        "builtins.compile",
        "builtins.__import__",
    }
)
IMPORT_FUNCTIONS: frozenset[str] = frozenset(
    {"__import__", "builtins.__import__", "importlib.import_module"}
)
PATH_WRITE_METHODS: frozenset[str] = frozenset({"write_text", "write_bytes", "unlink", "rmdir"})
SAFE_PATH_NAMES: frozenset[str] = frozenset({"tmp_path", "tmp_path_factory"})

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)


class TestGenerationError(Exception):
    """Generated tests are unusable. ``raw_response`` carries the original model reply."""

    __test__ = False  # not a pytest test class

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


def extract_code(text: str) -> str:
    """Return the body of the first python fenced block, or ``text`` itself when unfenced.

    An opening fence without a closing one (truncated output) is tolerated here; the later
    parse step is what rejects incomplete code.
    """
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text.strip()
    for lang, body in blocks:
        if lang.lower() in ("python", "py", "python3", ""):
            return body.strip()
    return blocks[0][1].strip()


def _matches(name: str, denied: str) -> bool:
    return name == denied or name.startswith(denied + ".")


def _denied_by(name: str) -> str | None:
    for entry in (*DENIED_MODULES, *DENIED_CALLS):
        if _matches(name, entry):
            return entry
    for prefix in DENIED_CALL_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return None


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve ``a.b.c`` to its fully qualified dotted name through import aliases.

    A dotted name only resolves when its root was imported, so a local variable that happens to
    be called ``socket`` or ``requests`` is not mistaken for the module. ``__import__("os")`` and
    ``importlib.import_module("os")`` with a literal argument resolve to that module.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Call) and parts:
        callee = _dotted(node.func, aliases)
        target = _literal_str(node.args[0]) if node.args else None
        if callee in IMPORT_FUNCTIONS and target:
            return ".".join([target, *reversed(parts)])
        return None
    if not isinstance(node, ast.Name):
        return None
    if node.id not in aliases:
        return node.id if not parts else None
    return ".".join([aliases[node.id], *reversed(parts)])


def _collect_aliases(tree: ast.AST) -> dict[str, str]:
    """Map every locally bound import name to the dotted name it refers to."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _safe_path_names(tree: ast.AST) -> set[str]:
    """Names that hold something derived from ``tmp_path``/``tmp_path_factory``."""
    safe = set(SAFE_PATH_NAMES)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                targets, value = [node.optional_vars], node.context_expr
            if value is None or not (_names_in(value) & safe):
                continue
            for target in targets:
                for name in _names_in(target):
                    if name not in safe:
                        safe.add(name)
                        changed = True
    return safe


def _open_mode(call: ast.Call, mode_index: int) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    if len(call.args) > mode_index:
        return call.args[mode_index]
    return None


def _writes(mode: ast.expr | None) -> bool:
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(c in mode.value for c in "wax+")
    return True  # mode computed at runtime: assume the worst


def _literal_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _root(name: str) -> str:
    return name.split(".")[0]


def _import_not_allowed(name: str) -> str:
    return (
        f"import of '{name}' is not allowed (only the standard library, numpy, pytest, "
        f"{RUNTIME_MODULE} and {TARGET_MODULE} may be imported)"
    )


def _check_import(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
        names = modules
    elif node.level or not node.module:
        return ["relative imports are not allowed"]
    else:
        modules = [node.module]
        names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        if any(a.name == "*" for a in node.names) and _root(node.module) not in STAR_IMPORT_OK:
            # A star import would hide denied names (``from os import *; system(...)``).
            return [f"star import from '{node.module}' is not allowed"]
    problems = [f"import of '{n}' is not allowed" for n in names if _denied_by(n)]
    problems.extend(
        _import_not_allowed(m)
        for m in modules
        if _root(m) not in ALLOWED_IMPORT_ROOTS and not _denied_by(m)
    )
    return problems


def _check_call(call: ast.Call, aliases: dict[str, str], safe_paths: set[str]) -> list[str]:
    problems: list[str] = []
    callee = None
    if isinstance(call.func, (ast.Name, ast.Attribute)):
        callee = _dotted(call.func, aliases)

    if callee in DYNAMIC_CODE_FUNCTIONS:
        target = _literal_str(call.args[0]) if call.args else None
        if target is None:
            problems.append(f"{callee}() of a non-literal is not allowed")
        elif callee in IMPORT_FUNCTIONS:
            if _denied_by(target):
                problems.append(f"import of '{target}' is not allowed")
            elif _root(target) not in ALLOWED_IMPORT_ROOTS:
                problems.append(_import_not_allowed(target))

    if callee in NUMPY_WRITE_FUNCTIONS or (
        isinstance(call.func, ast.Attribute) and call.func.attr == "tofile"
    ):
        path = call.args[0] if call.args else None
        if path is None or not (_names_in(path) & safe_paths):
            problems.append("numpy writes outside tmp_path are not allowed")
    elif callee == "numpy.load" and any(kw.arg == "allow_pickle" for kw in call.keywords):
        problems.append("numpy.load(allow_pickle=...) is not allowed")

    if callee in ("open", "builtins.open", "io.open"):
        if _writes(_open_mode(call, 1)):
            path = call.args[0] if call.args else None
            if path is None or not (_names_in(path) & safe_paths):
                problems.append("open() for writing outside tmp_path is not allowed")
    elif isinstance(call.func, ast.Attribute) and call.func.attr == "open":
        if _writes(_open_mode(call, 0)) and not (_names_in(call.func.value) & safe_paths):
            problems.append(".open() for writing outside tmp_path is not allowed")
    elif isinstance(call.func, ast.Attribute) and call.func.attr in PATH_WRITE_METHODS:
        if not (_names_in(call.func.value) & safe_paths):
            problems.append(f".{call.func.attr}() outside tmp_path is not allowed")
    return problems


def _violations(tree: ast.AST) -> list[str]:
    aliases = _collect_aliases(tree)
    safe_paths = _safe_path_names(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        problems: list[str] = []
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems = _check_import(node)
        elif isinstance(node, ast.Attribute):
            name = _dotted(node, aliases)
            if name and _denied_by(name):
                problems = [f"use of '{name}' is not allowed"]
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            resolved = aliases.get(node.id)
            if resolved and _denied_by(resolved):
                problems = [f"use of '{resolved}' is not allowed"]
        elif isinstance(node, ast.Call):
            problems = _check_call(node, aliases, safe_paths)
        found.extend(f"line {getattr(node, 'lineno', '?')}: {p}" for p in problems)
    return found


def _imports_target(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == TARGET_MODULE for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module.split(".")[0] == TARGET_MODULE:
                return True
    return False


def _has_test_function(tree: ast.AST) -> bool:
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        for n in ast.walk(tree)
    )


def sanitize_tests(source: str) -> str:
    """Return clean test source or raise ``TestGenerationError``.

    Strips markdown fences, then requires: valid Python, an import of ``target_module``, at least
    one ``test_*`` function, imports only from the standard library, numpy, pytest, ``f77rt`` and
    ``target_module``, and no use of the deny-list (process, network, destructive filesystem
    calls, dynamic code execution, writes outside ``tmp_path``). Best effort only; see the
    module docstring.
    """
    code = extract_code(source)
    if not code:
        raise TestGenerationError("empty test source", source)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise TestGenerationError(
            f"generated tests are not valid Python: {exc.msg} (line {exc.lineno})", source
        ) from exc
    if not _imports_target(tree):
        raise TestGenerationError(f"generated tests never import {TARGET_MODULE}", source)
    if not _has_test_function(tree):
        raise TestGenerationError("generated tests contain no test_ function", source)
    problems = _violations(tree)
    if problems:
        raise TestGenerationError(
            "generated tests use forbidden constructs: " + "; ".join(problems), source
        )
    return code + "\n"
