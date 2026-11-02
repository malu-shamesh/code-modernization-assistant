"""Static checks on the generated Python before any generated test is run.

Issue codes (``ValidationIssue.code``):

``empty_output``            nothing to check.
``syntax_error``            Python's own ``compile`` rejects the text (reported with its line).
``disallowed_import``       an import outside the standard library, numpy and ``f77rt``.
``unknown_runtime_helper``  ``from f77rt import X`` / ``f77rt.X`` where ``X`` is not a public
                            name of the runtime (``modernizer.runtime.public_names()``).
``fortran_remnant``         Fortran text that still compiles: calls to GOTO-like names or Fortran
                            statement words that nothing defines, dotted-operator attribute
                            chains (``a.eq.b``) and bare statement-label integers.
``api_changed``             a program unit's Python function is missing, has other parameters
                            than the Fortran dummy arguments (a keyword name takes a trailing
                            underscore), or returns the wrong shape (see
                            ``modernizer.conventions.CALLING_CONVENTION``).
"""

from __future__ import annotations

import ast
import keyword
import sys
import threading
import warnings
from collections.abc import Iterable

from .. import runtime
from ..models import ApiSymbol, ParsedModule, ValidationIssue, ValidationReport
from .analyzer import extract_api

_WARNINGS_LOCK = threading.Lock()  # warnings.catch_warnings swaps global state
_NAMES_LOCK = threading.Lock()
_NAMES_CACHE: tuple[object, int, frozenset[str]] | None = None
_ALLOWED_THIRD_PARTY = frozenset({"numpy", "f77rt"})
_RUNTIME_MODULE = "f77rt"
_REMNANT_CALLS = frozenset(
    {
        "goto",
        "go_to",
        "gosub",
        "endif",
        "enddo",
        "end_if",
        "end_do",
        "implicit",
        "common",
        "equivalence",
        "dimension",
        "subroutine",
        "function",
        "write",
        "read",
        "rewind",
        "backspace",
        "endfile",
    }
)
_DOTTED_ATTRIBUTES = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "eqv", "neqv", "xor"})
_MAX_SYNTAX_MESSAGE = 200


def _compile(source: str) -> tuple[ast.Module | None, ValidationIssue | None]:
    try:
        with _WARNINGS_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source, "<converted>")
            compile(tree, "<converted>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        message = (exc.msg or "invalid syntax")[:_MAX_SYNTAX_MESSAGE]
        return None, ValidationIssue("syntax_error", message, exc.lineno)
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:
        return None, ValidationIssue("syntax_error", f"source cannot be compiled: {exc}")
    return tree, None


def _import_issues(tree: ast.Module) -> tuple[list[ValidationIssue], set[str]]:
    """Import problems, plus the local names bound to the ``f77rt`` module."""
    issues: list[ValidationIssue] = []
    aliases: set[str] = set()
    stdlib = sys.stdlib_module_names
    helpers = _runtime_names()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top == _RUNTIME_MODULE and alias.name == _RUNTIME_MODULE:
                    aliases.add(alias.asname or alias.name)
                if top not in stdlib and top not in _ALLOWED_THIRD_PARTY:
                    issues.append(_disallowed(alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level or (top not in stdlib and top not in _ALLOWED_THIRD_PARTY):
                issues.append(_disallowed("." * node.level + (node.module or ""), node.lineno))
            elif node.module == _RUNTIME_MODULE and helpers is not None:
                for alias in node.names:
                    if alias.name == "*" or alias.name not in helpers:
                        issues.append(_unknown_helper(alias.name, node.lineno))
    return issues, aliases


def _disallowed(name: str, line: int) -> ValidationIssue:
    message = f"import of '{name}' is not allowed (standard library, numpy and f77rt only)"
    return ValidationIssue("disallowed_import", message, line)


def _unknown_helper(name: str, line: int) -> ValidationIssue:
    message = f"'{name}' is not a public helper of the {_RUNTIME_MODULE} runtime"
    return ValidationIssue("unknown_runtime_helper", message, line)


def _runtime_names() -> frozenset[str] | None:
    """Public helper names of the runtime, cached until the runtime source or accessor changes."""
    global _NAMES_CACHE
    accessor = runtime.public_names
    try:
        digest = hash(runtime.runtime_source())
        with _NAMES_LOCK:
            if _NAMES_CACHE is not None and _NAMES_CACHE[:2] == (accessor, digest):
                return _NAMES_CACHE[2]
        names = frozenset(accessor())
    except Exception:  # noqa: BLE001 - an unreadable runtime disables only the helper check
        return None
    with _NAMES_LOCK:
        _NAMES_CACHE = (accessor, digest, names)
    return names


def _helper_attribute_issues(tree: ast.Module, aliases: set[str]) -> list[ValidationIssue]:
    helpers = _runtime_names()
    if not aliases or helpers is None:
        return []
    issues = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr not in helpers
        ):
            issues.append(_unknown_helper(node.attr, node.lineno))
    return issues


def _module_names(tree: ast.Module) -> set[str]:
    """Names bound anywhere in the module (functions, classes, imports, assignments, args)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _remnant_issues(tree: ast.Module) -> list[ValidationIssue]:
    defined = _module_names(tree)
    modules = {
        (a.asname or a.name).split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in n.names
    }
    issues: list[ValidationIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id.lower()
            if name in _REMNANT_CALLS and node.func.id not in defined:
                message = f"Fortran statement '{node.func.id}(...)' is still in the output"
                issues.append(ValidationIssue("fortran_remnant", message, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr.lower() in _DOTTED_ATTRIBUTES:
            base = node.value
            if not (isinstance(base, ast.Name) and base.id in modules):
                message = f"unconverted dotted operator '.{node.attr}.' (attribute chain)"
                issues.append(ValidationIssue("fortran_remnant", message, node.lineno))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
            and not isinstance(node.value.value, bool)
        ):
            message = f"bare statement label '{node.value.value}' left in the code"
            issues.append(ValidationIssue("fortran_remnant", message, node.lineno))
    return issues


def _returns(func: ast.FunctionDef) -> list[ast.Return]:
    """``return`` statements of ``func`` itself (nested functions and classes excluded)."""
    found: list[ast.Return] = []
    stack: list[ast.AST] = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            found.append(node)
        elif not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            stack.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda r: r.lineno)


def _has_value(ret: ast.Return) -> bool:
    return ret.value is not None and not (
        isinstance(ret.value, ast.Constant) and ret.value.value is None
    )


def _expected_items(symbol: ApiSymbol) -> int:
    return (
        (1 if symbol.kind == "function" else 0)
        + len(symbol.returns)
        + (1 if symbol.alt_returns else 0)
    )


def _shape_issues(symbol: ApiSymbol, func: ast.FunctionDef) -> Iterable[ValidationIssue]:
    expected = _expected_items(symbol)
    returns = _returns(func)
    name = symbol.name
    if expected == 0:
        for ret in returns:
            if _has_value(ret):
                message = f"subroutine '{name}' has no outputs and must not return a value"
                yield ValidationIssue("api_changed", message, ret.lineno)
        return
    if not any(_has_value(r) for r in returns):
        yield ValidationIssue(
            "api_changed", f"'{name}' must return a value but never does", func.lineno
        )
        return
    for ret in returns:
        if not _has_value(ret):
            message = f"'{name}' must return {expected} value(s) on every path, found a bare return"
            yield ValidationIssue("api_changed", message, ret.lineno)
            continue
        is_tuple = isinstance(ret.value, ast.Tuple)
        if expected == 1 and is_tuple:
            message = f"'{name}' must return a single value, found a tuple"
        elif expected > 1 and is_tuple and len(ret.value.elts) != expected:
            message = f"'{name}' must return {expected} values, found {len(ret.value.elts)}"
        else:
            continue
        yield ValidationIssue("api_changed", message, ret.lineno)


def _python_name(dummy: str) -> str:
    """Python spelling of a dummy argument: a keyword gets a trailing underscore (IN -> in_)."""
    return f"{dummy}_" if keyword.iskeyword(dummy) else dummy


def _api_issues(tree: ast.Module, symbols: Iterable[ApiSymbol]) -> list[ValidationIssue]:
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.setdefault(node.name, node)
    issues: list[ValidationIssue] = []
    for symbol in symbols:
        if symbol.kind in {"program", "block_data"}:
            continue
        func = functions.get(symbol.name)
        if func is None:
            message = f"{symbol.kind} '{symbol.name}' has no module-level function of that name"
            issues.append(ValidationIssue("api_changed", message))
            continue
        args = func.args
        actual = [a.arg for a in args.posonlyargs + args.args]
        expected = [_python_name(p.name) for p in symbol.params]
        extra = bool(args.vararg or args.kwarg or args.kwonlyargs)
        if actual != expected or extra:
            found = actual + (["*"] if extra else [])
            message = (
                f"'{symbol.name}' must take parameters ({', '.join(expected)}), "
                f"found ({', '.join(found)})"
            )
            issues.append(ValidationIssue("api_changed", message, func.lineno))
            continue
        issues.extend(_shape_issues(symbol, func))
    return issues


def validate_conversion(legacy: ParsedModule, converted_source: str) -> ValidationReport:
    """Statically check generated Python against the legacy Fortran module. Never raises."""
    if not isinstance(converted_source, str) or not converted_source.strip():
        return ValidationReport(
            False, [ValidationIssue("empty_output", "the converted module is empty")]
        )
    try:
        tree, error = _compile(converted_source)
        if tree is None or error is not None:
            return ValidationReport(False, [error] if error else [])
        issues, aliases = _import_issues(tree)
        issues.extend(_helper_attribute_issues(tree, aliases))
        issues.extend(_remnant_issues(tree))
        symbols = legacy.api or extract_api(legacy.source)
        issues.extend(_api_issues(tree, symbols))
    except Exception as exc:  # noqa: BLE001 - validation must report, never crash the pipeline
        message = f"validation failed: {type(exc).__name__}: {exc}"
        return ValidationReport(False, [ValidationIssue("syntax_error", message)])
    return ValidationReport(not issues, issues)
