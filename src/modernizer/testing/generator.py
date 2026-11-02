"""LLM-driven pytest generation for a Python module converted from FORTRAN 77."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from ..conventions import CALLING_CONVENTION, RUNTIME_MODULE
from ..features import FEATURES
from ..models import ApiParam, ApiSymbol, ParsedModule, TokenUsage
from ..protocols import LLMClient
from .safety import TARGET_MODULE, TestGenerationError, extract_code, sanitize_tests

TEST_SYSTEM_PROMPT = (
    "You are a meticulous numerical-software test engineer. You write deterministic pytest "
    "tests for a Python (numpy) module that was converted from FORTRAN 77. The Fortran source "
    "is the behavioural specification; the converted module is what gets tested. Derive every "
    "expected value by simulating the Fortran semantics by hand, never by running or copying "
    "the converted code. Reply with exactly one fenced python code block and no other text."
)

_MAX_SEMANTIC_NOTES = 12
_CALLABLE_KINDS = frozenset({"subroutine", "function", "program"})
_TYPE_LABELS = {
    "integer": "INTEGER (int)",
    "real": "REAL (float, single precision in the Fortran source)",
    "double": "DOUBLE PRECISION (float)",
    "complex": "COMPLEX (complex)",
    "logical": "LOGICAL (bool)",
    "character": "CHARACTER (str)",
    "external": "EXTERNAL procedure (a Python callable following the same convention)",
    "unknown": "type not determined",
}
_ARRAY_DTYPES = {"integer": "dtype=int", "real": "dtype=float", "double": "dtype=float"}


@dataclass
class GeneratedTests:
    source: str
    usage: TokenUsage
    raw_response: str


def _number_lines(source: str) -> str:
    """Prefix each line with a ``NNNN| `` gutter; Fortran column 1 starts right after it."""
    lines = source.splitlines()
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{n:>{width}}| {line}" for n, line in enumerate(lines, start=1))


def _scalar_role(param: ApiParam, symbol: ApiSymbol) -> str:
    # ApiSymbol.returns alone decides the return shape (the converter prompt does the same), so
    # the role must not depend on intent or the contract could contradict its own return list.
    if param.name not in symbol.returns:
        return "input"
    if param.intent == "inout":
        return "OUTPUT: its input value is used and the updated value is returned"
    return "OUTPUT: assigned by the routine and returned; pass any placeholder value in"


def _describe_param(param: ApiParam, symbol: ApiSymbol) -> str:
    label = _TYPE_LABELS.get(param.type, param.type)
    if param.rank > 0:
        if param.type == "character":
            kind = f"rank-{param.rank} array of {label}: a Python list of str"
        else:
            dtype = _ARRAY_DTYPES.get(param.type)
            suffix = f", {dtype}" if dtype else ""
            kind = f"rank-{param.rank} array of {label}: numpy.ndarray{suffix}"
        if param.intent in ("out", "inout"):
            return f"{param.name}: {kind}; modified IN PLACE by the routine (never returned)"
        return f"{param.name}: {kind}; read only"
    if param.type == "external":
        return f"{param.name}: {label}"
    return f"{param.name}: {label} scalar; {_scalar_role(param, symbol)}"


def _return_elements(symbol: ApiSymbol) -> list[str]:
    types = {p.name: p.type for p in symbol.params}
    elements: list[str] = []
    if symbol.kind == "function":
        elements.append(f"function result ({symbol.result_type or 'unknown'})")
    elements.extend(f"{name} ({types.get(name, 'unknown')})" for name in symbol.returns)
    if symbol.alt_returns:
        elements.append("alternate-return number (int, 0 for a normal RETURN)")
    return elements


def _describe_returns(symbol: ApiSymbol) -> str:
    elements = _return_elements(symbol)
    if not elements:
        return "None"
    if len(elements) == 1:
        return f"the single value {elements[0]}"
    return "a tuple (" + ", ".join(elements) + ")"


def _describe_symbol(symbol: ApiSymbol) -> str:
    if symbol.kind == "block_data":
        return (
            f"BLOCK DATA {symbol.name}: initialises COMMON state at import time; "
            "it is not callable, do not test it directly"
        )
    if symbol.kind == "program":
        return (
            f"PROGRAM {symbol.name}: def main(), no arguments, returns None; observe its "
            "behaviour through capsys (standard output), a monkeypatched sys.stdin and files "
            "in tmp_path"
        )
    params = [p for p in symbol.params if p.name.isidentifier()]
    lines = [
        f"{symbol.kind.upper()} {symbol.name}: def {symbol.name}({', '.join(p.name for p in params)})"
    ]
    lines.extend(f"    {_describe_param(p, symbol)}" for p in params)
    if symbol.alt_returns:
        lines.append(
            f"    {symbol.alt_returns} alternate-return ('*') dummy argument(s): "
            "not Python parameters; the call reports which RETURN n was taken"
        )
    lines.append(f"    returns: {_describe_returns(symbol)}")
    return "\n".join(lines)


def render_api_contract(api: tuple[ApiSymbol, ...]) -> str:
    """Human-readable per-unit contract: signature, argument roles and the return shape."""
    return "\n".join(_describe_symbol(symbol) for symbol in api)


def _public_signatures(converted_source: str) -> list[str]:
    try:
        tree = ast.parse(converted_source)
    except SyntaxError:
        return []
    return [
        f"def {node.name}({ast.unparse(node.args)})"
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _describe_api(module: ParsedModule, converted_source: str) -> str:
    if module.api:
        return render_api_contract(module.api)
    signatures = _public_signatures(converted_source)
    if signatures:
        return "(Fortran contract unavailable; public functions of the converted module)\n" + (
            "\n".join(signatures)
        )
    return "(no public names could be determined)"


def _semantic_notes(module: ParsedModule) -> list[str]:
    specs = [FEATURES[fid] for fid in module.feature_ids if fid in FEATURES]
    semantic = [s for s in specs if s.severity == "semantic"]
    semantic.sort(key=lambda s: -s.weight)
    return [f"- {s.id}: {s.description}" for s in semantic[:_MAX_SEMANTIC_NOTES]]


def _coverage_phrase(module: ParsedModule, min_tests: int) -> str:
    names = [s.name for s in module.api if s.kind in _CALLABLE_KINDS]
    scope = f"program unit ({', '.join(names)})" if names else "program unit of the source"
    return f"at least one test per {scope} and at least {min_tests} test cases in total"


def build_test_prompt(module: ParsedModule, converted_source: str, *, min_tests: int = 8) -> str:
    """Prompt asking for pytest tests of ``converted_source`` specified by the Fortran source."""
    notes = _semantic_notes(module)
    notes_block = (
        "Constructs in this source whose behaviour is easy to get wrong:\n"
        + "\n".join(notes)
        + "\n\n"
        if notes
        else ""
    )
    return f"""Write a pytest test file for the module `{TARGET_MODULE}`.

FORTRAN 77 SOURCE (fixed form). This is the behavioural specification. Each line starts with a
`NNNN| ` gutter that is NOT part of the source: never reproduce it, and read the Fortran columns
(label in 1-5, continuation mark in 6, statements in 7-72, anything past column 72 is ignored)
from the first character after the gutter.
```fortran
{_number_lines(module.source)}
```

API CONTRACT of the converted Python module `{TARGET_MODULE}` (the implementation is hidden on purpose):
```
{_describe_api(module, converted_source)}
```

{CALLING_CONVENTION}
{notes_block}Requirements:
- Import the module under test as `import {TARGET_MODULE}` or `from {TARGET_MODULE} import ...`. Besides the standard library you may import only numpy, pytest and the runtime module (`from {RUNTIME_MODULE} import ...`, for example `FortranStop`). Nothing else.
- Write {_coverage_phrase(module, min_tests)}, as plain functions named `test_*`; `pytest.mark.parametrize` is welcome.
- Call the converted functions exactly as the contract and calling convention say: positional arguments in Fortran order, outputs taken from the return value (function result first, alternate-return number last), array arguments as numpy arrays that are checked after the call because they are modified in place.
- Cover edge cases: zero-trip DO loops, n = 1, negative numbers (truncating integer division, MOD with a negative operand, NINT of negative halves), zero values, empty or all-blank strings, strings longer than their CHARACTER*n length, and the smallest legal array sizes. Use only inputs for which the Fortran program is well defined: no subscript outside the declared bounds, no division by zero, no SQRT or LOG of a negative number.
- Derive every expected value by simulating the FORTRAN semantics step by step on paper, not from the converted code: integer division truncates toward zero, DO trip counts are max(0, (limit - start + step) / step) with truncating division, MOD takes the sign of the first argument, NINT rounds half away from zero, CHARACTER values are blank padded or truncated to their declared length and compare equal ignoring trailing blanks, array subscripts are 1-based in Fortran and 0-based in the numpy array (A(i, j) is a[i-1, j-1]; a lower bound other than 1 shifts the index), and SAVEd or COMMON state persists between calls.
- Floating point: REAL variables are single precision in the Fortran source, DOUBLE PRECISION is double. Compare REAL results with `pytest.approx(expected, rel=1e-6, abs=1e-9)` and DOUBLE PRECISION results with `pytest.approx(expected, rel=1e-12, abs=1e-12)`; for arrays use `numpy.testing.assert_allclose` with the same tolerances. Always give an absolute tolerance, because values near 0 cannot be matched relatively. Compare integers, logicals and strings with `==`, never with approx.
- Output and input: for PROGRAM units and WRITE/PRINT statements use `capsys` and assert only on output produced by an explicit FORMAT (a format string or FORMAT statement), comparing whole lines or fields. Never assert on the spacing of list-directed output (`WRITE(*,*)`, `PRINT *`); at most compare its whitespace-split tokens. There is no carriage-control character in the output. To feed READ from the standard unit, replace `sys.stdin` with an `io.StringIO` through `monkeypatch`. Files used by OPEN statements must live in `tmp_path` (use `monkeypatch.chdir(tmp_path)` or pass `tmp_path`-based names).
- STOP is expected to raise `FortranStop`: use `pytest.raises(FortranStop)` and check `.code` when the source gives one.
- COMMON blocks and SAVEd variables are module-level state of `{TARGET_MODULE}`: never read, write or reset them directly, only through the module's own functions. Make every test independent of the order in which tests run, by initialising such state with the module's own routines (for example the unit that sets the COMMON values) at the start of the test.
- Tests must be deterministic: no randomness, no network, no dependence on the current time or date, no subprocess, no `os.system`, no sockets.
- Write files only inside pytest's `tmp_path`. Never delete files or directories.
- Do not test private names that start with an underscore.
- Reply with exactly one ```python fenced block containing the complete test file and nothing else.
"""


def generate_tests(
    module: ParsedModule,
    converted_source: str,
    llm: LLMClient,
    *,
    min_tests: int = 8,
    max_output_tokens: int = 8192,
    tags: Mapping[str, str] | None = None,
) -> GeneratedTests:
    """Ask ``llm`` for tests, sanitize them, and return them with the token usage.

    Raises ``TestGenerationError`` (carrying the raw reply) when the reply is empty or the
    extracted code fails ``sanitize_tests``.
    """
    call_tags = {**(tags or {}), "stage": "test_gen"}
    response = llm.generate(
        build_test_prompt(module, converted_source, min_tests=min_tests),
        system=TEST_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        tags=call_tags,
    )
    raw = response.text or ""
    if not extract_code(raw):
        raise TestGenerationError("model returned no test code", raw)
    try:
        source = sanitize_tests(raw)
    except TestGenerationError as exc:
        raise TestGenerationError(str(exc), raw) from exc
    return GeneratedTests(source=source, usage=response.usage, raw_response=raw)
