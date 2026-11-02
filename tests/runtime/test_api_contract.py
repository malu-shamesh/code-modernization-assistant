"""The helper API listed in the task exists, is documented and the file stays standalone."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from modernizer.runtime import api_doc, public_names, runtime_source
from modernizer.runtime import f77rt as rt

REQUIRED_NAMES = [
    "idiv", "fmod", "sign", "nint", "anint", "aint", "int_", "dim", "ipow",
    "ichar", "char_", "index_", "len_trim", "lge", "lgt", "lle", "llt",
    "str_eq", "str_cmp", "char_assign", "substr", "set_substr",
    "do_trips", "do_range", "do_final",
    "FortranStop",
    "write_formatted", "write_list", "read_formatted", "read_list", "format_record",
    "open_unit", "close_unit", "rewind_unit", "backspace_unit", "endfile_unit",
    "write_internal", "read_internal",
    "iand", "ior", "ieor", "ishft", "not_",
    "d1mach", "r1mach", "i1mach",
    "cmplx", "zeros_f",
]  # fmt: skip

ALLOWED_IMPORTS = {
    "__future__", "atexit", "collections", "dataclasses", "decimal", "functools", "math",
    "numbers", "numpy", "operator", "os", "re", "struct", "sys", "tempfile", "threading", "typing",
}  # fmt: skip


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_required_name_exists(name):
    assert hasattr(rt, name)
    assert name in public_names()


def test_public_names_are_all_documented_in_api_doc():
    doc = api_doc()
    for name in public_names():
        assert name in doc


def test_every_public_function_has_docstring_and_full_type_hints():
    tree = ast.parse(runtime_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            assert doc and doc.splitlines()[0].strip(), node.name
            assert node.returns is not None, f"{node.name} lacks a return annotation"
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                args.append(node.args.vararg)
            for arg in args:
                assert arg.annotation is not None, f"{node.name}({arg.arg}) lacks a type hint"


def test_api_doc_has_a_summary_for_every_function():
    for line in api_doc().splitlines():
        head, _, summary = line.partition("  # ")
        assert summary.strip(), f"no summary for {head}"


def test_module_is_standalone():
    tree = ast.parse(runtime_source())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS


def test_imports_only_at_top_of_file():
    tree = ast.parse(runtime_source())
    top_level = {id(n) for n in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            assert id(node) in top_level


def test_source_has_no_em_dash():
    assert chr(0x2014) not in runtime_source()


def test_source_can_be_copied_and_imported_elsewhere(tmp_path: Path):
    target = tmp_path / "f77rt.py"
    target.write_text(runtime_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("f77rt_copy", target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["f77rt_copy"] = module
    try:
        spec.loader.exec_module(module)
        assert module.idiv(-7, 2) == -3
        assert module.format_record("(I5)", [-42]) == "  -42"
    finally:
        del sys.modules["f77rt_copy"]


def test_dunder_all_lists_exactly_the_public_names():
    assert set(rt.__all__) == set(public_names())
    assert len(rt.__all__) == len(set(rt.__all__))
