from __future__ import annotations

import ast
import warnings

from modernizer.conversion.merge import merge_sources


def test_future_first_imports_hoisted_deduped_and_bodies_in_order():
    sources = [
        "import numpy as np\n\n\ndef a():\n    return np.zeros(1)",
        "from __future__ import annotations\nimport numpy as np\nimport math\n\n\ndef b():\n    return math.pi",
    ]
    out = merge_sources(sources)
    assert out == (
        "from __future__ import annotations\nimport numpy as np\nimport math\n\n\n"
        "def a():\n    return np.zeros(1)\n\n\n"
        "def b():\n    return math.pi\n"
    )
    ast.parse(out)


def test_f77rt_imports_are_merged_into_one_line():
    sources = [
        "from f77rt import idiv\n\n\ndef a(n):\n    return idiv(n, 2)",
        "from f77rt import nint, idiv\n\n\ndef b(x):\n    return nint(x)",
        "from f77rt import sign as fsign\n\n\ndef c(x):\n    return fsign(x, 1.0)",
    ]
    out = merge_sources(sources)
    assert out.startswith("from f77rt import idiv, nint, sign as fsign\n\n\ndef a(n)")
    assert out.count("from f77rt import") == 1


def test_long_from_import_is_wrapped():
    names = [f"helper_function_{i}" for i in range(8)]
    out = merge_sources(
        [
            f"from f77rt import {', '.join(names[:4])}\nx = 1",
            f"from f77rt import {', '.join(names[4:])}\ny = 2",
        ]
    )
    assert out.startswith("from f77rt import (\n    helper_function_0,\n")
    ast.parse(out)


def test_star_import_absorbs_names():
    out = merge_sources(["from f77rt import *\nx = 1", "from f77rt import idiv\ny = 2"])
    assert out.startswith("from f77rt import *\n")
    assert out.count("from f77rt") == 1


def test_multiline_inline_and_nested_imports():
    sources = [
        "from os import (\n    path,\n    sep,\n)\n\n\ndef a():\n    import json\n    return path",
        "from os import path, sep\nx = 1; import re\n\n\ndef b():\n    return sep",
    ]
    out = merge_sources(sources)
    assert out.startswith("from os import path, sep\n\n\ndef a()")
    assert "    import json" in out
    assert "x = 1; import re" in out


def test_docstring_of_first_source_on_top_later_ones_dropped():
    sources = [
        '"""Port of DEMO."""\n\nimport math\n\n\ndef a():\n    return math.pi',
        '"""Second chunk."""\nimport math\n\n\ndef b():\n    return 1',
    ]
    out = merge_sources(sources)
    assert out.startswith('"""Port of DEMO."""\n\nimport math\n\n\ndef a()')
    assert "Second chunk" not in out


def test_unparseable_source_is_kept_in_place():
    out = merge_sources(["def a():\n    return 1", "def b(:\n    pass"])
    assert out.index("def a") < out.index("def b(:")


def test_prelude_names_are_not_redefined_by_later_sources():
    prelude = (
        "import numpy as np\n\n\nclass _Cnt:\n    ncalls = 0\n\n\ncnt = _Cnt()\n"
        "a_save, b_save = 1, 2\n"
    )
    chunk = (
        "import numpy as np\n\n\nclass _Cnt:\n    ncalls = 99\n\n\ncnt = _Cnt()\n\n\n"
        "def bump(n):\n    cnt.ncalls += n\n\n\na_save, b_save = 0, 0\nother = 5\n"
    )
    out = merge_sources([prelude, chunk], prelude=True)
    assert out.count("class _Cnt") == 1 and "ncalls = 99" not in out
    assert out.count("cnt = _Cnt()") == 1
    assert "a_save, b_save = 0, 0" not in out
    assert "other = 5" in out and "def bump(n):" in out
    assert out.index("class _Cnt") < out.index("def bump")
    assert out.count("import numpy as np") == 1


def test_without_prelude_flag_duplicates_are_kept():
    out = merge_sources(["x = 1", "x = 2"])
    assert "x = 1" in out and "x = 2" in out


def test_decorated_duplicate_is_removed_with_its_decorator():
    prelude = "def helper():\n    return 1\n"
    chunk = "@staticmethod\ndef helper():\n    return 2\n\n\ndef real():\n    return 3\n"
    out = merge_sources([prelude, chunk], prelude=True)
    assert "return 2" not in out and "@staticmethod" not in out and "def real" in out


def test_no_syntax_warnings_and_still_hoists():
    sources = [
        "import re\n\n\ndef a():\n    return re.compile('\\d+')",
        "import re\n\n\ndef b():\n    return re.compile('\\s+')",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = merge_sources(sources)
    assert out.count("import re") == 1


def test_main_guard_moves_behind_the_definitions_of_later_slices():
    program = (
        'import sys\n\n\ndef main():\n    helper()\n\n\nif __name__ == "__main__":\n    main()\n'
    )
    helper = "def helper():\n    return 1\n\n\nif '__main__' == __name__:\n    pass\n"
    out = merge_sources([program, helper])
    assert out.index("def helper") < out.index('if __name__ == "__main__":')
    assert out.rstrip().endswith("pass")
    assert out.count("__main__") == 2  # both guards kept, in source order, after all definitions
    assert out.index('if __name__ == "__main__":') < out.index("if '__main__' == __name__:")
    ast.parse(out)


def test_identical_main_guards_are_kept_once():
    guard = 'def f():\n    return 1\n\n\nif __name__ == "__main__":\n    f()\n'
    out = merge_sources([guard, guard.replace("def f", "def g").replace("f()", "g()")])
    assert out.count("if __name__") == 2
    same = 'if __name__ == "__main__":\n    f()\n'
    assert merge_sources([same, same]).count("if __name__") == 1


def test_other_if_statements_are_not_treated_as_main_guards():
    out = merge_sources(["if x == 1:\n    y = 2\n", "z = 3\n"])
    assert out.index("if x == 1") < out.index("z = 3")
