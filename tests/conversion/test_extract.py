from __future__ import annotations

import pytest

from modernizer.conversion import extract_code


def test_single_fenced_block():
    reply = "Here you go:\n```python\nprint('hi')\n```\nDone."
    assert extract_code(reply) == "print('hi')\n"


def test_longest_fenced_block_wins():
    reply = "```python\nx = 1\n```\ntext\n```py\ndef f():\n    return 1\n\nvalue = f()\n```\n"
    assert extract_code(reply) == "def f():\n    return 1\n\nvalue = f()\n"


def test_untagged_fence_is_accepted():
    assert extract_code("```\nx = 1\n```") == "x = 1\n"


def test_non_python_blocks_are_ignored():
    reply = "```bash\npip install a-very-long-package-name\n```\n```python\nx = 1\n```"
    assert extract_code(reply) == "x = 1\n"


def test_only_non_python_blocks_returns_empty():
    assert extract_code("```bash\nls -la\n```") == ""


def test_unterminated_fence_takes_rest_of_reply():
    reply = "```python\nimport os\n\ndef f():\n    return os.getcwd()\n"
    assert extract_code(reply) == "import os\n\ndef f():\n    return os.getcwd()\n"


def test_new_tagged_fence_closes_unterminated_one():
    reply = "```python\nx = 1\n```python\ny = 22222\n```"
    assert extract_code(reply) == "y = 22222\n"


def test_crlf_and_indented_fences():
    reply = "  ```python\r\nx = 1\r\n  ```\r\n"
    assert extract_code(reply) == "x = 1\n"


def test_unfenced_code_keeps_leading_indentation_structure():
    code = "import os\n\n\ndef f():\n    return 1\n"
    assert extract_code(code) == code


def test_prose_before_and_after_unfenced_code_is_stripped():
    reply = (
        "Here is the converted module:\n\n"
        "import os\n\n\ndef f():\n    return os.getcwd()\n\n"
        "This version keeps the public API intact."
    )
    assert extract_code(reply) == "import os\n\n\ndef f():\n    return os.getcwd()\n"


def test_code_that_starts_with_a_docstring_is_not_treated_as_prose():
    code = '"""Module docs."""\n\nx = 1\n'
    assert extract_code(code) == code


def test_code_lines_ending_with_colon_are_kept():
    code = "class A:\n    pass\n\nif True:\n    x = 1\nelse:\n    x = 2\n"
    assert extract_code(code) == code


def test_all_prose_returns_empty():
    assert extract_code("I am sorry, I cannot convert this file.") == ""


@pytest.mark.parametrize("text", ["", "   \n\n  ", "```python\n```", "```\n\n```"])
def test_empty_inputs(text):
    assert extract_code(text) == ""


def test_result_never_contains_fence_markers():
    replies = [
        "```python\nx = 1\n```",
        "```python\nx = 1\n",
        "x = 1\n```\n",
        "text ```python x = 1``` text",
        "```python\nx = '```'\n```",
    ]
    for reply in replies:
        assert "```" not in extract_code(reply)


def test_fortran_block_is_ignored_and_python_block_wins():
    reply = "```fortran\n      X = 1\n```\n```python\nx = 1\n```"
    assert extract_code(reply) == "x = 1\n"


def test_only_a_fortran_block_returns_empty():
    assert extract_code("```fortran\n      X = 1\n      END\n```") == ""


def test_numpy_module_with_f77rt_import_is_kept_verbatim():
    code = (
        "import numpy as np\nfrom f77rt import idiv, nint\n\n\n"
        "def half(n):\n    # C     comment kept\n    return idiv(n, 2)\n"
    )
    assert extract_code(f"Here:\n```python\n{code}```\nBye.") == code
