from __future__ import annotations

import pytest

from modernizer.testing import TestGenerationError, sanitize_tests

HEADER = "import pytest\nimport target_module\n\n"
GOOD_BODY = "def test_x():\n    assert target_module\n"


def check(body: str) -> str:
    return sanitize_tests(HEADER + body)


def test_accepts_plain_tests_and_adds_trailing_newline():
    out = sanitize_tests(HEADER + GOOD_BODY.rstrip())
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert "def test_x" in out


def test_strips_python_fence_and_surrounding_prose():
    reply = f"Here you go:\n```python\n{HEADER}{GOOD_BODY}```\nHope it helps."
    out = sanitize_tests(reply)
    assert "```" not in out and "Here you go" not in out
    assert out.startswith("import pytest")


def test_strips_untagged_fence_and_handles_crlf():
    reply = f"```\n{HEADER}{GOOD_BODY}```".replace("\n", "\r\n")
    assert "def test_x" in sanitize_tests(reply)


def test_prefers_first_python_block():
    reply = f"```text\nignored\n```\n```python\n{HEADER}{GOOD_BODY}```\n```python\nx = 1\n```"
    assert "def test_x" in sanitize_tests(reply)


def test_unclosed_fence_with_complete_code_is_accepted():
    assert "def test_x" in sanitize_tests(f"```python\n{HEADER}{GOOD_BODY}")


def test_from_import_of_target_module():
    assert sanitize_tests("from target_module import f\n\ndef test_a():\n    assert f\n")


def test_aliased_import_of_target_module():
    assert sanitize_tests("import target_module as tm\n\ndef test_a():\n    assert tm\n")


def test_class_based_tests_are_accepted():
    src = "import target_module\n\nclass TestA:\n    def test_a(self):\n        assert 1\n"
    assert sanitize_tests(src)


@pytest.mark.parametrize(
    "source, fragment",
    [
        ("", "empty"),
        ("   \n", "empty"),
        ("```python\n```", "empty"),
        ("import target_module\ndef test_a(:\n    pass\n", "not valid Python"),
        ("import target_module\nprint 'hi'\n", "not valid Python"),
        ("def test_a():\n    assert True\n", "never import target_module"),
        ("import other_module\ndef test_a():\n    pass\n", "never import target_module"),
        ("import target_module\n\ndef helper():\n    pass\n", "no test_ function"),
    ],
)
def test_structural_rejections(source: str, fragment: str):
    with pytest.raises(TestGenerationError, match=fragment) as info:
        sanitize_tests(source)
    assert info.value.raw_response == source


@pytest.mark.parametrize(
    "body",
    [
        "import subprocess\n",
        "import subprocess as sp\n",
        "from subprocess import run\n",
        "import socket\n",
        "import ctypes\n",
        "import multiprocessing\n",
        "from multiprocessing import Pool\n",
        "import urllib.request\n",
        "from urllib import request\n",
        "from urllib.request import urlopen\n",
        "import requests\n",
        "import http.client\n",
        "from http import client\n",
        "from os import system\n",
        "from os import system as run_it\n",
        "from shutil import rmtree\n",
        "import os\n\ndef test_a():\n    os.system('ls')\n",
        "import os as o\n\ndef test_a():\n    o.system('ls')\n",
        "import os\n\ndef test_a():\n    os.popen('ls')\n",
        "import os\n\ndef test_a():\n    os.remove('x')\n",
        "import os\n\ndef test_a():\n    os.rmdir('x')\n",
        "import os\n\ndef test_a():\n    os.removedirs('x')\n",
        "import os\n\ndef test_a():\n    os.unlink('x')\n",
        "import os\n\ndef test_a():\n    os.execv('x', [])\n",
        "import os\n\ndef test_a():\n    os.spawnl(0, 'x')\n",
        "import shutil\n\ndef test_a():\n    shutil.rmtree('x')\n",
        "import shutil as sh\n\ndef test_a():\n    sh.rmtree('x')\n",
        "import os\n\ndef test_a():\n    f = os.system\n",
        "import urllib\n\ndef test_a():\n    urllib.request.urlopen('http://x')\n",
        "def test_a():\n    __import__('subprocess')\n",
        "def test_a():\n    __import__('os').system('ls')\n",
        "import importlib\n\ndef test_a():\n    importlib.import_module('os').remove('x')\n",
        "import importlib\n\ndef test_a():\n    importlib.import_module('socket')\n",
    ],
)
def test_denied_constructs(body: str):
    with pytest.raises(TestGenerationError, match="forbidden constructs"):
        check(body if "def test_" in body else body + GOOD_BODY)


@pytest.mark.parametrize(
    "body",
    [
        "def test_a():\n    eval(compile_me)\n",
        "def test_a():\n    exec(code)\n",
        "def test_a():\n    compile(src, 'f', 'exec')\n",
        "def test_a():\n    __import__(name)\n",
        "def test_a():\n    eval('1' + '+1')\n",
        "def test_a():\n    eval()\n",
    ],
)
def test_dynamic_code_with_non_literals_is_rejected(body: str):
    with pytest.raises(TestGenerationError, match="non-literal"):
        check(body)


@pytest.mark.parametrize(
    "body",
    [
        "import builtins\n\ndef test_a():\n    builtins.eval(src)\n",
        "import builtins\n\ndef test_a():\n    builtins.__import__(name)\n",
    ],
)
def test_dynamic_code_through_builtins_module_is_rejected(body: str):
    with pytest.raises(TestGenerationError, match="non-literal"):
        check(body)


def test_local_names_shadowing_denied_modules_are_allowed():
    body = (
        "def test_a():\n"
        "    requests = []\n"
        "    requests.append(1)\n"
        "    socket = {}\n"
        "    assert socket.get('k') is None\n"
        "    subprocess = [1]\n"
        "    assert subprocess.pop() == 1\n"
    )
    assert check(body)


def test_literal_eval_and_import_are_allowed():
    assert check("def test_a():\n    assert eval('1 + 1') == 2\n    __import__('json')\n")


@pytest.mark.parametrize(
    "body",
    [
        "def test_a():\n    open('out.txt', 'w')\n",
        "def test_a():\n    open('out.txt', 'a')\n",
        "def test_a():\n    open('out.txt', mode='x')\n",
        "def test_a():\n    open('out.txt', 'r+')\n",
        "def test_a():\n    open('out.txt', 'wb')\n",
        "def test_a(tmp_path):\n    open('/etc/passwd', 'w')\n",
        "def test_a():\n    mode = 'w'\n    open('out.txt', mode)\n",
        "import io\n\ndef test_a():\n    io.open('out.txt', 'w')\n",
        "from pathlib import Path\n\ndef test_a():\n    Path('x').open('w')\n",
        "def test_a(tmp_path):\n    other = 'x'\n    open(other, 'w')\n",
        "from pathlib import Path\n\ndef test_a():\n    Path('out.txt').write_text('x')\n",
        "from pathlib import Path\n\ndef test_a():\n    Path('out.bin').write_bytes(b'x')\n",
        "from pathlib import Path\n\ndef test_a():\n    Path('out.txt').unlink()\n",
        "from pathlib import Path\n\ndef test_a():\n    Path('some_dir').rmdir()\n",
    ],
)
def test_writes_outside_tmp_path_are_rejected(body: str):
    with pytest.raises(TestGenerationError, match="outside tmp_path"):
        check(body)


@pytest.mark.parametrize(
    "body",
    [
        "def test_a(tmp_path):\n    open(tmp_path / 'f.txt', 'w').close()\n",
        "def test_a(tmp_path):\n    with open(str(tmp_path / 'f.txt'), 'w', encoding='utf-8') as f:\n        f.write('x')\n",
        "def test_a(tmp_path):\n    p = tmp_path / 'f.txt'\n    open(p, mode='w').close()\n",
        "def test_a(tmp_path):\n    d = tmp_path / 'd'\n    p = d / 'f.txt'\n    with p.open('w') as f:\n        f.write('x')\n",
        "def test_a(tmp_path_factory):\n    p = tmp_path_factory.mktemp('x') / 'f'\n    open(p, 'wb').close()\n",
        "def test_a(tmp_path):\n    p = tmp_path / 'f.txt'\n    p.write_text('x')\n    (tmp_path / 'g').write_bytes(b'y')\n    p.unlink()\n    tmp_path.joinpath('d').mkdir()\n",
        "def test_a():\n    open('input.txt')\n    open('input.txt', 'r')\n    open('input.txt', 'rb')\n",
    ],
)
def test_reads_and_tmp_path_writes_are_allowed(body: str):
    assert check(body)


def test_all_violations_are_listed_with_line_numbers():
    with pytest.raises(TestGenerationError) as info:
        check("import subprocess\nimport socket\n" + GOOD_BODY)
    message = str(info.value)
    assert "subprocess" in message and "socket" in message and "line " in message


def test_error_carries_raw_response_including_fences():
    reply = "```python\nimport subprocess\nimport target_module\ndef test_a():\n    pass\n```"
    with pytest.raises(TestGenerationError) as info:
        sanitize_tests(reply)
    assert info.value.raw_response == reply


@pytest.mark.parametrize(
    "header",
    [
        "import numpy\n",
        "import numpy as np\n",
        "from numpy import linalg\n",
        "import numpy.testing\n",
        "from numpy.testing import assert_allclose\n",
        "from f77rt import idiv, FortranStop\n",
        "import f77rt\n",
        "import io\nimport math\nimport re\nimport sys\nimport tempfile\n",
        "from __future__ import annotations\n",
        "from os import path\n",
        "from pathlib import Path\n",
        "import importlib\n",
    ],
)
def test_allowed_imports(header: str):
    assert sanitize_tests(header + HEADER + GOOD_BODY)


@pytest.mark.parametrize(
    "header",
    [
        "import scipy\n",
        "from scipy import linalg\n",
        "import pandas as pd\n",
        "import torch\n",
        "import yaml\n",
        "from sklearn.metrics import r2_score\n",
        "import not_a_real_module.sub\n",
    ],
)
def test_third_party_imports_are_rejected(header: str):
    with pytest.raises(TestGenerationError, match="only the standard library") as info:
        sanitize_tests(header + HEADER + GOOD_BODY)
    assert header.split()[1].split(".")[0] in str(info.value)


@pytest.mark.parametrize("module", ["os", "numpy", "pytest", "shutil"])
def test_star_imports_of_other_modules_are_rejected(module: str):
    with pytest.raises(TestGenerationError, match="star import"):
        check(f"from {module} import *\n" + GOOD_BODY)


def test_star_import_of_target_and_runtime_modules_is_allowed():
    assert check("from target_module import *\nfrom f77rt import *\n" + GOOD_BODY)


def test_relative_import_is_rejected():
    with pytest.raises(TestGenerationError, match="relative imports"):
        check("from . import helper\n" + GOOD_BODY)


@pytest.mark.parametrize("name", ["scipy", "pandas.io"])
def test_dynamic_import_of_third_party_literal_is_rejected(name: str):
    body = f"def test_a():\n    __import__({name!r})\n"
    with pytest.raises(TestGenerationError, match="only the standard library"):
        check(body)


def test_dynamic_import_of_numpy_literal_is_allowed():
    assert check("def test_a():\n    assert __import__('numpy')\n")


@pytest.mark.parametrize(
    "body",
    [
        "import numpy.f2py\n",
        "from numpy import f2py\n",
        "import numpy.ctypeslib\n",
        "import numpy as np\n\ndef test_a():\n    np.ctypeslib.load_library('x', '.')\n",
        "import numpy as np\n\ndef test_a():\n    np.f2py.compile('x')\n",
    ],
)
def test_numpy_compiler_and_ffi_modules_are_denied(body: str):
    with pytest.raises(TestGenerationError, match="forbidden constructs"):
        check(body if "def test_" in body else body + GOOD_BODY)


@pytest.mark.parametrize(
    "body",
    [
        "import numpy as np\n\ndef test_a():\n    np.save('out.npy', [1])\n",
        "import numpy as np\n\ndef test_a():\n    np.savetxt('out.txt', [1])\n",
        "import numpy as np\n\ndef test_a():\n    np.savez('out.npz', a=[1])\n",
        "from numpy import savetxt\n\ndef test_a():\n    savetxt('/tmp/x', [1])\n",
        "import numpy as np\n\ndef test_a():\n    np.zeros(3).tofile('out.bin')\n",
    ],
)
def test_numpy_writes_outside_tmp_path_are_rejected(body: str):
    with pytest.raises(TestGenerationError, match="numpy writes outside tmp_path"):
        check(body)


def test_numpy_writes_into_tmp_path_are_allowed():
    body = (
        "import numpy as np\n\n"
        "def test_a(tmp_path):\n"
        "    np.savetxt(tmp_path / 'x.txt', [1.0])\n"
        "    np.save(str(tmp_path / 'y.npy'), np.zeros(2))\n"
        "    np.zeros(3).tofile(tmp_path / 'z.bin')\n"
        "    assert np.loadtxt(tmp_path / 'x.txt') == 1.0\n"
    )
    assert check(body)


def test_numpy_pickle_loading_is_rejected():
    body = "import numpy as np\n\ndef test_a(tmp_path):\n    np.load(tmp_path / 'x', allow_pickle=True)\n"
    with pytest.raises(TestGenerationError, match="allow_pickle"):
        check(body)


def test_typical_fortran_test_file_is_accepted():
    source = (
        "import io\n"
        "import numpy as np\n"
        "import pytest\n"
        "from f77rt import FortranStop\n"
        "import target_module\n\n"
        "def test_scale(monkeypatch, capsys):\n"
        "    a = np.array([1.0, 2.0])\n"
        "    monkeypatch.setattr('sys.stdin', io.StringIO('1\\n'))\n"
        "    with pytest.raises(FortranStop):\n"
        "        target_module.main()\n"
        "    np.testing.assert_allclose(a, [1.0, 2.0], rtol=1e-6, atol=1e-9)\n"
    )
    assert sanitize_tests(source)
