from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from modernizer.models import TestRunResult
from modernizer.testing import run_test_file, run_tests, runner

# Every call spawns a real pytest process, so scenarios are combined to keep the suite fast.
MODULE = "def add(a, b):\n    return a + b\n"
ADD_TEST = "from target_module import add\ndef test_a():\n    assert add(1, 2) == 3\n"
FAKE_RUNTIME = (
    "FAKE_RUNTIME_MARKER = 'fake'\n\n\n"
    "def idiv(a, b):\n"
    "    q = abs(a) // abs(b)\n"
    "    return q if (a < 0) == (b < 0) else -q\n"
)


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real f77rt is tested elsewhere; the sandbox tests only need something importable."""
    monkeypatch.setattr(runner, "runtime_source", lambda: FAKE_RUNTIME)


def test_runtime_module_is_always_written_into_the_sandbox():
    module = "from f77rt import idiv\n\n\ndef half(k):\n    return idiv(k, 2)\n"
    tests = (
        "import f77rt\n"
        "from pathlib import Path\n"
        "import target_module\n"
        "def test_truncating_division():\n"
        "    assert target_module.half(-7) == -3\n"
        "def test_runtime_file_is_the_module_text():\n"
        "    assert f77rt.FAKE_RUNTIME_MARKER == 'fake'\n"
        "    assert Path(f77rt.__file__).parent == Path(target_module.__file__).parent\n"
    )
    result = run_tests(module, tests)
    assert result.all_passed, result.output
    assert (result.passed, result.total) == (2, 2)


def test_extra_files_can_replace_the_runtime():
    module = "import f77rt\n\n\ndef marker():\n    return f77rt.MARKER\n"
    tests = "import target_module\ndef test_m():\n    assert target_module.marker() == 'custom'\n"
    result = run_tests(module, tests, extra_files={"f77rt.py": "MARKER = 'custom'\n"})
    assert result.all_passed, result.output


def test_numpy_arrays_work_inside_the_sandbox_process():
    module = (
        "import numpy as np\n"
        "from f77rt import idiv\n\n\n"
        "def scale(n, a, k):\n"
        "    for i in range(n):\n"
        "        a[i] = idiv(a[i], k)\n"
        "    return int(np.sum(a[:n]))\n"
    )
    tests = (
        "import numpy as np\n"
        "import pytest\n"
        "from target_module import scale\n"
        "def test_in_place_and_return():\n"
        "    a = np.array([7, -7, 9, 100], dtype=np.int64)\n"
        "    assert scale(3, a, 2) == 4\n"
        "    np.testing.assert_array_equal(a, [3, -3, 4, 100])\n"
        "def test_float_tolerance():\n"
        "    assert np.float64(0.1) + 0.2 == pytest.approx(0.3, rel=1e-12, abs=1e-12)\n"
    )
    result = run_tests(module, tests)
    assert result.all_passed, result.output
    assert (result.passed, result.total) == (2, 2)


def test_pass_counts_environment_and_extra_files(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    module = "from helper import twice\n\n\ndef f(x):\n    return twice(x)\n"
    tests = (
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "import target_module\n"
        "def test_f():\n    assert target_module.f(4) == 8\n"
        "def test_extra_data_file():\n"
        "    assert Path('data/values.txt').read_text(encoding='utf-8').strip() == 'v'\n"
        "def test_no_secrets():\n"
        "    leaked = [k for k in os.environ if k.startswith(('GOOGLE_', 'AWS_', 'GITHUB_'))]\n"
        "    assert leaked == []\n"
        "    assert 'super-secret' not in ' '.join(os.environ.values())\n"
        "def test_sandbox_flags():\n"
        "    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
        "    assert sys.flags.utf8_mode == 1\n"
        "    assert os.environ['OPENBLAS_NUM_THREADS'] == '1'\n"
        "    assert os.environ['OMP_NUM_THREADS'] == '1'\n"
        f"    assert sys.executable == {sys.executable!r}\n"
        "def test_stdin_is_closed():\n"
        "    assert sys.__stdin__.read() == ''\n"
        "@pytest.mark.skip(reason='excluded from totals')\n"
        "def test_skipped():\n    assert False\n"
    )
    extra = {"helper.py": "def twice(x):\n    return 2 * x\n", "data/values.txt": "v\n"}
    result = run_tests(module, tests, extra_files=extra)
    assert isinstance(result, TestRunResult)
    assert result.all_passed, result.output
    assert (result.passed, result.failed, result.errors, result.total) == (5, 0, 0, 5)
    assert not result.collection_error and not result.timed_out
    assert result.duration_s > 0
    assert "5 passed" in result.output


def test_failures_errors_and_non_ascii_output():
    tests = (
        "import pytest\n"
        "from target_module import add\n"
        "def test_ok():\n    assert add(1, 2) == 3\n"
        "def test_bad():\n    print('caf\\u00e9 \\u2713')\n    assert add(1, 2) == 4\n"
        "@pytest.fixture\n"
        "def broken():\n    raise RuntimeError('boom')\n"
        "def test_err(broken):\n    pass\n"
    )
    result = run_tests(MODULE, tests)
    assert (result.passed, result.failed, result.errors, result.total) == (1, 1, 1, 3)
    assert not result.all_passed
    assert not result.collection_error
    assert result.pass_rate == pytest.approx(1 / 3)


def test_unconverted_fortran_module_is_collection_error():
    fortran = "      SUBROUTINE ADD(A, B, C)\n      C = A + B\n      RETURN\n      END\n"
    result = run_tests(fortran, ADD_TEST)
    assert result.collection_error
    assert not result.all_passed
    assert result.passed == 0
    assert "IndentationError" in result.output


def test_syntax_error_in_tests_is_collection_error():
    result = run_tests(MODULE, "import target_module\ndef test_a(:\n    pass\n")
    assert result.collection_error
    assert not result.all_passed


def test_no_tests_collected_is_not_all_passed():
    result = run_tests(MODULE, "import target_module\nVALUE = 1\n")
    assert result.total == 0
    assert not result.collection_error
    assert not result.all_passed


def test_timeout_is_reported():
    tests = "import target_module\ndef test_loop():\n    while True:\n        pass\n"
    result = run_tests(MODULE, tests, timeout_s=3)
    assert result.timed_out
    assert not result.all_passed
    assert result.total == 0
    assert not result.collection_error


def test_kill_tree_terminates_grandchildren(tmp_path: Path):
    ready = tmp_path / "child_ready.txt"
    survived = tmp_path / "child_survived.txt"
    grandchild = (
        f"import time; open({str(ready)!r}, 'w').write('x'); time.sleep(5); "
        f"open({str(survived)!r}, 'w').write('x')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **runner._popen_kwargs(30.0),
    )
    try:
        deadline = time.monotonic() + 60
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert ready.exists(), "grandchild never started"
        runner._kill_tree(proc)
        proc.wait(timeout=30)
        time.sleep(6)
        assert not survived.exists()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_run_test_file_truncates_huge_output_to_the_tail(tmp_path: Path):
    path = tmp_path / "reference_tests.py"
    path.write_text(
        "import target_module\n"
        "def test_noisy():\n"
        "    for i in range(5000):\n"
        "        print('line', i)\n"
        "    assert False\n",
        encoding="utf-8",
    )
    result = run_test_file(MODULE, path, max_output_chars=500)
    assert len(result.output) == 500
    assert result.failed == 1
    assert "1 failed" in result.output
    assert path.exists()


def test_extra_files_cannot_escape_sandbox():
    with pytest.raises(ValueError):
        run_tests(MODULE, ADD_TEST, extra_files={"../evil.py": "x = 1\n"})
