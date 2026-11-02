"""Run generated or reference tests against a converted module in a throwaway directory.

Isolation here is process-level only: a scrubbed environment, a wall-clock timeout that kills
the whole process tree, and (POSIX) CPU and memory rlimits. Untrusted code still runs with the
caller's filesystem permissions, so production deployments rely on the container boundary.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path

from ..conventions import RUNTIME_MODULE
from ..models import TestRunResult
from ..runtime import runtime_source

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

MODULE_FILENAME = "target_module.py"
RUNTIME_FILENAME = f"{RUNTIME_MODULE}.py"
TEST_FILENAME = "test_target_module.py"
REPORT_FILENAME = "report.xml"
ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
)
THREAD_LIMIT_VARS: tuple[str, ...] = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CPU_LIMIT_FACTOR = 2
MEMORY_LIMIT_BYTES = 2 * 1024**3
_COLLECTION_EXIT_CODES = frozenset({2, 4})


def _scrubbed_env(workdir: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
    env["PYTHONPATH"] = str(workdir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Same text encoding on every platform, and always decodable by the reader of the pipe.
    env["PYTHONUTF8"] = "1"
    # One BLAS/OpenMP thread: deterministic results, and importing numpy under the POSIX address
    # space limit fails when OpenBLAS tries to reserve buffers for every core.
    for name in THREAD_LIMIT_VARS:
        env[name] = "1"
    return env


def _posix_limits(timeout_s: float) -> Callable[[], None] | None:
    if resource is None or not sys.platform.startswith(("linux", "darwin")):
        return None
    cpu_seconds = max(1, int(timeout_s * CPU_LIMIT_FACTOR) + 1)

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        # RLIMIT_AS is not enforced on macOS, so only set it where it is meaningful.
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))

    return apply


def _popen_kwargs(timeout_s: float) -> dict[str, object]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    kwargs: dict[str, object] = {"start_new_session": True}
    limits = _posix_limits(timeout_s)
    if limits is not None:
        kwargs["preexec_fn"] = limits
    return kwargs


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()


def _tail(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text[-limit:]


def _parse_junit(path: Path) -> tuple[int, int, int, int, bool] | None:
    """Return (passed, failed, errors, total, has_collection_error) or None if unreadable."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    cases = root.iter("testcase")
    passed = failed = errors = skipped = 0
    collection_error = False
    for case in cases:
        outcome = {child.tag for child in case}
        if "skipped" in outcome:
            skipped += 1
        elif "error" in outcome:
            errors += 1
            # Collection failures are reported by pytest as an error on a testcase whose
            # name is the module path rather than a test function.
            if not case.get("name", "").startswith("test"):
                collection_error = True
        elif "failure" in outcome:
            failed += 1
        else:
            passed += 1
    return passed, failed, errors, passed + failed + errors, collection_error


def run_tests(
    module_source: str,
    test_source: str,
    *,
    timeout_s: float = 60.0,
    max_output_chars: int = 8000,
    extra_files: Mapping[str, str] | None = None,
) -> TestRunResult:
    """Run ``test_source`` against ``module_source`` in a fresh pytest subprocess.

    The sandbox directory always receives the runtime helper module as ``f77rt.py`` (the module
    under test and the tests import it); ``extra_files`` are written afterwards and may add
    further files or replace the runtime. Counts come from pytest's junit XML. ``total``
    excludes skips. A syntax error in either file, a missing report, or a pytest usage/collection
    exit code sets ``collection_error``.
    """
    with tempfile.TemporaryDirectory(prefix="modernizer-tests-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        (workdir / MODULE_FILENAME).write_text(module_source, encoding="utf-8")
        (workdir / TEST_FILENAME).write_text(test_source, encoding="utf-8")
        (workdir / RUNTIME_FILENAME).write_text(runtime_source(), encoding="utf-8")
        for name, content in (extra_files or {}).items():
            target = (workdir / name).resolve()
            if workdir.resolve() not in target.parents:
                raise ValueError(f"extra file path escapes the sandbox directory: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return _execute(workdir, timeout_s, max_output_chars)


def run_test_file(
    module_source: str,
    test_file: Path,
    *,
    timeout_s: float = 60.0,
    max_output_chars: int = 8000,
    extra_files: Mapping[str, str] | None = None,
) -> TestRunResult:
    """Like ``run_tests`` but reads the test source from an existing file."""
    return run_tests(
        module_source,
        Path(test_file).read_text(encoding="utf-8"),
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        extra_files=extra_files,
    )


def _execute(workdir: Path, timeout_s: float, max_output_chars: int) -> TestRunResult:
    report = workdir / REPORT_FILENAME
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "--tb=short",
        f"--junitxml={report}",
        "-o",
        "addopts=",
        TEST_FILENAME,
    ]
    start = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=_scrubbed_env(workdir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_popen_kwargs(timeout_s),  # type: ignore[arg-type]
    )
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            output = ""
    duration = time.monotonic() - start

    parsed = None if timed_out else _parse_junit(report)
    if timed_out:
        return TestRunResult(
            timed_out=True,
            output=_tail(output or "", max_output_chars),
            duration_s=duration,
        )
    if parsed is None:
        return TestRunResult(
            collection_error=True,
            output=_tail(output or "", max_output_chars),
            duration_s=duration,
        )
    passed, failed, errors, total, xml_collection_error = parsed
    collection_error = xml_collection_error or proc.returncode in _COLLECTION_EXIT_CODES
    return TestRunResult(
        passed=passed,
        failed=failed,
        errors=errors,
        total=total,
        collection_error=collection_error,
        output=_tail(output or "", max_output_chars),
        duration_s=duration,
    )
