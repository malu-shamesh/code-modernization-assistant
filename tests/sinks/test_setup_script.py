from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy" / "bigquery"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not available")

STUB_BQ = """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$BQ_LOG"
if [[ " $* " == *" mk "* && -n "${BQ_EXISTS:-}" ]]; then
  echo "BigQuery error in mk operation: already exists." >&2
  exit 1
fi
if [[ -n "${BQ_FAIL:-}" ]]; then
  echo "permission denied" >&2
  exit 1
fi
"""


def run_setup(
    tmp_path: Path, *args: str, **env: str
) -> tuple[subprocess.CompletedProcess, list[str]]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "bq"
    stub.write_bytes(STUB_BQ.encode())
    stub.chmod(0o755)
    log = tmp_path / "bq.log"
    log.write_text("")
    # CRLF views.sql, as produced by a Windows checkout with autocrlf
    deploy = tmp_path / "deploy"
    shutil.copytree(DEPLOY, deploy, dirs_exist_ok=True)
    views = deploy / "views.sql"
    views.write_bytes(views.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    full_env = {
        **os.environ,
        "PATH": f"{bindir.as_posix()}{os.pathsep}{os.environ['PATH']}",
        "BQ_LOG": log.as_posix(),
        **env,
    }
    proc = subprocess.run(
        [str(BASH), (deploy / "setup.sh").as_posix(), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc, log.read_text().splitlines()


def test_creates_dataset_tables_and_views(tmp_path: Path) -> None:
    proc, _ = run_setup(tmp_path, "my-proj", "my_ds")
    assert proc.returncode == 0, proc.stderr
    log = (tmp_path / "bq.log").read_text()
    assert "mk --dataset" in log
    assert "--time_partitioning_field timestamp" in log
    assert log.count("CREATE OR REPLACE VIEW `my-proj.my_ds.v_") == 4
    assert "${" not in log


def test_already_existing_objects_are_tolerated(tmp_path: Path) -> None:
    proc, _ = run_setup(tmp_path, "my-proj", "my_ds", BQ_EXISTS="1")
    assert proc.returncode == 0, proc.stderr
    assert "exists, skipping" in proc.stdout


def test_other_bq_failures_abort(tmp_path: Path) -> None:
    proc, _ = run_setup(tmp_path, "my-proj", "my_ds", BQ_FAIL="1")
    assert proc.returncode != 0
    assert "permission denied" in proc.stderr


@pytest.mark.parametrize(
    "args", [(), ("my-proj",), ("My Proj", "ds"), ("proj", "bad-name"), ("p|x", "ds")]
)
def test_invalid_arguments_are_rejected_before_calling_bq(tmp_path: Path, args: tuple) -> None:
    proc, calls = run_setup(tmp_path, *args, PROJECT="", DATASET="")
    assert proc.returncode == 2
    assert calls == []
