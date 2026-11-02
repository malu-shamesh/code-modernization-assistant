"""Checks on the deployment assets that can run offline: shell syntax, the configuration checks
of ``deploy.sh`` in dry-run mode, the Gemma helper script, and consistency between the files."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from modernizer.llm import PricingTable, amortised_price_per_mtok

ROOT = Path(__file__).resolve().parents[2]
CLOUDRUN = ROOT / "deploy" / "cloudrun"
GEMMA = ROOT / "deploy" / "vertex_gemma"
SCRIPTS = [
    CLOUDRUN / "deploy.sh",
    ROOT / "deploy" / "vector_search" / "provision.sh",
]
TEXT_ASSETS = [
    ROOT / "Dockerfile",
    ROOT / ".dockerignore",
    CLOUDRUN / "README.md",
    CLOUDRUN / "env.example",
    CLOUDRUN / "deploy.sh",
    GEMMA / "README.md",
    GEMMA / "gemma_endpoint.py",
    ROOT / "deploy" / "vector_search" / "README.md",
    *sorted((ROOT / "src" / "modernizer" / "service").glob("*.py")),
    *sorted(Path(__file__).parent.glob("*.py")),
]
OPEN_TIER_VARIABLES = [
    "MODERNIZER_OPEN_BACKEND",
    "MODERNIZER_OPEN_ENDPOINT_ID",
    "MODERNIZER_OPEN_BASE_URL",
    "MODERNIZER_OPEN_FOLD_SYSTEM",
    "MODERNIZER_OPEN_MODEL",
    "MODERNIZER_OPEN_FRACTION",
    "MODERNIZER_PRICING_FILE",
]
DEPLOY_VARIABLES = (
    "PROJECT_ID REGION SERVICE SA_NAME AR_REPO BQ_DATASET IMAGE IMAGE_TAG JSONL_SINK_DIR "
    "INVOKER_MEMBER PRICING_SECRET DRY_RUN ENV_FILE PROJECT"
).split()
EM_DASH = chr(0x2014)


def find_bash() -> str | None:
    """A bash that gets this process' environment and runs the scripts.

    The name is resolved with ``shutil.which`` and passed as a full path: on Windows a bare
    ``bash`` can resolve to the WSL launcher, which drops the environment variables.
    """
    path = shutil.which("bash")
    if path is None:
        return None
    probe = {**os.environ, "MODERNIZER_BASH_PROBE": "yes"}
    try:
        echo = subprocess.run(
            [path, "-c", 'printf %s "$MODERNIZER_BASH_PROBE"'],
            env=probe,
            capture_output=True,
            text=True,
            timeout=60,
        )
        helped = subprocess.run(
            [path, "deploy.sh", "--help"],
            cwd=CLOUDRUN,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return path if echo.stdout == "yes" and helped.returncode == 0 else None


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="a working bash is required")


def deploy_dry_run(**variables: str) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    env = {k: v for k, v in os.environ.items() if k not in DEPLOY_VARIABLES}
    env.update({"PROJECT_ID": "my-project-1", **variables})
    return subprocess.run(
        [BASH, "deploy.sh", "--dry-run"],
        cwd=CLOUDRUN,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def load_gemma_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gemma_endpoint", GEMMA / "gemma_endpoint.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gemma_endpoint"] = module
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep deploy/ free of __pycache__
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("gemma_endpoint", None)
        raise
    finally:
        sys.dont_write_bytecode = write_bytecode
    return module


@pytest.fixture
def gemma() -> Iterator[ModuleType]:
    module = load_gemma_script()
    yield module
    sys.modules.pop("gemma_endpoint", None)


@pytest.mark.parametrize("path", TEXT_ASSETS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dash_in_the_deployment_assets(path: Path) -> None:
    assert EM_DASH not in path.read_text(encoding="utf-8")


@needs_bash
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script: Path) -> None:
    done = subprocess.run([BASH, "-n", script.name], cwd=script.parent, capture_output=True)
    assert done.returncode == 0, done.stderr.decode(errors="replace")


@needs_bash
def test_default_open_backend_without_an_endpoint_id_fails_fast() -> None:
    done = deploy_dry_run()
    assert done.returncode != 0
    assert "MODERNIZER_OPEN_ENDPOINT_ID" in done.stderr
    assert "+ gcloud" not in done.stderr, "nothing may be attempted before the check"


@needs_bash
@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ({"MODERNIZER_OPEN_ENDPOINT_ID": "123"}, "MODERNIZER_OPEN_LOCATION"),
        ({"MODERNIZER_OPEN_BACKEND": "gpu"}, "MODERNIZER_OPEN_BACKEND"),
        ({"MODERNIZER_OPEN_BACKEND": "openai"}, "MODERNIZER_OPEN_BASE_URL"),
        ({"MODERNIZER_OPEN_BASE_URL": "ftp://host/v1"}, "MODERNIZER_OPEN_BASE_URL"),
        (
            {
                "MODERNIZER_OPEN_ENDPOINT_ID": "projects/1/endpoints/2",
                "MODERNIZER_OPEN_LOCATION": "a",
            },
            "MODERNIZER_OPEN_ENDPOINT_ID",
        ),
    ],
)
def test_inconsistent_open_tier_configuration_is_rejected(
    variables: dict[str, str], message: str
) -> None:
    done = deploy_dry_run(**variables)
    assert done.returncode != 0
    assert message in done.stderr


@needs_bash
@pytest.mark.parametrize("fraction", ["1.5", "-0.2", "most", "2", "1.01"])
def test_open_fraction_must_be_a_share(fraction: str) -> None:
    done = deploy_dry_run(
        MODERNIZER_OPEN_BASE_URL="https://gemma.example/v1", MODERNIZER_OPEN_FRACTION=fraction
    )
    assert done.returncode != 0
    assert "MODERNIZER_OPEN_FRACTION" in done.stderr


@needs_bash
def test_endpoint_configuration_is_forwarded_to_the_service() -> None:
    done = deploy_dry_run(
        MODERNIZER_OPEN_ENDPOINT_ID="1234567890",
        MODERNIZER_OPEN_LOCATION="us-central1",
        MODERNIZER_OPEN_FRACTION="0.7",
        MODERNIZER_OPEN_FOLD_SYSTEM="True",
        MODERNIZER_OPEN_MODEL="gemma-3-12b-it",
    )
    assert done.returncode == 0, done.stderr
    deploy = next(line for line in done.stderr.splitlines() if "gcloud run deploy" in line)
    for expected in (
        "MODERNIZER_OPEN_BACKEND=endpoint",
        "MODERNIZER_OPEN_ENDPOINT_ID=1234567890",
        "MODERNIZER_OPEN_LOCATION=us-central1",
        "MODERNIZER_OPEN_FRACTION=0.7",
        "MODERNIZER_OPEN_FOLD_SYSTEM=true",
        "MODERNIZER_OPEN_MODEL=gemma-3-12b-it",
    ):
        assert expected in deploy


@needs_bash
def test_another_backend_needs_no_endpoint_id() -> None:
    maas = deploy_dry_run(MODERNIZER_OPEN_BACKEND="maas")
    assert maas.returncode == 0, maas.stderr
    openai = deploy_dry_run(
        MODERNIZER_OPEN_BACKEND="openai", MODERNIZER_OPEN_BASE_URL="https://gemma.example/v1"
    )
    assert openai.returncode == 0, openai.stderr
    assert "MODERNIZER_OPEN_ENDPOINT_ID" not in openai.stderr


@needs_bash
def test_pricing_secret_is_mounted_and_readable_by_the_service_account() -> None:
    done = deploy_dry_run(
        MODERNIZER_OPEN_BACKEND="maas", PRICING_SECRET="my-pricing", PROJECT_ID="my-project-1"
    )
    assert done.returncode == 0, done.stderr
    assert "roles/secretmanager.secretAccessor" in done.stderr
    deploy = next(line for line in done.stderr.splitlines() if "gcloud run deploy" in line)
    assert "--set-secrets /etc/modernizer/pricing.yaml=my-pricing:latest" in deploy
    assert "MODERNIZER_PRICING_FILE=/etc/modernizer/pricing.yaml" in deploy


@needs_bash
def test_pricing_file_must_be_an_absolute_container_path() -> None:
    done = deploy_dry_run(MODERNIZER_OPEN_BACKEND="maas", MODERNIZER_PRICING_FILE="../pricing.yaml")
    assert done.returncode != 0
    assert "MODERNIZER_PRICING_FILE" in done.stderr


def test_new_variables_are_documented_where_users_look() -> None:
    sources = {
        "env.example": (CLOUDRUN / "env.example").read_text(encoding="utf-8"),
        "cloudrun README": (CLOUDRUN / "README.md").read_text(encoding="utf-8"),
        "deploy.sh": (CLOUDRUN / "deploy.sh").read_text(encoding="utf-8"),
    }
    for name in OPEN_TIER_VARIABLES:
        for label, text in sources.items():
            assert name in text, f"{name} is missing from {label}"


def test_dockerfile_covers_the_runtime_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert '"numpy' in core and '"pytest' in core, "the sandbox needs numpy and pytest"
    install = next(line for line in dockerfile.splitlines() if "pip install" in line)
    assert "gcp" in install and "serve" in install
    if "tree-sitter-fortran" in core:
        assert "tree_sitter_fortran" in dockerfile
    assert "tree-sitter-python" not in core
    assert "f77rt" in dockerfile
    assert "python:3" in dockerfile and "USER app" in dockerfile


def test_dockerignore_keeps_what_the_build_needs() -> None:
    entries = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for needed in ("src", "pyproject.toml", "LICENSE"):
        assert needed not in entries, "the image build needs it"
    assert {"tests", "benchmarks", "deploy", ".venv"} <= entries


def test_gemma_price_command_uses_the_pricing_formula(
    gemma: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    gemma.main(
        ["price", "--node-hour-usd", "3.5", "--tokens-per-second", "250", "--utilisation", "0.4"]
    )
    out = capsys.readouterr().out
    data = yaml.safe_load("\n".join(line for line in out.splitlines() if not line.startswith("#")))
    entry = data["models"]["gemma-3-12b-it"]
    expected = amortised_price_per_mtok(3.5, 250, 0.4)
    assert entry["input_per_mtok"] == pytest.approx(expected, abs=1e-6)
    assert entry["output_per_mtok"] == entry["input_per_mtok"]
    assert "gemma-3-12b-it" in PricingTable.default(), "the helper names an entry that exists"


@pytest.mark.parametrize(
    "args",
    [
        ["--node-hour-usd", "3", "--tokens-per-second", "0"],
        ["--node-hour-usd", "-1", "--tokens-per-second", "10"],
        ["--node-hour-usd", "3", "--tokens-per-second", "10", "--utilisation", "1.5"],
        ["--node-hour-usd", "nan", "--tokens-per-second", "10"],
    ],
)
def test_gemma_price_command_rejects_impossible_figures(gemma: ModuleType, args: list[str]) -> None:
    with pytest.raises(SystemExit, match="error"):
        gemma.main(["price", *args])


def test_gemma_deploy_dry_run_prints_the_request_and_calls_nothing(
    gemma: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gemma, "model_garden", None)  # any SDK use would now fail loudly
    gemma.main(
        [
            "deploy",
            "--project",
            "p",
            "--region",
            "r",
            "--machine-type",
            "MT",
            "--accelerator-type",
            "AT",
            "--accelerator-count",
            "2",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "machine_type = 'MT'" in out and "accelerator_count = 2" in out
    assert "bills per node hour" in out


def test_gemma_deploy_needs_every_hardware_choice_and_the_license(gemma: ModuleType) -> None:
    base = ["deploy", "--project", "p", "--region", "r"]
    with pytest.raises(SystemExit):
        gemma.main([*base, "--accelerator-type", "AT", "--accelerator-count", "1"])
    with pytest.raises(SystemExit, match="accept-eula"):
        gemma.main(
            [
                *base,
                "--machine-type",
                "MT",
                "--accelerator-type",
                "AT",
                "--accelerator-count",
                "1",
            ]
        )


def test_gemma_commands_need_the_sdk_and_a_project(
    gemma: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    for name in ("vertexai", "aiplatform", "model_garden"):
        monkeypatch.setattr(gemma, name, object())  # pretend the SDK is installed
    with pytest.raises(SystemExit, match="--project"):
        gemma.main(["describe", "--region", "r", "--endpoint-id", "1", "--project", ""])
    monkeypatch.setattr(gemma, "vertexai", None)
    with pytest.raises(SystemExit, match="SDK"):
        gemma.main(["models", "--project", "p", "--region", "r"])
