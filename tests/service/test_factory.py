from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modernizer.service import __main__ as entrypoint
from modernizer.service import create_app

from .fakes import LEGACY


def test_create_app_from_env_with_mock_provider_starts_and_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``uvicorn --factory`` path: no arguments, everything from the environment."""
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("MODERNIZER_RETRIEVAL", "local")
    monkeypatch.setenv("MODERNIZER_TELEMETRY", "none")

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["provider"] == "mock"
        assert ready.json()["frontier_model"] == "mock-frontier"
        assert ready.json()["retrieval"] == "local"
        assert ready.json()["open_backend"] == "endpoint"

        response = client.post(
            "/v1/modernize", json={"files": [{"path": "sumsq.f", "source": LEGACY}]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["modules"] == 1
        (result,) = data["results"]
        assert result["success"] is False  # the mock model only ever answers a stub
        assert result["error"].startswith("validation failed")
        assert data["summary"]["planned_open_share"] is None

        analysis = client.post(
            "/v1/analyze", json={"files": [{"path": "sumsq.f", "source": LEGACY}]}
        )
        assert analysis.json()["files"][0]["routing"]["model"].startswith("mock-")


def test_invalid_configuration_fails_at_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_CONCURRENT", "zero")
    with pytest.raises(ValueError, match="MODERNIZER_MAX_CONCURRENT"):
        create_app()


def test_startup_failure_propagates_instead_of_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("MODERNIZER_RETRIEVAL", "vertex")  # needs MODERNIZER_VS_* which are unset
    with pytest.raises(ValueError, match="MODERNIZER_VS_INDEX"), TestClient(create_app()):
        pass


def test_vertex_endpoint_backend_without_an_endpoint_id_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default open backend needs a deployed endpoint; say so instead of failing per request."""
    monkeypatch.setenv("MODERNIZER_PROVIDER", "vertex")
    monkeypatch.setenv("MODERNIZER_RETRIEVAL", "local")
    monkeypatch.setenv("MODERNIZER_TELEMETRY", "none")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    with pytest.raises(ValueError, match="MODERNIZER_OPEN_ENDPOINT_ID"), TestClient(create_app()):
        pass


def test_open_fraction_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("MODERNIZER_OPEN_FRACTION", "0.7")
    with TestClient(create_app()) as client:
        assert client.get("/readyz").json()["open_fraction"] == 0.7


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "inf"])
def test_an_open_fraction_outside_zero_to_one_fails_at_creation(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Otherwise every complexity request would be a 500 while /readyz says ready."""
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("MODERNIZER_OPEN_FRACTION", value)
    with pytest.raises(ValueError, match="MODERNIZER_OPEN_FRACTION"):
        create_app()


def test_pricing_file_is_loaded_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n  mock-frontier:\n    input_per_mtok: 1.0\n    output_per_mtok: 2.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("MODERNIZER_PRICING_FILE", str(pricing))
    with caplog.at_level(logging.WARNING), TestClient(create_app()) as client:
        assert client.get("/readyz").status_code == 200
    unpriced = [r.getMessage() for r in caplog.records if "no pricing entry" in r.getMessage()]
    assert unpriced and "mock-open" in unpriced[0], "the open model is missing from the table"
    assert not any("mock-frontier" in m for m in unpriced)


def test_missing_pricing_file_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    monkeypatch.setenv("MODERNIZER_PRICING_FILE", str(tmp_path / "absent.yaml"))
    with pytest.raises(ValueError, match="MODERNIZER_PRICING_FILE"), TestClient(create_app()):
        pass


def test_main_serves_the_factory_on_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(entrypoint.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    monkeypatch.setenv("PORT", "9123")
    entrypoint.main()
    monkeypatch.delenv("PORT")
    entrypoint.main()

    assert [kw["port"] for _, kw in calls] == [9123, 8080]
    assert calls[0][0] == ("modernizer.service.app:create_app",)
    assert calls[0][1]["factory"] is True
    assert calls[0][1]["host"] == "0.0.0.0"


def test_main_rejects_a_bad_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "http")
    with pytest.raises(SystemExit, match="PORT"):
        entrypoint.main()
