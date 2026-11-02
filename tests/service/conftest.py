from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient

from modernizer.service import create_app
from modernizer.settings import Settings

from .fakes import Harness


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's own MODERNIZER_* / project settings must not leak into these tests."""
    for key in list(os.environ):
        if key.startswith("MODERNIZER_") or key == "GOOGLE_CLOUD_PROJECT":
            monkeypatch.delenv(key)


@pytest.fixture
def harness() -> Iterator[Harness]:
    h = Harness()
    yield h
    h.close()


@pytest.fixture
def make_client(harness: Harness) -> Iterator[Callable[..., TestClient]]:
    """Start the app (lifespan included) over the fake pipeline; env vars are read at creation."""
    with ExitStack() as stack:

        def make(settings: Settings | None = None) -> TestClient:
            app = create_app(settings or Settings(), pipeline_factory=harness)
            return stack.enter_context(TestClient(app, raise_server_exceptions=False))

        yield make


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
