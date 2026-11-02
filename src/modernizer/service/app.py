"""FastAPI application for Cloud Run: Fortran 77 sources in, converted Python modules out.

Shared, expensive parts (model clients, retriever, telemetry, result sink) are built once in the
lifespan handler. Every ``/v1/modernize`` request gets its own ``Pipeline`` (own run id and usage
ledger) over those parts and runs it in a worker thread; telemetry and sinks are flushed before
the response is returned because Cloud Run may throttle the CPU between requests, which would
leave periodic exporters and buffered sink rows stuck.

Authentication: on Cloud Run use IAM (``--no-allow-unauthenticated``). ``MODERNIZER_API_KEY`` is
a shared secret for local and development use; it reads the same ``Authorization`` header that
Cloud Run IAM consumes, so do not combine the two on one service.
"""

from __future__ import annotations

import hmac
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .. import __version__
from ..factory import build_pipeline, build_retriever
from ..llm import ModelSet, PricingTable, build_model_set
from ..parsing import analyze
from ..pipeline import Pipeline
from ..protocols import ResultSink
from ..report import summarize
from ..retrieval import Retriever
from ..routing import Router, difficulty_score, explain_difficulty
from ..settings import Settings
from ..sinks import sink_from_spec
from ..telemetry import Telemetry, setup_telemetry
from .limits import ConcurrencyGate, LimitExceeded, Limits, Slot
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    FileAnalysis,
    ModernizeRequest,
    ModernizeResponse,
    ModuleResultOut,
)

logger = logging.getLogger(__name__)

API_PREFIX = "/v1"
GENERIC_ERROR = "internal server error"


@dataclass
class _Shared:
    models: ModelSet
    retriever: Retriever
    telemetry: Telemetry
    sink: ResultSink


@dataclass
class _State:
    settings: Settings
    limits: Limits
    gate: ConcurrencyGate
    api_key: str | None
    pipeline_factory: Callable[[], Pipeline] | None
    ready: bool = False
    shared: _Shared | None = None
    frontier_name: str = ""
    open_name: str = ""
    router: Router | None = field(default=None, repr=False)


def _build_shared(settings: Settings) -> _Shared:
    with ExitStack() as cleanup:
        telemetry = setup_telemetry(
            settings.telemetry,
            otlp_endpoint=settings.otlp_endpoint,
            project_id=settings.models.project,
        )
        cleanup.callback(telemetry.shutdown)
        models = build_model_set(settings.models)
        _check_pricing(models.frontier.name, models.open.name)
        retriever = build_retriever(settings)
        sink = sink_from_spec(settings.sink)
        cleanup.callback(sink.close)
        shared = _Shared(models, retriever, telemetry, sink)
        cleanup.pop_all()
    return shared


def _check_pricing(*model_names: str) -> None:
    """Fail at startup on an unreadable pricing file; warn about models it cannot price.

    ``MODERNIZER_PRICING_FILE`` is otherwise only read when the first cost is computed, which
    would turn a wrong path into a failure in the middle of the first request.
    """
    try:
        table = PricingTable.default()
    except Exception as exc:
        raise ValueError(
            f"the pricing table cannot be loaded (check MODERNIZER_PRICING_FILE): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for name in model_names:
        if name not in table:
            logger.warning("no pricing entry for model %r: its cost will be reported as 0", name)


def _check_open_fraction(settings: Settings) -> None:
    """Fail at creation on a target share outside [0, 1].

    ``Settings.from_env`` only converts the number. Left alone, the error would surface in
    ``Router.calibrated_for_batch`` and turn every ``complexity`` request into a 500 while
    ``/readyz`` reports ready.
    """
    fraction = settings.pipeline.target_open_fraction
    if fraction is not None and not 0.0 <= fraction <= 1.0:
        raise ValueError(f"MODERNIZER_OPEN_FRACTION must be a number from 0 to 1, got {fraction!r}")


def _flush_sink(sink: ResultSink) -> None:
    """Flush buffered rows. ``ResultSink`` has no flush, so use it where a sink offers one."""
    for child in getattr(sink, "sinks", ()):
        _flush_sink(child)
    flush = getattr(sink, "flush", None)
    if callable(flush):
        flush()


def _close_shared(shared: _Shared) -> None:
    for name, step in (
        ("flush sink", lambda: _flush_sink(shared.sink)),
        ("close sink", shared.sink.close),
        ("shut down telemetry", shared.telemetry.shutdown),
    ):
        try:
            step()
        except Exception:
            logger.exception("failed to %s", name)


def _flush(pipeline: Pipeline) -> None:
    """Best effort: a failing exporter must not turn a finished conversion into an error."""
    for name, step in (
        ("telemetry", pipeline.telemetry.flush),
        ("sink", lambda: _flush_sink(pipeline.sink)),
    ):
        try:
            step()
        except Exception:
            logger.exception("flushing %s failed (run %s)", name, pipeline.run_id)


def _configure_logging() -> None:
    """Make INFO logs visible when started by uvicorn, which only configures its own loggers."""
    root = logging.getLogger()
    if root.handlers:
        return
    level = getattr(logging, os.environ.get("MODERNIZER_LOG_LEVEL", "INFO").strip().upper(), None)
    logging.basicConfig(
        level=level if isinstance(level, int) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _read_api_key() -> str | None:
    return os.environ.get("MODERNIZER_API_KEY", "").strip() or None


def _error(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def _bearer_ok(request: Request, key: str) -> bool:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return False
    # Header values arrive decoded as latin-1, so go back to the bytes the client sent.
    return hmac.compare_digest(token.strip().encode("latin-1"), key.encode("utf-8"))


def _state(request: Request) -> _State:
    state: _State = request.app.state.service
    return state


def _require_ready(request: Request) -> _State:
    state = _state(request)
    if not state.ready:
        raise HTTPException(status_code=503, detail="service is not ready")
    return state


def _enforce_limits(state: _State, body: AnalyzeRequest) -> None:
    try:
        state.limits.check_sources(body.pairs())
    except LimitExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None


def _derived_router(current: Router, body: ModernizeRequest) -> Router:
    """The request's router: ``current`` with the policy and threshold overrides applied.

    Everything else (models, forced-frontier rule and feature set, difficulty weights) is carried
    over from ``current``. ``Router`` is shared between threads, so this builds a new one.
    """
    router = current
    if body.policy is not None and body.policy != router.policy:
        router = Router(
            router.frontier_model,
            router.open_model,
            policy=body.policy,
            threshold=router.threshold,
            force_frontier_on_features=router.force_frontier_on_features,
            force_frontier_features=router.force_frontier_features,
            weights=router.weights,
        )
    if body.threshold is not None:
        router = router.with_threshold(body.threshold)
    return router


def _apply_overrides(pipeline: Pipeline, body: ModernizeRequest) -> None:
    """Per-request settings; safe because the pipeline is private to this request."""
    pipeline.router = _derived_router(pipeline.router, body)
    config = replace(pipeline.config, run_tests=body.run_tests)
    if body.open_fraction is not None:
        config = replace(config, target_open_fraction=body.open_fraction)
    pipeline.config = config


def _new_pipeline(state: _State) -> Pipeline:
    if state.pipeline_factory is not None:
        return state.pipeline_factory()
    shared = state.shared
    if shared is None:
        raise RuntimeError("shared components are not initialised")
    return build_pipeline(
        state.settings,
        models=shared.models,
        retriever=shared.retriever,
        telemetry=shared.telemetry,
        sink=shared.sink,
    )


def _execute(
    state: _State, body: ModernizeRequest, slot: Slot
) -> tuple[str, list[ModuleResultOut], dict[str, float | int | None]]:
    """Runs in a worker thread and owns the concurrency slot until it is done.

    The slot is released here, so a client that disconnects (which can cancel the handler) does
    not free it while the pipeline is still running.
    """
    pipeline: Pipeline | None = None
    try:
        pipeline = _new_pipeline(state)
        _apply_overrides(pipeline, body)
        results = pipeline.run({f.path: f.source for f in body.files})
        summary: dict[str, float | int | None] = {**summarize(results).to_dict()}
        summary["planned_open_share"] = pipeline.planned_open_share
        out = [ModuleResultOut.from_result(r, include_tests=body.include_tests) for r in results]
        logger.info(
            "run %s finished: %d modules, %d succeeded",
            pipeline.run_id,
            summary["modules"],
            summary["succeeded"],
        )
        return pipeline.run_id, out, summary
    finally:
        try:
            if pipeline is not None:
                _flush(pipeline)
        finally:
            slot.release()


def _analyze_all(state: _State, body: AnalyzeRequest) -> list[FileAnalysis]:
    router = state.router
    if router is None:
        raise RuntimeError("router is not initialised")
    files: list[FileAnalysis] = []
    for f in body.files:
        try:
            parsed = analyze(f.path, f.source)
            files.append(
                FileAnalysis.from_parsed(
                    parsed,
                    score=difficulty_score(parsed, router.weights),
                    reasons=explain_difficulty(parsed, router.weights),
                    decision=router.route(parsed),
                )
            )
        except Exception:
            logger.exception("analysis of %s failed", f.path)
            files.append(FileAnalysis(path=f.path, error="analysis failed"))
    return files


def _open_backend(settings: Settings) -> str:
    """How the open tier is reached: ``endpoint``, ``maas`` or ``openai``.

    A configured base URL wins over the backend setting (see ``llm.ModelConfig``), which makes
    it an OpenAI-compatible server. Only this label is reported, never the URL or endpoint id.
    """
    models = settings.models
    return "openai" if models.open_base_url else models.open_backend


router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    state = _state(request)
    if not state.ready:
        return _error(503, "starting")
    settings = state.settings
    return JSONResponse(
        {
            "status": "ready",
            "provider": settings.models.provider,
            "frontier_model": state.frontier_name,
            "open_model": state.open_name,
            "open_backend": _open_backend(settings),
            "retrieval": settings.retrieval,
            "telemetry": settings.telemetry,
            "policy": settings.policy,
            "threshold": settings.threshold,
            "open_fraction": settings.pipeline.target_open_fraction,
        }
    )


@router.post(
    f"{API_PREFIX}/modernize",
    response_model=ModernizeResponse,
    response_model_exclude_unset=True,
)
async def modernize(body: ModernizeRequest, request: Request) -> ModernizeResponse:
    state = _require_ready(request)
    _enforce_limits(state, body)
    slot = state.gate.acquire_slot()
    if slot is None:
        raise HTTPException(
            status_code=429,
            detail="too many concurrent requests, retry later",
            headers={"Retry-After": str(state.limits.retry_after_s)},
        )
    try:
        run_id, results, summary = await run_in_threadpool(_execute, state, body, slot)
    finally:
        # A no-op once the worker released it. It matters only when the worker never started
        # (for example the handler was cancelled while waiting for a free thread), which would
        # otherwise leak the slot for the life of the process. A started worker is not
        # interrupted by cancellation, so this cannot free a slot that is still in use.
        slot.release()
    return ModernizeResponse(run_id=run_id, summary=summary, results=results)


@router.post(f"{API_PREFIX}/analyze", response_model=AnalyzeResponse)
async def analyze_files(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    state = _require_ready(request)
    _enforce_limits(state, body)
    files = await run_in_threadpool(_analyze_all, state, body)
    return AnalyzeResponse(files=files)


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_factory: Callable[[], Pipeline] | None = None,
) -> FastAPI:
    """Build the application.

    ``pipeline_factory`` replaces the per-request pipeline construction (and, with it, the
    shared model/retriever/telemetry/sink set, which is then not built at all); tests use it to
    run without any cloud access. Usable as ``uvicorn --factory modernizer.service.app:create_app``.
    """
    settings = settings or Settings.from_env()
    _check_open_fraction(settings)
    limits = Limits.from_env()
    state = _State(
        settings=settings,
        limits=limits,
        gate=ConcurrencyGate(limits.max_concurrent),
        api_key=_read_api_key(),
        pipeline_factory=pipeline_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _configure_logging()
        names = (settings.models.frontier_model, settings.models.open_model)
        if pipeline_factory is None:
            state.shared = await run_in_threadpool(_build_shared, settings)
            names = (state.shared.models.frontier.name, state.shared.models.open.name)
        state.frontier_name, state.open_name = names
        state.router = Router(
            *names,
            policy=settings.policy,
            threshold=settings.threshold,
        )
        state.ready = True
        logger.info(
            "service ready (provider=%s, retrieval=%s)",
            settings.models.provider,
            settings.retrieval,
        )
        try:
            yield
        finally:
            state.ready = False
            if state.shared is not None:
                await run_in_threadpool(_close_shared, state.shared)

    app = FastAPI(title="code-modernization-assistant", version=__version__, lifespan=lifespan)
    app.state.service = state
    app.include_router(router)

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Any]) -> Response:
        # One place for auth, the pre-read size check and the catch-all, so no failure mode can
        # leak a traceback and auth runs before any request body is parsed.
        try:
            path = request.url.path
            if path == API_PREFIX or path.startswith(API_PREFIX + "/"):
                if state.api_key is not None and not _bearer_ok(request, state.api_key):
                    return _error(
                        401, "missing or invalid bearer token", {"WWW-Authenticate": "Bearer"}
                    )
                declared = request.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > limits.max_body_bytes:
                    return _error(413, "request body is too large")
            response: Response = await call_next(request)
            return response
        except Exception:
            error_id = uuid.uuid4().hex[:12]
            logger.exception(
                "unhandled error on %s %s (error_id=%s)", request.method, request.url.path, error_id
            )
            return JSONResponse({"detail": GENERIC_ERROR, "error_id": error_id}, status_code=500)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default handler echoes the offending input, which here is user source code.
        errors = [
            {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    return app
