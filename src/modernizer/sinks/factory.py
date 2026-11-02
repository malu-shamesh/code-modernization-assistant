from __future__ import annotations

from ..protocols import ResultSink
from .bigquery import BigQuerySink
from .composite import CompositeSink, NullSink
from .jsonl import JsonlSink

_EXPECTED = "expected none, jsonl:<dir> or bigquery:<project>.<dataset>"


def _build_one(part: str, run_dir: str | None, include_source: bool) -> ResultSink | None:
    scheme, sep, arg = part.partition(":")
    scheme = scheme.strip().lower()
    arg = arg.strip()
    if scheme == "none" and not sep:
        return None
    if scheme == "jsonl":
        directory = arg or run_dir
        if not directory:
            raise ValueError("jsonl sink needs a directory: 'jsonl:<dir>' or pass run_dir")
        return JsonlSink(directory, include_source=include_source)
    if scheme == "bigquery":
        # rpartition: project ids may contain dots (domain-scoped), dataset names cannot.
        project, dot, dataset = arg.rpartition(".")
        if not (project and dot and dataset):
            raise ValueError(f"bad bigquery sink spec {part!r}; {_EXPECTED}")
        return BigQuerySink(project, dataset, include_source=include_source)
    raise ValueError(f"unknown sink spec {part!r}; {_EXPECTED}")


def sink_from_spec(
    spec: str | None, *, run_dir: str | None = None, include_source: bool = False
) -> ResultSink:
    """Build a sink from a spec string.

    ``None``/``"none"`` gives a ``NullSink``; ``"jsonl:<dir>"`` a ``JsonlSink`` (the directory
    defaults to ``run_dir`` when omitted); ``"bigquery:<project>.<dataset>"`` a ``BigQuerySink``.
    Comma-separated specs are combined with a ``CompositeSink``, so paths cannot contain commas.
    """
    if spec is None:
        return NullSink()
    sinks: list[ResultSink] = []
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            sink = _build_one(part, run_dir, include_source)
            if sink is not None:
                sinks.append(sink)
    except Exception:
        for built in sinks:
            built.close()
        raise
    if not sinks:
        return NullSink()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeSink(*sinks)
