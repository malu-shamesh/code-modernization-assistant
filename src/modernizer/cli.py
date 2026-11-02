"""Command line interface: ``modernize analyze | convert | eval | index | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .evaluation import (
    PolicySpec,
    calibrate_open_threshold,
    default_policies,
    load_corpus,
    oracle_model_set,
    run_evaluation,
    split_calibration,
)
from .factory import build_pipeline, build_retriever, build_router
from .fileio import collect_sources, write_output
from .llm import ModelSet, build_model_set
from .parsing import analyze
from .report import format_summary, summarize
from .retrieval import Retriever, provision_vector_search
from .routing import difficulty_score, explain_difficulty
from .runtime import RUNTIME_FILENAME, runtime_source
from .settings import Settings
from .sinks import sink_from_spec
from .telemetry import setup_telemetry

try:
    import uvicorn
except ImportError:
    uvicorn = None

POLICIES = ("always_frontier", "always_open", "complexity")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modernize",
        description="AST-guided LLM conversion of Fortran 77 code to Python.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="show legacy constructs and routing for files (no LLM)")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    p.add_argument("--threshold", type=float, help="complexity routing threshold")
    p.set_defaults(handler=_cmd_analyze)

    p = sub.add_parser("convert", help="convert files or directories")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    _add_run_options(p)
    p.add_argument("--no-tests", action="store_true", help="validate statically, skip the tests")
    p.add_argument("--write-tests", action="store_true", help="also write generated tests")
    p.add_argument("--report", type=Path, help="write a JSON report here")
    p.set_defaults(handler=_cmd_convert)

    p = sub.add_parser("eval", help="run the benchmark corpus under routing policies")
    p.add_argument("--corpus", type=Path, default=Path("benchmarks/f77"))
    p.add_argument("--policies", help=f"comma separated subset of {', '.join(POLICIES)}")
    p.add_argument("--ids", help="comma separated benchmark ids")
    p.add_argument("--limit", type=int, help="use only the first N cases")
    p.add_argument(
        "--dry-run", action="store_true", help="oracle stub models, proves plumbing only"
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate the threshold on half of the cases and evaluate on the other half",
    )
    p.add_argument("--min-pass-rate", type=float, default=0.9, help="used with --calibrate")
    p.add_argument("--no-tests", action="store_true", help="skip generated tests")
    p.add_argument("--json", type=Path, dest="json_out", help="write the report as JSON")
    p.add_argument("--markdown", type=Path, help="write the report as Markdown")
    _add_run_options(p)
    p.set_defaults(handler=_cmd_eval)

    index = sub.add_parser("index", help="manage the Vertex AI Vector Search index")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    p = index_sub.add_parser("provision", help="create and deploy a streaming index (billable)")
    p.add_argument("--project")
    p.add_argument("--location", default="us-central1")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--display-name", default="modernizer-kb")
    p.set_defaults(handler=_cmd_index_provision)
    p = index_sub.add_parser("build", help="embed the knowledge base and upsert it")
    p.set_defaults(handler=_cmd_index_build)

    p = sub.add_parser("serve", help="run the HTTP service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(handler=_cmd_serve)
    return parser


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", choices=POLICIES, help="routing policy (default: env or complexity)")
    p.add_argument("--threshold", type=float, help="complexity routing threshold")
    p.add_argument("--provider", choices=("vertex", "mock"))
    p.add_argument("--retrieval", choices=("local", "vertex"))
    p.add_argument("--telemetry", choices=("none", "console", "otlp", "gcp"))
    p.add_argument("--sink", help="jsonl:<dir>, bigquery:<project>.<dataset>, comma separated")
    p.add_argument("--workers", type=int, help="modules converted concurrently")
    p.add_argument("--max-repairs", type=int, help="repair rounds per tier")
    p.add_argument(
        "--open-fraction",
        type=float,
        help="complexity policy: route about this share of the batch (0-1) to the open model",
    )


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "provider", None):
        settings = replace(settings, models=replace(settings.models, provider=args.provider))
    if getattr(args, "retrieval", None):
        settings = replace(settings, retrieval=args.retrieval)
    if getattr(args, "telemetry", None):
        settings = replace(settings, telemetry=args.telemetry)
    if getattr(args, "sink", None):
        settings = replace(settings, sink=args.sink)
    if getattr(args, "policy", None):
        settings = replace(settings, policy=args.policy)
    if getattr(args, "threshold", None) is not None:
        settings = replace(settings, threshold=args.threshold)
    pipeline = settings.pipeline
    if getattr(args, "workers", None):
        pipeline = replace(pipeline, max_workers=args.workers)
    if getattr(args, "max_repairs", None) is not None:
        pipeline = replace(pipeline, max_repairs=args.max_repairs)
    if getattr(args, "open_fraction", None) is not None:
        if not 0.0 <= args.open_fraction <= 1.0:
            raise ValueError("--open-fraction must be between 0 and 1")
        pipeline = replace(pipeline, target_open_fraction=args.open_fraction)
    if getattr(args, "no_tests", False):
        pipeline = replace(pipeline, run_tests=False)
    return replace(settings, pipeline=pipeline)


def _cmd_analyze(args: argparse.Namespace) -> int:
    settings = _settings(args)
    models = build_model_set(replace(settings.models, provider="mock"))
    router = build_router(settings, models)
    sources = collect_sources(args.paths)
    if not sources:
        raise ValueError("no Fortran files (.f, .for, .f77, .ftn) found")
    rows = []
    for path, source in sources.items():
        module = analyze(path, source)
        decision = router.route(module)
        rows.append(
            {
                "path": path,
                "sloc": module.metrics.sloc,
                "hits": module.metrics.n_feature_hits,
                "features": module.feature_ids,
                "difficulty": round(difficulty_score(module), 3),
                "tier": decision.tier.value,
                "reasons": explain_difficulty(module)[:3],
            }
        )
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0
    width = max(len(r["path"]) for r in rows)
    print(f"{'module'.ljust(width)}  sloc  hits  difficulty  tier")
    for r in rows:
        print(
            f"{r['path'].ljust(width)}  {r['sloc']:>4}  {r['hits']:>4}  "
            f"{r['difficulty']:>10.3f}  {r['tier']}"
        )
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    settings = _settings(args)
    sources = collect_sources(args.paths)
    if not sources:
        raise ValueError("no Fortran files (.f, .for, .f77, .ftn) found")
    telemetry = setup_telemetry(
        settings.telemetry, otlp_endpoint=settings.otlp_endpoint, project_id=settings.models.project
    )
    sink = sink_from_spec(settings.sink)
    try:
        pipeline = build_pipeline(settings, telemetry=telemetry, sink=sink)
        results = pipeline.run(sources)
    finally:
        telemetry.shutdown()
        sink.close()

    out: Path = args.out
    write_output(out, RUNTIME_FILENAME, runtime_source())
    for result in results:
        if result.converted_source:
            python_path = Path(result.path).with_suffix(".py").as_posix()
            target = python_path if result.success else f"_failed/{python_path}"
            write_output(out, target, result.converted_source)
        if args.write_tests and result.generated_tests:
            write_output(out, f"_tests/test_{Path(result.path).stem}.py", result.generated_tests)
        status = "ok  " if result.success else "FAIL"
        note = f"  ({result.error})" if result.error else ""
        print(f"{status} {result.path}  tier={result.final_tier.value}{note}")
    summary = summarize(results)
    print()
    print(format_summary(summary))
    if pipeline.planned_open_share is not None:
        print(
            f"planned open share: {pipeline.planned_open_share:.1%} (threshold {pipeline.router.threshold:.3f})"
        )
    if args.report:
        payload = {
            "run_id": pipeline.run_id,
            "summary": summary.to_dict(),
            "modules": [
                {
                    "path": r.path,
                    "success": r.success,
                    "routed_tier": r.routing.tier.value,
                    "final_tier": r.final_tier.value,
                    "escalated": r.escalated,
                    "difficulty": r.routing.difficulty,
                    "attempts": len(r.attempts),
                    "cost_usd": r.cost_usd,
                    "error": r.error,
                }
                for r in results
            ],
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if summary.succeeded == summary.modules else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    cases = load_corpus(args.corpus, ids=ids, limit=args.limit)
    if not cases:
        raise ValueError(f"no benchmark cases found under {args.corpus}")

    models: ModelSet = oracle_model_set(cases) if args.dry_run else build_model_set(settings.models)
    retriever: Retriever = build_retriever(
        replace(settings, retrieval="local") if args.dry_run else settings
    )

    threshold = settings.threshold
    eval_cases = cases
    if args.calibrate and settings.pipeline.target_open_fraction is not None:
        raise ValueError("use either --calibrate or --open-fraction, not both")
    if args.calibrate:
        calibration_cases, eval_cases = split_calibration(cases)
        probe = run_evaluation(
            calibration_cases,
            [PolicySpec("always_open", "always_open", escalate=False)],
            settings=settings,
            models=models,
            retriever=retriever,
            run_tests=settings.pipeline.run_tests,
            dry_run=args.dry_run,
            progress=_progress,
        )
        threshold = calibrate_open_threshold(
            calibration_cases, probe.policies[0], min_pass_rate=args.min_pass_rate
        )
        print(
            f"calibrated threshold {threshold:.3f} on {len(calibration_cases)} cases, "
            f"evaluating on {len(eval_cases)} others",
            file=sys.stderr,
        )

    policies = default_policies(threshold, settings.pipeline.target_open_fraction)
    if args.policies:
        wanted = {p.strip() for p in args.policies.split(",")}
        unknown = wanted - set(POLICIES)
        if unknown:
            raise ValueError(f"unknown policies: {', '.join(sorted(unknown))}")
        policies = [p for p in policies if p.policy in wanted]

    report = run_evaluation(
        eval_cases,
        policies,
        settings=settings,
        models=models,
        retriever=retriever,
        run_tests=settings.pipeline.run_tests,
        dry_run=args.dry_run,
        progress=_progress,
    )
    markdown = report.to_markdown()
    print(markdown)
    if args.json_out:
        args.json_out.write_text(report.to_json(), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    return 0


def _progress(message: str) -> None:
    print(message, file=sys.stderr)


def _cmd_index_provision(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    project = args.project or settings.models.project
    if not project:
        raise ValueError("pass --project or set GOOGLE_CLOUD_PROJECT")
    names = provision_vector_search(
        project=project, location=args.location, dim=args.dim, display_name=args.display_name
    )
    print(json.dumps(names, indent=2, default=str))
    print()
    print("# add these to your environment")
    print(f"export MODERNIZER_VS_INDEX={names.get('index_name', '')}")
    print(f"export MODERNIZER_VS_ENDPOINT={names.get('endpoint_name', '')}")
    print(f"export MODERNIZER_VS_DEPLOYED_ID={names.get('deployed_index_id', '')}")
    print(f"export MODERNIZER_VS_LOCATION={args.location}")
    print("export MODERNIZER_RETRIEVAL=vertex")
    return 0


def _cmd_index_build(args: argparse.Namespace) -> int:
    settings = replace(Settings.from_env(), retrieval="vertex")
    retriever = build_retriever(settings)
    count = retriever.build_index()
    print(f"upserted {count} knowledge entries into Vertex AI Vector Search")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    if uvicorn is None:
        raise ImportError("install the service extra: pip install '.[serve]'")
    uvicorn.run("modernizer.service.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
