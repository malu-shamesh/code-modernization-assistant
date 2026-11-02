# code-modernization-assistant

An AST-guided, LLM-driven tool that converts **FORTRAN 77 to Python (numpy)**.

* **tree-sitter** parses the legacy source (a fixed-form normaliser turns columns, labels and
  continuation lines into something tree-sitter-fortran understands) and finds the constructs
  that make Fortran hard to port: GOTO webs, arithmetic IF, COMMON, EQUIVALENCE, column-major
  arrays, FORMAT I/O.
* An **LLM rewrites** each module with context retrieved from a **vector index** (Vertex AI
  Vector Search in production, a local numpy index offline).
* **Generated pytest suites** run in a sandbox and check the result. Failures drive repair
  rounds and escalation.
* **OpenTelemetry** meters tokens, cost and latency per model call.
* A **routing policy** sends a configurable share of modules to a small open model
  (**Gemma 3 12B**) and the rest to a frontier model (**Gemini**), escalating when the small
  model fails.
* Runs **serverless on Cloud Run** with Vertex AI and BigQuery.

> **Status.** The code is complete and covered by an offline test suite. Nothing has been run
> against Vertex AI, BigQuery or Cloud Run yet (no GCP access in the development environment),
> so **there are no published results**. Pass rates, the share routed to the open model and the
> cost reduction are produced by `modernize eval` on your own project; see
> [Evaluation](#evaluation). The deployment scripts have never been executed.

## Quick start (offline)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[gcp,serve,dev]"

# which legacy constructs does each file contain, and where would it be routed?
modernize analyze path/to/fortran/

# prove the plumbing end to end with stub models (NOT model results)
modernize eval --dry-run --limit 8
pytest
```

`modernize analyze` needs no credentials. `eval --dry-run` replaces both models with an oracle
that returns the hand-written reference solutions; its report carries a banner saying so.

## Converting code

```bash
export GOOGLE_CLOUD_PROJECT=my-project
export MODERNIZER_OPEN_ENDPOINT_ID=<Vertex endpoint serving Gemma 3 12B>
export MODERNIZER_OPEN_LOCATION=us-central1

modernize convert legacy/ --out python_out/ --open-fraction 0.7 \
    --sink jsonl:runs/ --telemetry console
```

Output: one `.py` per Fortran file (failed modules go to `_failed/`), the `f77rt.py` runtime
helper module the converted code imports, and with `--write-tests` the generated test suites.
`--open-fraction 0.7` calibrates the routing threshold on the batch so about 70 % of the
modules (excluding those forced to the frontier model) go to the open model. That share is a
policy setting, not a result: escalations after failures lower the final open share, and the
realised numbers are in the summary and the sink.

| Setting | Variable | Notes |
|---|---|---|
| Provider | `MODERNIZER_PROVIDER` | `vertex` or `mock` |
| Frontier model | `MODERNIZER_FRONTIER_MODEL` | default `gemini-2.5-pro`, check the Vertex catalogue |
| Open model | `MODERNIZER_OPEN_MODEL` | default `gemma-3-12b-it` |
| Open backend | `MODERNIZER_OPEN_BACKEND` | `endpoint` (self-deployed), `maas`, or `openai` + `MODERNIZER_OPEN_BASE_URL` |
| Retrieval | `MODERNIZER_RETRIEVAL` | `local` or `vertex` (`MODERNIZER_VS_*`) |
| Telemetry | `MODERNIZER_TELEMETRY` | `none`, `console`, `otlp`, `gcp` |
| Sink | `MODERNIZER_SINK` | `jsonl:<dir>`, `bigquery:<project>.<dataset>` |
| Open share | `MODERNIZER_OPEN_FRACTION` | 0-1, complexity policy only |
| Pricing | `MODERNIZER_PRICING_FILE` | replaces the packaged placeholder price table |

## How a module is converted

See [docs/architecture.md](docs/architecture.md) for the full flow. In short:

1. `modernizer.parsing` normalises fixed-form source, runs 47 construct detectors, builds a
   per-unit symbol table and infers each routine's **API contract**: Fortran passes arguments
   by reference, so scalars a routine writes become return values and arrays are modified in
   place ([calling convention](src/modernizer/conventions.py)).
2. `modernizer.retrieval` fetches migration notes for the detected constructs.
3. `modernizer.routing` scores difficulty and picks Gemma or Gemini.
4. `modernizer.conversion` builds a prompt from the convention, the runtime helper catalogue,
   the unit contracts, a checklist of the constructs found with line numbers, the notes and the
   numbered source.
5. `modernizer.parsing.validate` checks the Python (compiles, units and parameter names match
   the contract, only real `f77rt` helpers, no Fortran leftovers).
6. `modernizer.testing` generates and runs tests once per module; the same tests judge every
   tier. The pipeline repairs on the same tier, then escalates.

## Evaluation

`benchmarks/f77/` holds 60 FORTRAN 77 modules in six groups (numerics, linear algebra, sorting
and statistics, character and formatted output, unstructured control flow, stateful COMMON /
EQUIVALENCE code). Each has the legacy source, a hand-written Python port, held-out pytest
tests derived by hand from the Fortran semantics, and metadata. The tests were checked with
mutation testing; no Fortran compiler was available, so expected values rest on reading the
standard, not on running the legacy code.

```bash
# three arms: always frontier, always open, complexity routing (70 % open target)
modernize eval --open-fraction 0.7 --json report.json --markdown report.md

# or calibrate the threshold on one half of the corpus and evaluate on the other half
modernize eval --calibrate --min-pass-rate 0.9
```

The report gives, per policy, the pass rate under the **generated** tests and under the
**held-out** tests, the routed and final share per tier, tokens and cost taken from
OpenTelemetry (cross-checked against the per-call ledger), cost with and without test
generation, and for each policy against the always-frontier baseline: token and cost reduction,
paired discordant counts, an exact sign test and Wilson intervals. The verdict text is
deliberately conservative: it never says "same pass rate", only whether a difference was
detected given the sample size.

To make a claim such as "N modules pass, X % routed to Gemma, Y % cheaper at the same pass
rate" you need a corpus of N modules (the 60 bundled ones, or your own: directories with
`legacy.f` and, for held-out scoring, `reference.py` + `test_reference.py`), a real run on
Vertex AI, and a price for the self-hosted model:

* Gemma 3 12B on a Vertex AI endpoint is billed **per accelerator hour**, not per token. Compute an
  effective rate with `modernizer.llm.pricing.amortised_price_per_mtok(node_hour_usd,
  tokens_per_second, utilisation)` from your actual node price and **measured** throughput, write
  it to a YAML file and point `MODERNIZER_PRICING_FILE` at it. The packaged prices are
  placeholders.
* Tokens from different tokenizers are not equivalent; report cost, not just token counts.

## Deployment

* `deploy/vertex_gemma/` - deploy Gemma 3 12B from Model Garden to an endpoint, test it, tear it
  down (it bills while deployed).
* `deploy/bigquery/` - tables and analysis views (cost by tier, pass rate by tier, routing
  summary, cost per passing module).
* `deploy/vector_search/` - provision the Vector Search index (`modernize index provision`,
  `modernize index build`).
* `deploy/cloudrun/` and the `Dockerfile` - the HTTP service (`POST /v1/modernize`,
  `POST /v1/analyze`), deployed with IAM authentication, a dedicated least-privilege service
  account and telemetry flushed after every request.

None of the cloud scripts or the Vertex endpoint URL shapes have been run or verified against
live services; each README marks what is unverified.

## Layout

```
src/modernizer/
  parsing/      fixed-form normaliser, tree-sitter-fortran, detectors, unit API, validator
  retrieval/    knowledge base (149 notes), embedders, local and Vertex indexes, retriever
  routing/      difficulty score, policies, batch and outcome calibration
  conversion/   prompts, chunking, merge
  testing/      test generation, sanitiser, sandboxed runner
  runtime/      f77rt.py: Fortran semantics for converted code (formats, DO, MOD, SIGN ...)
  llm/          Gemini, OpenAI-compatible / Vertex endpoint clients, retry, pricing
  telemetry/    OpenTelemetry metering
  sinks/        JSONL and BigQuery result sinks
  evaluation/   benchmark harness and statistics
  service/      FastAPI app for Cloud Run
benchmarks/f77/ 60 modules with held-out tests
deploy/         BigQuery, Vector Search, Vertex Gemma, Cloud Run
```

## Known limits

* Generated tests encode a model's reading of the Fortran source; without a Fortran compiler
  there is no ground truth. Differential testing against a gfortran build of the legacy code
  would be the stronger oracle.
* Single-precision `REAL` is computed in float64; list-directed output is approximate.
* The test sandbox is a subprocess with a scrubbed environment and an AST deny-list. The
  container is the real isolation boundary.
* The bundled 60 modules are original, written for this project in the style of numerical
  libraries; they are not a sample of any real code base, so results on them do not transfer
  to yours.

## License

MIT
