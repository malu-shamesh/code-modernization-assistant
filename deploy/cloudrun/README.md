# Deploying to Cloud Run

This directory deploys the modernizer HTTP service (`src/modernizer/service`) as a serverless
Cloud Run service. The service converts Fortran 77 modules to Python (numpy) per request: it
parses the source, asks a model for the Python, has generated pytest tests run against it in a
sandbox, and routes each module either to a small open model (Gemma 3 12B on a Vertex AI endpoint
you deploy yourself) or to a frontier model (Gemini), escalating on failure. Results go to
BigQuery, metrics and traces to Cloud Monitoring and Cloud Trace.

Read the cost warnings and the security section before deploying. Read "What has NOT been
verified" before relying on any script here.

**Prerequisite for the default configuration: a Vertex AI endpoint serving Gemma.** The open tier
defaults to `MODERNIZER_OPEN_BACKEND=endpoint`, which calls an endpoint you deployed from Model
Garden. Do that first with `deploy/vertex_gemma/README.md`, then pass the endpoint id and its
region to the deployment. Without an endpoint id, `deploy.sh` stops before it creates anything,
and a service started without one fails at startup (not on the first request) with a message that
names `MODERNIZER_OPEN_ENDPOINT_ID`. To skip the self-hosted endpoint, choose another backend
(see "Open model configuration").

## What gets created

| Resource | Purpose |
| --- | --- |
| Enabled APIs | Cloud Run, Cloud Build, Artifact Registry, Vertex AI, BigQuery, Monitoring, Trace |
| Service account `modernizer-run` | Identity of the service, with the roles below and nothing else |
| Artifact Registry repo `code-modernizer` | Holds the container image |
| Cloud Run service `code-modernizer` | The API; authentication required |

Roles of the service account:

| Role | Scope | Why |
| --- | --- | --- |
| `roles/aiplatform.user` | project | call Gemini, your Gemma endpoint (online prediction), Model Garden APIs, embeddings, Vector Search |
| `roles/bigquery.dataEditor` | the dataset only (project-wide as a fallback, see below) | stream result rows |
| `roles/bigquery.jobUser` | project | required by the BigQuery client library |
| `roles/monitoring.metricWriter` | project | export OpenTelemetry metrics |
| `roles/cloudtrace.agent` | project | export OpenTelemetry spans |

With `PRICING_SECRET` set (optional, see "Pricing"), the service account also gets
`roles/secretmanager.secretAccessor` on that one secret, and the Secret Manager API is enabled.

The dataset-level grant uses `bq show` and `bq update` plus `jq`. If either tool is missing, the
script warns and grants `roles/bigquery.dataEditor` on the whole project instead, which is
broader than needed.

No role beyond `roles/aiplatform.user` is added for the Gemma endpoint: that role covers
prediction on endpoints in the same project (not verified against a live project). An endpoint
in another project needs the role granted there.

## Step by step

All commands run from the repository root. You need the Google Cloud SDK (`gcloud`, `bq`), `jq`,
a project with billing enabled, and permission to enable APIs, create service accounts, bind IAM
roles and deploy Cloud Run services.

1. **Authenticate and pick the project.**

   ```bash
   gcloud auth login
   export PROJECT_ID=my-project
   ```

2. **Create the BigQuery dataset, tables and views.** This must happen before the deploy,
   because the deploy grants the service account access to the dataset.

   ```bash
   PROJECT="$PROJECT_ID" DATASET=modernizer bash deploy/bigquery/setup.sh
   ```

3. **Optional: provision Vector Search.** Skip this to use the local in-memory index, which is
   built from the bundled knowledge base at startup and costs nothing extra. Vector Search is
   billable while deployed (see the cost section). See `deploy/vector_search/README.md`.

   ```bash
   gcloud auth application-default login
   PROJECT_ID="$PROJECT_ID" BUILD_INDEX=1 bash deploy/vector_search/provision.sh
   # then export the three MODERNIZER_VS_* values it prints
   ```

4. **Deploy Gemma and write the configuration.** Follow `deploy/vertex_gemma/README.md` to get an
   endpoint id (a billable step, read its cost warning), then:

   ```bash
   cp deploy/cloudrun/env.example deploy/cloudrun/my.env
   # edit my.env: PROJECT_ID, MODERNIZER_OPEN_ENDPOINT_ID and MODERNIZER_OPEN_LOCATION (the
   # region of the endpoint), INVOKER_MEMBER so you can call the service, and usually
   # MODERNIZER_OPEN_FRACTION (for example 0.7) and a pricing file (see "Pricing")
   ```

5. **Check what would happen, then deploy.**

   ```bash
   bash deploy/cloudrun/deploy.sh --env-file deploy/cloudrun/my.env --dry-run
   bash deploy/cloudrun/deploy.sh --env-file deploy/cloudrun/my.env
   ```

   The deploy runs: enable APIs, create the service account and bind roles, create the
   repository, `gcloud builds submit --tag`, `gcloud run deploy`. Re-running converges on the
   same state: existing resources are skipped and `--set-env-vars` replaces the whole
   environment of the service with the values the script computes.

6. **Call it.** The service requires an identity token of a principal with `roles/run.invoker`:

   ```bash
   URL="$(gcloud run services describe code-modernizer --region us-central1 \
     --project "$PROJECT_ID" --format 'value(status.url)')"
   TOKEN="$(gcloud auth print-identity-token)"

   curl -H "Authorization: Bearer $TOKEN" "$URL/readyz"

   curl -X POST "$URL/v1/modernize" \
     -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"files": [{"path": "isumsq.f", "source": "      INTEGER FUNCTION ISUMSQ(N)\n      INTEGER N, I\n      ISUMSQ = 0\n      DO 10 I = 1, N\n         ISUMSQ = ISUMSQ + I*I\n   10 CONTINUE\n      RETURN\n      END\n"}], "policy": "complexity", "open_fraction": 0.7}'
   ```

   Sources are fixed-form Fortran 77. The columns matter (label in 1-5, continuation mark in 6,
   statements from 7), so build the JSON from files with a tool rather than by hand, for example
   `jq -n --rawfile s isumsq.f '{files: [{path: "isumsq.f", source: $s}]}'`.

Deployment settings fixed in the script: 2 vCPU, 2 GiB memory, 900 s request timeout, container
concurrency 4, at most 5 instances, startup CPU boost, `--no-allow-unauthenticated`. The service
itself admits at most `MODERNIZER_MAX_CONCURRENT` (default 2) in-flight `/v1/modernize`
requests per instance and answers the rest with `429` and `Retry-After`; the remaining
container concurrency leaves room for probes and `/v1/analyze`.

## API summary

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | liveness; always 200, touches nothing |
| `GET /readyz` | 200 once startup finished, with a non-secret config summary (model names, the open backend `endpoint`, `maas` or `openai`, retrieval, policy, threshold, open fraction; never ids, URLs or credentials) |
| `POST /v1/analyze` | detected Fortran constructs, metrics, difficulty and the routing decision; no model calls |
| `POST /v1/modernize` | convert, test and repair; response has a run summary and one result per file |

Request fields of `/v1/modernize` besides `files` (each `{path, source}`, the path naming a
`.f`, `.for`, `.f77` or `.ftn` file in any letter case): `policy`, `threshold`, `open_fraction`
(a number from 0 to 1: the target share of the modules in THIS request that go to the open model;
under the `complexity` policy the threshold is calibrated on the batch and takes precedence over
`threshold`), `run_tests` (default true) and `include_tests`. The summary reports
`planned_open_share` (null when no target was in effect) next to the realised `open_fraction`,
`routed_open`, `escalated` and the token and cost totals. A small batch can only approximate the
target (a request with 3 modules cannot route exactly 70% to the open model).

Errors: `401` bad or missing key (only when `MODERNIZER_API_KEY` is set), `413` over a size
limit, `422` invalid body or path (absolute, `..`, backslash, not a Fortran extension, duplicate),
`429` too many concurrent conversions, `500` with a generic body (details are in the server log
only). A module that fails is not an HTTP error: its result has `success: false` and a short
generic `error` (for example the exception type, never its text).

Run it locally without any cloud access:

```bash
MODERNIZER_PROVIDER=mock MODERNIZER_RETRIEVAL=local python -m modernizer.service
```

The `mock` provider answers every request with a stub, so conversions report `success: false`.

## Open model configuration

All of these are environment variables of the service; `deploy.sh` reads them from the shell or
the env file, validates them, and forwards them.

| Variable | Meaning |
| --- | --- |
| `MODERNIZER_OPEN_BACKEND` | `endpoint` (default), `maas` or `openai` |
| `MODERNIZER_OPEN_ENDPOINT_ID` | required for `endpoint`: id of your Vertex AI endpoint |
| `MODERNIZER_OPEN_LOCATION` | region of that endpoint; `deploy.sh` requires it for `endpoint` |
| `MODERNIZER_OPEN_BASE_URL` | required for `openai` (vLLM, Ollama, GKE ...); takes precedence over the backend if set |
| `MODERNIZER_OPEN_FOLD_SYSTEM` | `true` (default) puts the system prompt into the user message, because Gemma chat templates have no system role |
| `MODERNIZER_OPEN_MODEL` | model name sent with the request and used for the price lookup; default `gemma-3-12b-it` |
| `MODERNIZER_OPEN_FRACTION` | default target share for the open model, for example `0.7`; requests may override it |
| `MODERNIZER_PRICING_FILE` | price table (YAML) used for the cost figures, see "Pricing" |

Fail-fast behaviour: `deploy.sh` rejects an `endpoint` backend without an endpoint id (and
without a location), an `openai` backend without a base URL, and malformed values, before it
touches any resource, also in `--dry-run`. The service itself raises the same error at startup,
so a revision that is missing its endpoint id never becomes ready.

## Pricing

Cost figures in responses, BigQuery rows and reports come from a price table in USD per million
tokens. The packaged table (`src/modernizer/llm/pricing.yaml`) contains illustrative
placeholders, not prices. Replace them before drawing any conclusion.

For Gemini, take the current Vertex AI price list. For a self-hosted Gemma endpoint the table
entry is an effective, amortised rate that you compute yourself from the node-hour price and a
throughput you measured on your own traffic (`modernizer.llm.pricing.amortised_price_per_mtok`;
worked steps in `deploy/vertex_gemma/README.md`). Keep the YAML outside the repository and point
`MODERNIZER_PRICING_FILE` at it. The path is read INSIDE the container, so the file must be
part of the image or mounted. `deploy.sh` supports mounting from Secret Manager:

```bash
gcloud secrets create modernizer-pricing --data-file=my-pricing.yaml --project "$PROJECT_ID"
PRICING_SECRET=modernizer-pricing   # in my.env; the file appears at MODERNIZER_PRICING_FILE,
                                    # default /etc/modernizer/pricing.yaml
```

The service reads the table at startup: an unreadable file stops the start, and a configured
model without an entry is logged as a warning (it would be costed at 0.0 USD).

## Cost warnings

* **Vector Search is billed while deployed.** A deployed index bills for its replica machines
  every hour, whether or not it serves queries, and an endpoint exists independently of the
  index. Undeploy and delete when you stop experimenting (commands in
  `deploy/vector_search/README.md`). The local index has no such cost, so use it unless you
  specifically want to exercise the managed path.
* **A self-hosted Gemma endpoint is billed per node hour, not per token.** From the moment the
  model is deployed to the endpoint until you undeploy it, the accelerator node bills every
  hour, including when no request arrives and while Cloud Run is scaled to zero. This is
  usually the largest and the least visible cost of this setup. Undeploy and delete the endpoint
  when you stop (commands in `deploy/vertex_gemma/README.md`).
* **Per-token comparisons of Gemma and Gemini depend on a number you compute.** Gemini is
  billed per token. The Gemma rate in the pricing table is only meaningful as an amortised
  figure (node-hour price divided by the tokens the node actually served, including idle time),
  so the reports' cost comparison is only as good as the throughput and utilisation behind that
  figure. With few requests the node is mostly idle and the effective per-token price of the open
  tier can be far above the frontier model's.
* **Gemini calls are billed per token.** A single module can trigger several calls: conversion,
  test generation, repair rounds on the open tier, then repeated attempts on the frontier tier
  after escalation. Check the current Vertex AI price list. The response of every request
  reports the estimated cost (`cost_usd`) computed from the pricing table, which is an estimate
  and can differ from your invoice.
* **Cloud Run** bills for CPU, memory and requests while instances handle requests. Cloud Build,
  Artifact Registry storage, BigQuery storage and streaming inserts, Cloud Monitoring and Cloud
  Trace ingestion also bill according to their own price lists, mostly at small volumes.
* The limits (`MODERNIZER_MAX_FILES`, `..._CHARS`, `MODERNIZER_MAX_CONCURRENT`) and the maximum
  of 5 instances bound how much one caller can spend. Lower them if the service is shared.

## Security

**Access control.** The service is deployed with `--no-allow-unauthenticated`: Cloud Run IAM is
the protection, and only principals with `roles/run.invoker` can reach it. This is the
recommended setup. Never run this service with public access.

`MODERNIZER_API_KEY` (a bearer token checked in constant time on `/v1/*`) exists for local and
development use. It reads the same `Authorization` header that Cloud Run IAM consumes, so the
two cannot be combined on one Cloud Run service, and `deploy.sh` deliberately does not forward
the variable: no secrets belong in the service environment.

**Generated tests are untrusted code executed inside the container.** The pipeline asks a model
to write pytest suites and runs them against the converted module, and the converted module is
derived from the caller's input. Anyone who can call `/v1/modernize` can therefore influence code
that runs in the container. What limits the damage:

* The test process runs in a throwaway directory with a scrubbed environment (no service
  credentials in its environment variables), a wall-clock timeout that kills the process tree,
  and on Linux CPU and address-space limits.
* An AST deny-list rejects obviously dangerous generated tests. It is best effort only and
  cannot be a security boundary: Python offers too many ways around any such list.
* The real boundary is Cloud Run's sandbox (a gVisor-based or microVM container with no access
  to the host), the unprivileged user in the Dockerfile, a service account with only the roles
  above, and no secrets in the environment or image.
* `MODERNIZER_MAX_CONCURRENT` plus a small instance count bound resource abuse; they do not
  isolate requests from each other: concurrent requests on one instance share the container.

**Residual risk.** Code running in the container can reach the Cloud Run metadata server and
obtain an access token for the service account. With the roles above, a token holder could call
Vertex AI models at your expense, write rows into the dataset, and write metrics and traces.
It cannot touch other datasets, other projects' resources or IAM. If that is not acceptable,
add network controls (for example direct VPC egress with firewall rules that block outbound
traffic other than Google APIs), keep the invoker list short, and consider a budget alert.

**Errors.** Unexpected exceptions return a generic `500` with an `error_id`; the traceback goes
to Cloud Logging only. Request bodies (which are source code) are never logged by the service.

## What has NOT been verified

The scripts and the Dockerfile were written on a machine without `gcloud`, `bq`, `jq` or Docker.

* `deploy.sh` and `../vector_search/provision.sh` were only run in `--dry-run` mode (the
  printed commands were inspected) and with stub interpreters; no real `gcloud` or `bq` command
  has been executed. Flag names, the `^|^` delimiter syntax of `--set-env-vars`, the
  `bq show`/`bq update --source` dataset grant with the `WRITER` role, and `--cpu-boost` are
  written from memory of the CLI documentation. Run with `--dry-run` first and compare with
  `gcloud run deploy --help`.
* The Dockerfile has not been built. Dependency resolution on `python:3.12-slim` for the `gcp`
  and `serve` extras, and the runtime behaviour of the non-root user, are untested. In particular
  it is unknown whether prebuilt wheels of `tree-sitter` and `tree-sitter-fortran` exist for that
  Python version and platform (the slim image has no compiler; if there are none, the build
  fails at `pip install`). The check after the install (imports of numpy, pytest and the parser,
  the packaged `f77rt` runtime and pricing table) is meant to fail the build instead of the
  first request; it was run only in a local virtual environment.
* The new open-tier checks and options of `deploy.sh` (validation, `PRICING_SECRET`) were
  exercised in `--dry-run` only. The `--set-secrets PATH=SECRET:latest` file mount, the
  `roles/secretmanager.secretAccessor` binding on the secret and the claim that
  `roles/aiplatform.user` covers prediction on a Vertex AI endpoint are written from memory of
  the documentation.
* Whether the Vertex AI OpenAI-compatible route accepts the requests the open-tier client sends
  to a self-deployed Gemma endpoint (URL shape, API version, the `model` field) is documented as
  unverified in `src/modernizer/llm/openai_compat.py` and in `deploy/vertex_gemma/README.md`.
* Nothing was run against real Vertex AI, BigQuery, Cloud Monitoring or Cloud Trace. The HTTP
  layer is covered by offline tests with fake models (`tests/service`) and a local start with the
  `mock` provider.
* The permissions of the identity running `gcloud builds submit` (newer projects run builds as
  the default Compute Engine service account, which may need extra roles) and the availability
  of the default model ids in your region are not checked by the script.
* No performance, latency or cost figures have been measured for the deployed service.
