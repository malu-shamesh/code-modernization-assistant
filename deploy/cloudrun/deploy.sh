#!/usr/bin/env bash
# Builds the container and deploys the Fortran 77 -> Python modernizer service to Cloud Run.
#
# Usage:
#   PROJECT_ID=my-project bash deploy.sh [--env-file deploy/cloudrun/env] [--dry-run]
#
# Reads PROJECT_ID, REGION, SERVICE ... from the environment (see env.example for every
# variable). Values in --env-file are sourced and override the environment, so only point it at
# a file you trust. With --dry-run (or DRY_RUN=1) every state-changing command is printed
# instead of executed and nothing is queried, which is how this script can be checked without
# credentials.
#
# What it does, in order (each step can be re-run):
#   1. enables the required APIs
#   2. creates a dedicated service account with least-privilege roles
#   3. creates the Artifact Registry repository
#   4. builds the image with Cloud Build (skipped when IMAGE is set)
#   5. deploys to Cloud Run with authentication required (--no-allow-unauthenticated)
#
# The open tier (Gemma 3 12B by default) is a Vertex AI endpoint you deploy yourself, see
# deploy/vertex_gemma/README.md. The default backend is "endpoint", so MODERNIZER_OPEN_ENDPOINT_ID
# and MODERNIZER_OPEN_LOCATION are required and the script stops before changing anything when
# they are missing. Choose MODERNIZER_OPEN_BACKEND=maas, or openai with MODERNIZER_OPEN_BASE_URL,
# to use another open backend.
#
# Run deploy/bigquery/setup.sh first: the dataset must exist before the service account is
# granted access to it. See README.md in this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
ENV_FILE="${ENV_FILE:-}"

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || { echo "--env-file needs a path" >&2; exit 2; }
      ENV_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { echo "env file not found: $ENV_FILE" >&2; exit 2; }
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-code-modernizer}"
SA_NAME="${SA_NAME:-modernizer-run}"
AR_REPO="${AR_REPO:-code-modernizer}"
BQ_DATASET="${BQ_DATASET:-modernizer}"
IMAGE="${IMAGE:-}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
JSONL_SINK_DIR="${JSONL_SINK_DIR:-}"
INVOKER_MEMBER="${INVOKER_MEMBER:-}"
PRICING_SECRET="${PRICING_SECRET:-}"

die() {
  echo "error: $*" >&2
  exit 2
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

[[ -n "$PROJECT_ID" ]] || die "PROJECT_ID is required (see env.example)"
[[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "invalid PROJECT_ID: $PROJECT_ID"
[[ "$REGION" =~ ^[a-z]+-[a-z]+[0-9]+$ ]] || die "invalid REGION: $REGION"
[[ "$SERVICE" =~ ^[a-z]([-a-z0-9]{0,47}[a-z0-9])?$ ]] || die "invalid SERVICE: $SERVICE"
[[ "$SA_NAME" =~ ^[a-z][-a-z0-9]{4,28}[a-z0-9]$ ]] || die "invalid SA_NAME (6-30 chars): $SA_NAME"
[[ "$AR_REPO" =~ ^[a-z]([-a-z0-9]*[a-z0-9])?$ ]] || die "invalid AR_REPO: $AR_REPO"
[[ "$BQ_DATASET" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid BQ_DATASET: $BQ_DATASET"
[[ "$IMAGE_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid IMAGE_TAG: $IMAGE_TAG"

# Open-tier configuration. The service refuses to start with an inconsistent combination, which
# on Cloud Run shows up as a failed revision; check it here, before anything is created. This
# runs in a dry run too, so a configuration can be checked without credentials.
OPEN_BACKEND="$(lower "${MODERNIZER_OPEN_BACKEND:-endpoint}")"
OPEN_ENDPOINT_ID="${MODERNIZER_OPEN_ENDPOINT_ID:-}"
OPEN_BASE_URL="${MODERNIZER_OPEN_BASE_URL:-}"
OPEN_LOCATION="${MODERNIZER_OPEN_LOCATION:-}"
OPEN_FOLD="$(lower "${MODERNIZER_OPEN_FOLD_SYSTEM:-}")"
OPEN_FRACTION="${MODERNIZER_OPEN_FRACTION:-}"
PRICING_FILE_PATH="${MODERNIZER_PRICING_FILE:-}"

# Kept in variables because quoting inside [[ =~ ]] is easy to get wrong.
URL_RE='^https?://[^[:space:]|]+$'
FRACTION_RE='^(0|1|0?\.[0-9]+|[01]\.[0-9]+)$'
PATH_RE='^/[A-Za-z0-9_./-]+$'

case "$OPEN_BACKEND" in
  endpoint | maas | openai) ;;
  *) die "invalid MODERNIZER_OPEN_BACKEND '${OPEN_BACKEND}': use endpoint, maas or openai" ;;
esac
if [[ -n "$OPEN_BASE_URL" ]]; then
  # The service gives a base URL precedence over the backend setting.
  [[ "$OPEN_BASE_URL" =~ $URL_RE ]] \
    || die "invalid MODERNIZER_OPEN_BASE_URL: expected http(s)://host/path without spaces or '|'"
elif [[ "$OPEN_BACKEND" == endpoint ]]; then
  if [[ -z "$OPEN_ENDPOINT_ID" ]]; then
    die "MODERNIZER_OPEN_BACKEND is 'endpoint' (the default) but MODERNIZER_OPEN_ENDPOINT_ID is not set.
Deploy Gemma to a Vertex AI endpoint first (deploy/vertex_gemma/README.md) and set its endpoint id,
or choose another backend: MODERNIZER_OPEN_BACKEND=maas, or openai with MODERNIZER_OPEN_BASE_URL."
  fi
  [[ "$OPEN_ENDPOINT_ID" =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "invalid MODERNIZER_OPEN_ENDPOINT_ID: give the endpoint id only, not a resource path"
  [[ -n "$OPEN_LOCATION" ]] \
    || die "set MODERNIZER_OPEN_LOCATION to the region the Vertex AI endpoint is deployed in"
elif [[ "$OPEN_BACKEND" == openai ]]; then
  die "MODERNIZER_OPEN_BACKEND is 'openai' but MODERNIZER_OPEN_BASE_URL is not set"
fi
if [[ -n "$OPEN_LOCATION" && ! "$OPEN_LOCATION" =~ ^[a-z]+(-[a-z]+[0-9]+)?$ ]]; then
  die "invalid MODERNIZER_OPEN_LOCATION: $OPEN_LOCATION"
fi
if [[ -n "$OPEN_FOLD" && ! "$OPEN_FOLD" =~ ^(1|0|true|false|yes|no|on|off)$ ]]; then
  die "invalid MODERNIZER_OPEN_FOLD_SYSTEM: use true or false"
fi
if [[ -n "$OPEN_FRACTION" ]]; then
  [[ "$OPEN_FRACTION" =~ $FRACTION_RE ]] && awk -v f="$OPEN_FRACTION" 'BEGIN { exit !(f <= 1) }' \
    || die "invalid MODERNIZER_OPEN_FRACTION '${OPEN_FRACTION}': use a number from 0 to 1, for example 0.7"
fi
if [[ -n "$PRICING_SECRET" ]]; then
  [[ "$PRICING_SECRET" =~ ^[A-Za-z0-9_-]{1,255}$ ]] || die "invalid PRICING_SECRET: $PRICING_SECRET"
  PRICING_FILE_PATH="${PRICING_FILE_PATH:-/etc/modernizer/pricing.yaml}"
fi
if [[ -n "$PRICING_FILE_PATH" ]]; then
  [[ "$PRICING_FILE_PATH" =~ $PATH_RE && "$PRICING_FILE_PATH" != *..* ]] \
    || die "invalid MODERNIZER_PRICING_FILE: expected an absolute path inside the container"
  if [[ -z "$PRICING_SECRET" ]]; then
    echo "warning: MODERNIZER_PRICING_FILE=${PRICING_FILE_PATH} must exist inside the image." \
      "Set PRICING_SECRET to mount it from Secret Manager instead. The service refuses to" \
      "start when the file cannot be read." >&2
  fi
fi

# Assemble the service environment before touching anything, so that a bad configuration
# fails here and not halfway through the deployment.
SINK_SPEC="bigquery:${PROJECT_ID}.${BQ_DATASET}"
if [[ -n "$JSONL_SINK_DIR" ]]; then
  # Cloud Run's filesystem is ephemeral and per instance: only useful for debugging.
  SINK_SPEC="${SINK_SPEC},jsonl:${JSONL_SINK_DIR}"
fi

ENV_VARS=(
  "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
  "MODERNIZER_PROVIDER=vertex"
  "MODERNIZER_TELEMETRY=gcp"
  "MODERNIZER_SINK=${SINK_SPEC}"
  "MODERNIZER_OPEN_BACKEND=${OPEN_BACKEND}"
)
[[ -z "$OPEN_ENDPOINT_ID" ]] || ENV_VARS+=("MODERNIZER_OPEN_ENDPOINT_ID=${OPEN_ENDPOINT_ID}")
[[ -z "$OPEN_BASE_URL" ]] || ENV_VARS+=("MODERNIZER_OPEN_BASE_URL=${OPEN_BASE_URL}")
[[ -z "$OPEN_FOLD" ]] || ENV_VARS+=("MODERNIZER_OPEN_FOLD_SYSTEM=${OPEN_FOLD}")
[[ -z "$OPEN_FRACTION" ]] || ENV_VARS+=("MODERNIZER_OPEN_FRACTION=${OPEN_FRACTION}")
[[ -z "$PRICING_FILE_PATH" ]] || ENV_VARS+=("MODERNIZER_PRICING_FILE=${PRICING_FILE_PATH}")

vs_set=0
for name in MODERNIZER_VS_INDEX MODERNIZER_VS_ENDPOINT MODERNIZER_VS_DEPLOYED_ID; do
  [[ -n "${!name:-}" ]] && vs_set=$((vs_set + 1))
done
if ((vs_set == 3)); then
  ENV_VARS+=(
    "MODERNIZER_RETRIEVAL=vertex"
    "MODERNIZER_VS_INDEX=${MODERNIZER_VS_INDEX}"
    "MODERNIZER_VS_ENDPOINT=${MODERNIZER_VS_ENDPOINT}"
    "MODERNIZER_VS_DEPLOYED_ID=${MODERNIZER_VS_DEPLOYED_ID}"
    "MODERNIZER_VS_LOCATION=${MODERNIZER_VS_LOCATION:-$REGION}"
  )
elif ((vs_set == 0)); then
  ENV_VARS+=("MODERNIZER_RETRIEVAL=local")
else
  die "set all of MODERNIZER_VS_INDEX, MODERNIZER_VS_ENDPOINT and MODERNIZER_VS_DEPLOYED_ID, or none"
fi

# Optional tuning, forwarded only when set (the open backend, endpoint, fraction and pricing
# variables are handled above). MODERNIZER_API_KEY is deliberately not forwarded:
# on Cloud Run access is controlled by IAM and no secret should sit in the environment.
for name in \
  MODERNIZER_FRONTIER_MODEL MODERNIZER_OPEN_MODEL \
  MODERNIZER_FRONTIER_LOCATION MODERNIZER_OPEN_LOCATION \
  MODERNIZER_POLICY MODERNIZER_THRESHOLD \
  MODERNIZER_EMBED_MODEL MODERNIZER_EMBED_DIM \
  MODERNIZER_MAX_REPAIRS MODERNIZER_WORKERS MODERNIZER_TEST_TIMEOUT_S \
  MODERNIZER_MAX_FILES MODERNIZER_MAX_FILE_CHARS MODERNIZER_MAX_TOTAL_CHARS \
  MODERNIZER_MAX_CONCURRENT MODERNIZER_RETRY_AFTER_S MODERNIZER_LOG_LEVEL; do
  if [[ -n "${!name:-}" ]]; then
    ENV_VARS+=("${name}=${!name}")
  fi
done

# gcloud splits --set-env-vars on commas, and the sink spec contains one, so use the
# '^DELIM^' prefix described in `gcloud topic escaping`, with '|' as the delimiter.
ENV_ARG="^|^"
for pair in "${ENV_VARS[@]}"; do
  if [[ "$pair" == *"|"* || "$pair" == *$'\n'* ]]; then
    die "environment values must not contain '|' or newlines: ${pair%%=*}"
  fi
  ENV_ARG+="${pair}|"
done
ENV_ARG="${ENV_ARG%|}"

if [[ "$DRY_RUN" != 1 ]]; then
  command -v gcloud >/dev/null || die "gcloud is not installed or not on PATH"
fi

# Prints the command (to stderr) and runs it unless this is a dry run.
run() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

# True when the command succeeds. Never true in a dry run, so that creation steps are shown.
exists() {
  [[ "$DRY_RUN" != 1 ]] && "$@" >/dev/null 2>&1
}

# IAM is eventually consistent: a binding made right after creating the service account can
# fail for a few seconds.
retry() {
  local attempt=1
  until "$@"; do
    if ((attempt >= 5)); then
      return 1
    fi
    echo "failed, retrying in 5s (attempt ${attempt} of 5)" >&2
    sleep 5
    attempt=$((attempt + 1))
  done
}

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "== 1/5 enabling APIs" >&2
APIS=(
  run.googleapis.com
  cloudbuild.googleapis.com
  artifactregistry.googleapis.com
  aiplatform.googleapis.com
  bigquery.googleapis.com
  monitoring.googleapis.com
  cloudtrace.googleapis.com
)
[[ -z "$PRICING_SECRET" ]] || APIS+=(secretmanager.googleapis.com)
run gcloud services enable "${APIS[@]}" --project "$PROJECT_ID"

echo "== 2/5 service account ${SA_EMAIL}" >&2
if ! exists gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID"; then
  run gcloud iam service-accounts create "$SA_NAME" \
    --display-name "Code modernizer (Cloud Run)" \
    --project "$PROJECT_ID"
fi

# Project-level roles. add-iam-policy-binding is idempotent. --condition=None avoids the
# interactive prompt that appears when the project policy already contains conditional bindings.
for role in \
  roles/aiplatform.user \
  roles/bigquery.jobUser \
  roles/monitoring.metricWriter \
  roles/cloudtrace.agent; do
  retry run gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:${SA_EMAIL}" \
    --role "$role" \
    --condition=None \
    --quiet >/dev/null
done

# Optional: mount the pricing file from a Secret Manager secret that you created beforehand, for
# example with: gcloud secrets create NAME --data-file=my-pricing.yaml. Pricing is not
# confidential; a secret is just the least effort way to hand a file to Cloud Run without
# rebuilding the image. Calling the Vertex AI endpoint needs no extra role: roles/aiplatform.user
# above covers online prediction.
SECRET_ARGS=()
if [[ -n "$PRICING_SECRET" ]]; then
  if [[ "$DRY_RUN" != 1 ]] && ! exists gcloud secrets describe "$PRICING_SECRET" \
    --project "$PROJECT_ID"; then
    die "secret ${PRICING_SECRET} not found in ${PROJECT_ID}; create it from your pricing YAML first"
  fi
  retry run gcloud secrets add-iam-policy-binding "$PRICING_SECRET" \
    --member "serviceAccount:${SA_EMAIL}" \
    --role roles/secretmanager.secretAccessor \
    --project "$PROJECT_ID" \
    --quiet >/dev/null
  SECRET_ARGS=(--set-secrets "${PRICING_FILE_PATH}=${PRICING_SECRET}:latest")
fi

# roles/bigquery.dataEditor is granted on the dataset only (legacy dataset role WRITER), so the
# service cannot touch other datasets. This needs the bq CLI and jq; without them the script
# falls back to a project-level grant, which is broader than necessary.
grant_dataset_editor() {
  local dataset_ref="${PROJECT_ID}:${BQ_DATASET}" current updated scratch
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "+ (dataset-level WRITER grant on ${dataset_ref} for ${SA_EMAIL} via bq show / bq update)" >&2
    return 0
  fi
  if ! command -v bq >/dev/null || ! command -v jq >/dev/null; then
    echo "warning: bq or jq not found, granting roles/bigquery.dataEditor on the whole project" >&2
    retry run gcloud projects add-iam-policy-binding "$PROJECT_ID" \
      --member "serviceAccount:${SA_EMAIL}" \
      --role roles/bigquery.dataEditor \
      --condition=None \
      --quiet >/dev/null
    return 0
  fi
  scratch="$(mktemp -d)"
  # Expanded now: the local variable is gone by the time the EXIT trap runs.
  # shellcheck disable=SC2064
  trap "rm -rf '${scratch}'" EXIT
  current="${scratch}/current.json"
  updated="${scratch}/updated.json"
  if ! bq --project_id="$PROJECT_ID" show --format=prettyjson "$dataset_ref" >"$current" 2>/dev/null; then
    echo "error: dataset ${dataset_ref} not found; run deploy/bigquery/setup.sh first" >&2
    return 1
  fi
  if jq -e --arg e "$SA_EMAIL" \
    '(.access // []) | any(((.userByEmail // "") | ascii_downcase) == ($e | ascii_downcase) and .role == "WRITER")' \
    "$current" >/dev/null; then
    echo "dataset ${dataset_ref} already grants ${SA_EMAIL} write access" >&2
    return 0
  fi
  jq --arg e "$SA_EMAIL" \
    '.access = ((.access // []) + [{"role": "WRITER", "userByEmail": $e}])' \
    "$current" >"$updated"
  retry bq --project_id="$PROJECT_ID" update --source "$updated" "$dataset_ref"
}
grant_dataset_editor

echo "== 3/5 Artifact Registry repository ${AR_REPO}" >&2
if ! exists gcloud artifacts repositories describe "$AR_REPO" \
  --location "$REGION" --project "$PROJECT_ID"; then
  run gcloud artifacts repositories create "$AR_REPO" \
    --repository-format docker \
    --location "$REGION" \
    --description "Code modernizer images" \
    --project "$PROJECT_ID"
fi

echo "== 4/5 image" >&2
if [[ -z "$IMAGE" ]]; then
  IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE}:${IMAGE_TAG}"
  # Uses the Dockerfile at the repo root. Cloud Build reads .gcloudignore, or .gitignore when
  # there is none; .dockerignore then trims the build context.
  run gcloud builds submit "$REPO_ROOT" --tag "$IMAGE" --project "$PROJECT_ID"
else
  echo "using prebuilt image ${IMAGE}" >&2
fi

echo "== 5/5 Cloud Run service ${SERVICE}" >&2
# --set-env-vars replaces the whole environment of the service, so a deploy always converges on
# exactly the values above.
run gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --service-account "$SA_EMAIL" \
  --no-allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --timeout 900 \
  --concurrency 4 \
  --max-instances 5 \
  --cpu-boost \
  --port 8080 \
  --set-env-vars "$ENV_ARG" \
  ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"} \
  --quiet

if [[ -n "$INVOKER_MEMBER" ]]; then
  run gcloud run services add-iam-policy-binding "$SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --member "$INVOKER_MEMBER" \
    --role roles/run.invoker \
    --quiet >/dev/null
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "dry run finished, nothing was changed" >&2
  exit 0
fi

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" \
  --format 'value(status.url)')"
cat <<EOF

Deployed ${SERVICE}: ${URL}
Callers need roles/run.invoker (set INVOKER_MEMBER, or grant it yourself). Try:

  curl -H "Authorization: Bearer \$(gcloud auth print-identity-token)" ${URL}/readyz
EOF
