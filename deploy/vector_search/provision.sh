#!/usr/bin/env bash
# Creates a Vertex AI Vector Search index and endpoint, deploys the index, and prints the
# MODERNIZER_VS_* values the service and the CLI need.
#
# THIS CREATES BILLABLE RESOURCES. A deployed index bills for its replicas for as long as it is
# deployed, whether or not it serves queries. See README.md for the teardown commands.
#
# Usage:
#   PROJECT_ID=my-project bash provision.sh [--dry-run]
#
# Variables (all optional except PROJECT_ID):
#   REGION         location of the index and endpoint, default us-central1. The service must use
#                  the same location (MODERNIZER_VS_LOCATION, set by deploy.sh).
#   EMBED_DIM      embedding dimension, default 768 (the text-embedding-005 default)
#   DISPLAY_NAME   display name of the index, default modernizer-kb
#   PYTHON         interpreter that has this package installed with the gcp extra, default python3
#   BUILD_INDEX=1  after provisioning, also embed the knowledge base and upsert it
#                  (`index build`), which calls the embedding API
#
# Needs application default credentials with Vertex AI admin rights on the project
# (gcloud auth application-default login).
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
EMBED_DIM="${EMBED_DIM:-768}"
DISPLAY_NAME="${DISPLAY_NAME:-modernizer-kb}"
PYTHON="${PYTHON:-python3}"
BUILD_INDEX="${BUILD_INDEX:-0}"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ -n "$PROJECT_ID" ]] || die "PROJECT_ID is required"
[[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "invalid PROJECT_ID: $PROJECT_ID"
[[ "$REGION" =~ ^[a-z]+-[a-z]+[0-9]+$ ]] || die "invalid REGION: $REGION"
[[ "$EMBED_DIM" =~ ^[0-9]+$ && "$EMBED_DIM" -gt 0 ]] || die "invalid EMBED_DIM: $EMBED_DIM"
[[ "$DISPLAY_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "invalid DISPLAY_NAME: $DISPLAY_NAME"

command_line=("$PYTHON" -m modernizer.cli index provision
  --project "$PROJECT_ID" --location "$REGION" --dim "$EMBED_DIM" --display-name "$DISPLAY_NAME")

if [[ "$DRY_RUN" == 1 ]]; then
  printf '+'
  printf ' %q' "${command_line[@]}"
  printf '\n'
  echo "dry run finished, nothing was created"
  exit 0
fi

command -v "$PYTHON" >/dev/null || die "$PYTHON not found; set PYTHON to an interpreter with the package installed"

echo "Provisioning in ${PROJECT_ID}/${REGION}; index creation and deployment can take tens of minutes." >&2
# The CLI prints a JSON summary followed by 'export MODERNIZER_VS_...=...' lines on stdout.
# tee keeps the output visible even when the CLI fails half way (pipefail propagates the status).
OUT_FILE="$(mktemp)"
trap 'rm -f "$OUT_FILE"' EXIT
"${command_line[@]}" | tee "$OUT_FILE"
OUTPUT="$(cat "$OUT_FILE")"

EXPORTS="$(grep -E '^export MODERNIZER_VS_(INDEX|ENDPOINT|DEPLOYED_ID|LOCATION)=[^[:space:]]+$' <<<"$OUTPUT" || true)"
for name in MODERNIZER_VS_INDEX MODERNIZER_VS_ENDPOINT MODERNIZER_VS_DEPLOYED_ID; do
  if ! grep -q "^export ${name}=" <<<"$EXPORTS"; then
    echo "error: the CLI output did not contain a value for ${name}; check the output above" >&2
    echo "       and the console (Vertex AI > Vector Search): resources may already exist and bill." >&2
    exit 1
  fi
done

cat <<EOF

Provisioned. Export these for 'modernize index build' and for deploy/cloudrun/deploy.sh:

${EXPORTS}

Resources keep billing until torn down: see deploy/vector_search/README.md.
EOF

if [[ "$BUILD_INDEX" == 1 ]]; then
  while IFS='=' read -r key value; do
    export "${key#export }=${value}"
  done <<<"$EXPORTS"
  export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
  "$PYTHON" -m modernizer.cli index build
else
  echo "Next: export the values above and run: ${PYTHON} -m modernizer.cli index build (or re-run with BUILD_INDEX=1)."
fi
