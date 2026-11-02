#!/usr/bin/env bash
# Creates the dataset, the two partitioned tables and the reporting views. Safe to re-run.
#
# Usage: PROJECT=my-project DATASET=modernizer ./setup.sh
#    or: ./setup.sh my-project modernizer [location]
# Needs the gcloud SDK's `bq` CLI, authenticated with BigQuery admin rights on the project.
set -euo pipefail

PROJECT="${1:-${PROJECT:-}}"
DATASET="${2:-${DATASET:-}}"
LOCATION="${3:-${LOCATION:-US}}"

if [[ -z "$PROJECT" || -z "$DATASET" ]]; then
  echo "usage: PROJECT=<project> DATASET=<dataset> $0   (or: $0 <project> <dataset> [location])" >&2
  exit 2
fi
# The values are substituted into SQL with sed, so keep them to the characters GCP allows.
if [[ ! "$PROJECT" =~ ^[a-z][a-z0-9.:-]*$ ]]; then
  echo "invalid PROJECT: $PROJECT" >&2
  exit 2
fi
if [[ ! "$DATASET" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "invalid DATASET: $DATASET" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# `bq mk` exits non-zero when the object exists. Tolerate exactly that, fail on anything else.
mk_idempotent() {
  local label="$1" out
  shift
  if out="$("$@" 2>&1)"; then
    echo "$out"
  elif grep -qi "already exists" <<<"$out"; then
    echo "exists, skipping: ${label}"
  else
    echo "$out" >&2
    return 1
  fi
}

mk_idempotent "${PROJECT}:${DATASET}" bq --project_id="$PROJECT" --location="$LOCATION" mk --dataset \
  --description "Code modernization assistant results" \
  "${PROJECT}:${DATASET}"

for table in module_results llm_calls; do
  mk_idempotent "${PROJECT}:${DATASET}.${table}" bq --project_id="$PROJECT" --location="$LOCATION" mk --table \
    --time_partitioning_field timestamp \
    --time_partitioning_type DAY \
    "${PROJECT}:${DATASET}.${table}" "${HERE}/schema_${table}.json"
done

# The views are CREATE OR REPLACE, so re-running always converges on views.sql.
# Statements are sent one at a time so a failure points at the offending view.
SQL="$(tr -d '\r' <"${HERE}/views.sql" | sed -e "s|\${PROJECT}|${PROJECT}|g" -e "s|\${DATASET}|${DATASET}|g")"
STATEMENT=""
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" == --* ]] && continue
  STATEMENT+="${line}"$'\n'
  if [[ "$line" == *";" ]]; then
    bq --project_id="$PROJECT" --location="$LOCATION" query --use_legacy_sql=false --quiet "$STATEMENT"
    STATEMENT=""
  fi
done <<<"$SQL"

echo "BigQuery setup complete: ${PROJECT}:${DATASET}"
