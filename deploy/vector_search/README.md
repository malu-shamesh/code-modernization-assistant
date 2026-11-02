# Vertex AI Vector Search for the retrieval stage

By default the modernizer retrieves Fortran 77 to Python migration notes from a local in-memory
index that is built at startup from the bundled knowledge base. This directory provisions the managed alternative.

## Provision

```bash
gcloud auth application-default login
PROJECT_ID=my-project bash deploy/vector_search/provision.sh --dry-run   # shows the command
PROJECT_ID=my-project BUILD_INDEX=1 bash deploy/vector_search/provision.sh
```

`provision.sh` calls `python -m modernizer.cli index provision --project ... --location ...
--dim ...`, which creates a streaming-update index and a public endpoint and deploys the index to
it. It then prints the three values the service needs:

```
export MODERNIZER_VS_INDEX=...
export MODERNIZER_VS_ENDPOINT=...
export MODERNIZER_VS_DEPLOYED_ID=...
```

With `BUILD_INDEX=1` it also runs `python -m modernizer.cli index build`, which embeds the
knowledge base with the embedding API and upserts it. Without it, export the values and run that
command yourself. Export the same three values (and optionally `MODERNIZER_VS_LOCATION`) before
`deploy/cloudrun/deploy.sh` to make the service use Vector Search; the script then sets
`MODERNIZER_RETRIEVAL=vertex`.

Variables: `PROJECT_ID` (required), `REGION` (default `us-central1`; the service must use the same
location), `EMBED_DIM` (default 768, must match `MODERNIZER_EMBED_DIM`), `DISPLAY_NAME`, `PYTHON`.
Index creation and deployment are synchronous and can take tens of minutes.

## Cost

A deployed index bills for the machine replicas behind it for as long as it is deployed, whether
or not any query arrives. An index and an endpoint that are merely created, but not deployed, are
cheaper but still resources you own. The embedding API bills per token when the knowledge base is
built and for every query embedding. See the current Vertex AI Vector Search and embeddings price
lists. Nothing here is free: tear the resources down when you stop using them.

## Teardown

Order matters: an index cannot be deleted while it is deployed to an endpoint.

```bash
PROJECT_ID=my-project
REGION=us-central1
ENDPOINT_ID=...          # last path segment of MODERNIZER_VS_ENDPOINT
INDEX_ID=...             # last path segment of MODERNIZER_VS_INDEX
DEPLOYED_ID=...          # MODERNIZER_VS_DEPLOYED_ID

# 1. stop the replica billing
gcloud ai index-endpoints undeploy-index "$ENDPOINT_ID" \
  --deployed-index-id="$DEPLOYED_ID" --region="$REGION" --project="$PROJECT_ID"

# 2. delete the endpoint
gcloud ai index-endpoints delete "$ENDPOINT_ID" --region="$REGION" --project="$PROJECT_ID"

# 3. delete the index
gcloud ai indexes delete "$INDEX_ID" --region="$REGION" --project="$PROJECT_ID"

# check that nothing is left
gcloud ai index-endpoints list --region="$REGION" --project="$PROJECT_ID"
gcloud ai indexes list --region="$REGION" --project="$PROJECT_ID"
```

These command names and flags are written from memory of the `gcloud ai` reference and have NOT
been run or checked against the installed SDK. Confirm each with `gcloud ai index-endpoints
--help` and `gcloud ai indexes --help` (or in the console under Vertex AI, Vector Search) before
relying on them, and verify in the console that no deployed index remains.

After teardown, redeploy the Cloud Run service without the `MODERNIZER_VS_*` variables so that it
falls back to the local index, otherwise it will fail at startup.

## What has NOT been verified

`provision.sh` was only run with `--dry-run` and against stub interpreters. It has never created
a real index. It relies on the output format of `modernizer.cli index provision` (JSON followed by
`export MODERNIZER_VS_*=...` lines) and fails with a message if the three values are missing.
