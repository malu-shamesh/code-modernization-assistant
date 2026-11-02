# Gemma 3 12B on a Vertex AI endpoint (the open tier)

The modernizer sends a target share of the modules (for example 70%) to a small open model and
the rest to a frontier model. By default the open model is Gemma 3 12B (instruction tuned,
`gemma-3-12b-it`) on a Vertex AI endpoint that you deploy yourself from Model Garden. This
directory explains how to deploy it, find the endpoint id, test it, wire it into the service,
price it honestly, and tear it down.

## Read this first: cost

**A deployed endpoint bills per node hour, not per token.** From the moment the model is deployed
until you undeploy it, the machine and accelerator behind it bill every hour, including hours in
which no request arrives. Vertex AI endpoints with dedicated resources keep at least one replica
running (to my knowledge there is no scale to zero for them; not verified). Leaving an endpoint
up over a weekend is a real cost. Use the teardown section when you are done and check the console
afterwards.

Consequence for the reports: Gemini is billed per token, Gemma on your endpoint is billed per
hour. Any per-token price for Gemma is therefore an *amortised* figure that depends on how busy
you keep the node. See "The amortised price" below; the number in the packaged price table is a
placeholder.

## Status of this document

Nothing here could be run: the machine this was written on has no `gcloud`, no Docker and no
Google Cloud project. Statements are labelled:

* **[SDK source]** checked by reading the installed `google-cloud-aiplatform` 2.7.0 and
  `vertexai` sources, not by running them against Google Cloud.
* **[memory]** written from memory of the `gcloud` or console documentation. Confirm with
  `gcloud ai endpoints --help` and in the console before relying on it.
* **[unverified]** a guess or an assumption that matters; each is also listed at the end.

Machine types, accelerators, regions, quotas and prices change. This document names none of them
on purpose: take them from the Model Garden deployment panel at the time you deploy.

## 1. Before deploying

1. A project with billing enabled and the Vertex AI API on (`gcloud services enable
   aiplatform.googleapis.com`) [memory]. You need permission to deploy models and create
   endpoints (a Vertex AI administrator role is sufficient; the least privilege is not
   verified).
2. **Accelerator quota.** Online prediction accelerators are quota-limited per region and per
   accelerator type. The default may be zero for the type you need. Check the quotas page of the
   console (IAM and admin, Quotas) for the accelerator type and region you intend to use, and
   request an increase early, because approval is not instant [memory].
3. **Region.** Pick a region in which the deployment panel offers an accelerator you have quota
   for. Remember it: the service must be told the same region (`MODERNIZER_OPEN_LOCATION`). The
   code default for that variable is only a default for the code, not a recommendation.
4. **Gemma terms of use.** Model Garden asks you to accept the license before deploying; the
   script needs an explicit `--accept-eula` for the same reason.
5. **Which machine and accelerator?** Open the Gemma 3 12B card in Model Garden and read the
   options in its Deploy panel, or ask the SDK for the verified options [SDK source: the call
   exists; its output was not seen]:

   ```bash
   python deploy/vertex_gemma/gemma_endpoint.py models  --project "$PROJECT_ID" --region "$REGION"
   python deploy/vertex_gemma/gemma_endpoint.py options --project "$PROJECT_ID" --region "$REGION"
   ```

   `models` lists deployable model names containing `gemma-3` so you can confirm the exact name
   (the default `google/gemma3@gemma-3-12b-it` is built by analogy with the SDK docstring
   example `google/gemma2@gemma-2-2b-it` [unverified]). `options` prints the machine type and
   accelerator combinations Google verified for that model. Choose the smallest one that you
   have quota for; a larger one costs more per hour and only pays off if you keep it busy.

## 2. Deploy

**Shared or dedicated endpoint.** Model Garden endpoints can be dedicated (reached on their own
DNS name) or shared (reached on the regional host) [SDK source: the deploy call has
`dedicated_endpoint_disabled`, and `Endpoint.dedicated_endpoint_dns` exists]. The service's
default route (`MODERNIZER_OPEN_BACKEND=endpoint`) uses the regional host. Whether a dedicated
endpoint accepts requests there is [unverified]; the script therefore prints both routes (A and B
below) after deploying, and `--shared-endpoint` asks for a shared endpoint.

### Option A: the console

Vertex AI, Model Garden, search for Gemma 3, open the 12B instruction-tuned model, Deploy. Choose
the region, an endpoint name, and a machine type and accelerator from the panel; accept the terms.
Deployment takes a while; wait until the endpoint shows the model as deployed.

### Option B: the script (Python SDK)

```bash
python deploy/vertex_gemma/gemma_endpoint.py deploy \
  --project "$PROJECT_ID" --region "$REGION" \
  --machine-type "<from the options output>" \
  --accelerator-type "<from the options output>" --accelerator-count <n> \
  --accept-eula --dry-run          # first: prints the request and calls nothing
```

Remove `--dry-run` to deploy. The script has no default machine or accelerator on purpose. It
needs `pip install 'code-modernization-assistant[gcp]'` and application default credentials
(`gcloud auth application-default login`). When the deployment finishes it prints the endpoint
resource name and the `MODERNIZER_OPEN_*` values for it. The call is synchronous; the SDK's own
default timeout for it is two hours [SDK source].

### Option C: the SDK directly

```python
import vertexai
from vertexai import model_garden

vertexai.init(project="<project>", location="<region>")
model = model_garden.OpenModel("google/gemma3@gemma-3-12b-it")  # confirm the name, see above
print(model.list_deploy_options(concise=True))
endpoint = model.deploy(
    accept_eula=True,
    machine_type="<from the options>",
    accelerator_type="<from the options>",
    accelerator_count=1,
    endpoint_display_name="gemma-3-12b-it-modernizer",
)
print(endpoint.name)  # the endpoint id
```

The parameters are those of `OpenModel.deploy` in the installed SDK [SDK source].

## 3. Find the endpoint id

The endpoint id is the number at the end of the endpoint resource name
`projects/<project number>/locations/<region>/endpoints/<ENDPOINT_ID>`.

* Console: Vertex AI, Online prediction, Endpoints; the ID column.
* `gcloud ai endpoints list --region="$REGION" --project="$PROJECT_ID"` [memory]
* `gcloud ai endpoints describe "$ENDPOINT_ID" --region="$REGION" --project="$PROJECT_ID"` shows
  the deployed model and its resources [memory].
* `python deploy/vertex_gemma/gemma_endpoint.py describe --project ... --region ...
  --endpoint-id ...` prints the same plus the settings for the service.

Give only the id, not the resource path, to `MODERNIZER_OPEN_ENDPOINT_ID`.

## 4. Test it with curl

Both routes speak the OpenAI chat-completions protocol. Set the variables first:

```bash
PROJECT_ID=my-project
REGION=us-central1        # the region of the endpoint
ENDPOINT_ID=...
TOKEN="$(gcloud auth print-access-token)"
BODY='{"model": "gemma-3-12b-it", "temperature": 0, "max_tokens": 64,
       "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}'
```

**Route A, regional host** (what `MODERNIZER_OPEN_BACKEND=endpoint` calls; the URL shape is
documented in `src/modernizer/llm/openai_compat.py` and is [unverified] against live docs):

```bash
curl -sS -X POST "https://${REGION}-aiplatform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}/chat/completions" \
  -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" -d "$BODY"
```

**Route B, dedicated endpoint DNS** (derived from `Endpoint.invoke` in the installed SDK, which
posts to `https://<dedicated dns>/v1/<resource name>/invoke<path>` [SDK source]; not run):

```bash
DNS="<dedicated endpoint dns from the describe output>"
curl -sS -X POST "https://${DNS}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}/invoke/v1/chat/completions" \
  -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" -d "$BODY"
```

A working call returns JSON with `choices[0].message.content` and a `usage` object with
`prompt_tokens` and `completion_tokens`; the service needs both (without `usage` it records zero
tokens and the cost figures are wrong). What the `model` field must contain for a self-deployed
server is [unverified]: the SDK's own example sends an empty string, the service sends
`MODERNIZER_OPEN_MODEL`. If a call is rejected because of the model name, try `""`, and tell
whoever maintains the client. Gemma chat templates have no system role; keep
`MODERNIZER_OPEN_FOLD_SYSTEM=true` (the default) so the system prompt is folded into the user
message.

## 5. Point the service at it

```bash
export MODERNIZER_OPEN_BACKEND=endpoint
export MODERNIZER_OPEN_ENDPOINT_ID="$ENDPOINT_ID"
export MODERNIZER_OPEN_LOCATION="$REGION"
export MODERNIZER_OPEN_MODEL=gemma-3-12b-it
export MODERNIZER_OPEN_FRACTION=0.7        # target share of each batch for the open model
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
```

If route A is refused but route B works, replace the first three with
`MODERNIZER_OPEN_BASE_URL="https://${DNS}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}/invoke/v1"`.
A non-empty base URL takes precedence over the backend; the client appends `/chat/completions`,
and sends the Google access token because the host ends in a Google domain. For Cloud Run put
the same values in the env file of `deploy/cloudrun` (see its README). The service account needs
`roles/aiplatform.user`, which `deploy.sh` grants; whether that covers prediction on a dedicated
endpoint exactly as it does on a shared one is [unverified].

## 6. The amortised price per million tokens

A per-token cost for a self-hosted model is
`node_hour_usd / (tokens_per_second * 3600 * utilisation) * 1e6`, which is what
`modernizer.llm.pricing.amortised_price_per_mtok` computes. Every input is yours to supply:

* `node_hour_usd`: what the machine plus accelerators of the deployment cost per hour, from the
  current Google Cloud price list and your own billing data (or the console's estimate when
  deploying). Include every accelerator and replica.
* `tokens_per_second`: the throughput you MEASURED, not a datasheet figure. Run a representative
  batch of Fortran modules with the open tier alone (`MODERNIZER_POLICY=always_open`, tests on)
  at the concurrency you plan to use (`MODERNIZER_WORKERS`), take the open-model tokens from the
  run summary (`tokens_open`, or `input_tokens + output_tokens` of that run) and divide by the
  seconds during which the endpoint was busy. The result depends on prompt length, output length
  and concurrency, so measure with real modules. Input tokens are processed much faster than
  output tokens; if you can measure them separately, compute two rates and give them different
  `input_per_mtok` and `output_per_mtok` values.
* `utilisation`: the share of the billed hours in which the node actually served requests, in
  (0, 1]. An idle node still bills. If the endpoint stays deployed for a month and converts code
  for a few hours, the utilisation is tiny and the effective price per token is enormous
  compared with the same node kept busy. State the utilisation assumption next to every cost
  figure you publish.

Compute the entry (the figures below are placeholders for your own values):

```bash
python deploy/vertex_gemma/gemma_endpoint.py price \
  --node-hour-usd "$NODE_HOUR_USD" --tokens-per-second "$MEASURED_TOKENS_PER_SECOND" \
  --utilisation "$UTILISATION"
```

or in Python:

```python
from modernizer.llm.pricing import amortised_price_per_mtok

rate = amortised_price_per_mtok(
    node_hour_usd=NODE_HOUR_USD,
    tokens_per_second=MEASURED_TOKENS_PER_SECOND,
    utilisation=UTILISATION,
)
```

Then make a copy of `src/modernizer/llm/pricing.yaml` outside the repository, replace the
`gemma-3-12b-it` entry with your figures, check every other entry against the current price
list, and point the service at the copy:

```yaml
models:
  gemini-2.5-pro:           # take the current rates from the Vertex AI price list
    input_per_mtok: <usd>
    output_per_mtok: <usd>
  gemma-3-12b-it:           # amortised, computed above
    input_per_mtok: <rate>
    output_per_mtok: <rate>
  mock-:
    input_per_mtok: 0.0
    output_per_mtok: 0.0
```

```bash
export MODERNIZER_PRICING_FILE=/path/to/my-pricing.yaml
```

The file REPLACES the packaged table (it is not merged), so list every model you use. On Cloud
Run the path is read inside the container; see "Pricing" in `deploy/cloudrun/README.md`.

When is the open tier cheaper per token than the frontier model? Exactly when its amortised rate
is below the frontier rate for the same tokens, so the answer depends on utilisation and cannot
be read off the model size. Compare against the *total* cost of the deployed hours, not only the
cost of the tokens of one experiment.

## 7. Teardown

Do this as soon as you stop using the endpoint. **Billing continues until the model is
undeployed.**

```bash
# the id of the deployed model on the endpoint:
gcloud ai endpoints describe "$ENDPOINT_ID" --region="$REGION" --project="$PROJECT_ID"   # [memory]

# 1. stop the node-hour billing
gcloud ai endpoints undeploy-model "$ENDPOINT_ID" --deployed-model-id="$DEPLOYED_MODEL_ID" \
  --region="$REGION" --project="$PROJECT_ID"                                              # [memory]

# 2. delete the endpoint
gcloud ai endpoints delete "$ENDPOINT_ID" --region="$REGION" --project="$PROJECT_ID"      # [memory]

# check that nothing is left
gcloud ai endpoints list --region="$REGION" --project="$PROJECT_ID"
```

Or in one step with the script, which undeploys every model and then deletes the endpoint
[SDK source: `Endpoint.delete(force=True)` undeploys first]:

```bash
python deploy/vertex_gemma/gemma_endpoint.py teardown \
  --project "$PROJECT_ID" --region "$REGION" --endpoint-id "$ENDPOINT_ID"
```

Model Garden also registers a model resource in the Vertex AI model registry; it is not what
bills, and you can remove it with `gcloud ai models list` and `gcloud ai models delete` [memory].
Afterwards, redeploy or reconfigure the Cloud Run service: with the default backend and a
deleted endpoint every open-tier call fails and modules escalate to the frontier model (which
costs more per token). Confirm in the console that no endpoint remains.

## What has NOT been verified

* No command or script here was run against Google Cloud. `gemma_endpoint.py` was only run with
  `--help`, `deploy --dry-run` and `price` on a machine without credentials.
* The Model Garden name `google/gemma3@gemma-3-12b-it`, and which machine and accelerator
  combinations exist and are quota-free for you.
* All `gcloud ai` command names and flags (`endpoints list|describe|undeploy-model|delete`,
  `models list|delete`) are from memory.
* The OpenAI-compatible routes: route A (regional host, `v1beta1`, `.../chat/completions`) and
  route B (dedicated DNS, `.../invoke/v1/chat/completions`), whether each works for shared and
  dedicated endpoints, and what the `model` field must hold.
* That a Vertex AI endpoint with dedicated resources cannot scale to zero, and that
  `roles/aiplatform.user` covers prediction on it.
* No throughput, latency, quota or price figure appears here because none was measured or looked
  up; every number in your pricing file must come from your own measurements and the current
  price list.
