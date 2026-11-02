#!/usr/bin/env python3
"""Deploy, inspect and tear down a Gemma 3 12B endpoint on Vertex AI (Model Garden).

Subcommands:

  models    list Model Garden models you can deploy (to find the exact Gemma model name)
  options   list the verified deployment options (machine type, accelerator) of one model
  deploy    deploy the model to a new endpoint; THIS STARTS NODE-HOUR BILLING
  describe  show an existing endpoint and the MODERNIZER_OPEN_* values for it
  teardown  undeploy every model and delete the endpoint; this STOPS the billing
  price     turn a node-hour price and a MEASURED throughput into a pricing-table entry

Needs the Vertex AI SDK (pip install 'code-modernization-assistant[gcp]') and application
default credentials (gcloud auth application-default login). Written against the SDK sources
installed with google-cloud-aiplatform 2.7.0; it has NOT been run against a live project.
Read README.md in this directory first.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import yaml

from modernizer.llm.pricing import amortised_price_per_mtok

try:
    import vertexai
    from google.cloud import aiplatform
    from vertexai import model_garden
except ImportError:
    vertexai = None
    aiplatform = None
    model_garden = None

# Model Garden name in the "publisher/model@version" form, by analogy with the SDK docstring
# example (google/gemma2@gemma-2-2b-it). Unverified: confirm it with the `models` subcommand.
DEFAULT_MODEL = "google/gemma3@gemma-3-12b-it"
SERVED_MODEL_NAME = "gemma-3-12b-it"

BILLING_WARNING = (
    "WARNING: a deployed endpoint bills per node hour (accelerator plus host), also while it\n"
    "serves no request. Run the teardown subcommand when you are done and check the console."
)


def fail(message: str) -> SystemExit:
    return SystemExit(f"error: {message}")


def require_sdk() -> None:
    if vertexai is None or aiplatform is None or model_garden is None:
        raise fail(
            "the Vertex AI SDK is not installed; run: "
            "pip install 'code-modernization-assistant[gcp]'"
        )


def init_sdk(args: argparse.Namespace) -> None:
    require_sdk()
    if not args.project:
        raise fail("--project is required (or set GOOGLE_CLOUD_PROJECT)")
    if not args.region:
        raise fail("--region is required")
    vertexai.init(project=args.project, location=args.region)


def modernizer_exports(endpoint: aiplatform.Endpoint, region: str) -> list[str]:
    """The service settings for this endpoint; both URL routes are unverified (see README)."""
    lines = [
        "# Route A: regional host, used by MODERNIZER_OPEN_BACKEND=endpoint (shared endpoints):",
        "export MODERNIZER_OPEN_BACKEND=endpoint",
        f"export MODERNIZER_OPEN_ENDPOINT_ID={endpoint.name}",
        f"export MODERNIZER_OPEN_LOCATION={region}",
        f"export MODERNIZER_OPEN_MODEL={SERVED_MODEL_NAME}",
    ]
    dns = endpoint.dedicated_endpoint_dns if endpoint.dedicated_endpoint_enabled else None
    if dns:
        base = f"https://{dns}/v1/{endpoint.resource_name}/invoke/v1"
        lines += [
            "# Route B: this is a DEDICATED endpoint; the SDK reaches it on its own DNS name and",
            "# the 'invoke' path. If route A is refused, use this instead (it replaces the above):",
            f"export MODERNIZER_OPEN_BASE_URL={base}",
        ]
    return lines


def command_models(args: argparse.Namespace) -> None:
    init_sdk(args)
    for name in model_garden.list_deployable_models(model_filter=args.filter):
        print(name)


def command_options(args: argparse.Namespace) -> None:
    init_sdk(args)
    print(model_garden.OpenModel(args.model).list_deploy_options(concise=True))


def command_deploy(args: argparse.Namespace) -> None:
    parameters = {
        "accept_eula": args.accept_eula,
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "accelerator_count": args.accelerator_count,
        "min_replica_count": args.replicas,
        "max_replica_count": args.replicas,
        "endpoint_display_name": args.endpoint_name,
        "dedicated_endpoint_disabled": args.shared_endpoint,
    }
    if args.dry_run:
        print(f"dry run: would deploy {args.model} in {args.region} of {args.project} with:")
        for key, value in parameters.items():
            print(f"  {key} = {value!r}")
        print(BILLING_WARNING)
        return
    if not args.accept_eula:
        raise fail("deploying needs --accept-eula, after you have read the Gemma terms of use")
    init_sdk(args)
    print(BILLING_WARNING, file=sys.stderr)
    endpoint = model_garden.OpenModel(args.model).deploy(**parameters)
    print(f"deployed: {endpoint.resource_name}")
    print("\n".join(modernizer_exports(endpoint, args.region)))


def command_describe(args: argparse.Namespace) -> None:
    init_sdk(args)
    endpoint = aiplatform.Endpoint(endpoint_name=args.endpoint_id)
    print(f"endpoint:   {endpoint.resource_name}")
    print(f"name:       {endpoint.display_name}")
    print(f"dedicated:  {endpoint.dedicated_endpoint_enabled}")
    deployed = endpoint.list_models()
    if not deployed:
        print("deployed models: none (nothing is billing for node hours)")
    for model in deployed:
        spec = model.dedicated_resources.machine_spec
        print(
            f"deployed model {model.id}: machine_type={spec.machine_type!r} "
            f"accelerator={spec.accelerator_type!r} x{spec.accelerator_count} "
            f"replicas={model.dedicated_resources.min_replica_count}"
            f"-{model.dedicated_resources.max_replica_count}"
        )
    print("\n".join(modernizer_exports(endpoint, args.region)))


def command_teardown(args: argparse.Namespace) -> None:
    init_sdk(args)
    endpoint = aiplatform.Endpoint(endpoint_name=args.endpoint_id)
    print(f"about to undeploy all models from and DELETE {endpoint.resource_name}")
    if not args.yes and input("type 'delete' to continue: ").strip() != "delete":
        raise fail("aborted, nothing was changed")
    endpoint.delete(force=True)
    print(
        "deleted. Confirm in the console (Vertex AI, Online prediction) that no endpoint is left."
    )


def command_price(args: argparse.Namespace) -> None:
    try:
        rate = amortised_price_per_mtok(
            args.node_hour_usd, args.tokens_per_second, args.utilisation
        )
    except ValueError as exc:
        raise fail(str(exc)) from None
    rate = round(rate, 6)
    entry = {
        "models": {
            SERVED_MODEL_NAME: {"input_per_mtok": rate, "output_per_mtok": rate},
        }
    }
    print(
        "# Amortised rate computed from the figures you passed in (node hour "
        f"{args.node_hour_usd}, {args.tokens_per_second} tokens/s,\n"
        f"# utilisation {args.utilisation}). Copy this entry into your pricing file; "
        "that file replaces the packaged table."
    )
    print(yaml.safe_dump(entry, sort_keys=False).rstrip())


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="default: $GOOGLE_CLOUD_PROJECT",
    )
    common.add_argument("--region", help="Vertex AI region, for example the one you were quoted")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", parents=[common], help="list deployable models")
    models.add_argument("--filter", default="gemma-3", help="substring filter (default: gemma-3)")
    models.set_defaults(func=command_models)

    options = sub.add_parser("options", parents=[common], help="list verified deploy options")
    options.add_argument("--model", default=DEFAULT_MODEL)
    options.set_defaults(func=command_options)

    deploy = sub.add_parser("deploy", parents=[common], help="deploy (STARTS BILLING)")
    deploy.add_argument("--model", default=DEFAULT_MODEL)
    deploy.add_argument("--endpoint-name", default="gemma-3-12b-it-modernizer")
    deploy.add_argument("--machine-type", required=True, help="from the `options` output")
    deploy.add_argument("--accelerator-type", required=True, help="from the `options` output")
    deploy.add_argument("--accelerator-count", type=positive_int, required=True)
    deploy.add_argument("--replicas", type=positive_int, default=1, help="min = max replicas")
    deploy.add_argument(
        "--shared-endpoint",
        action="store_true",
        help="ask for a shared (non-dedicated) endpoint, which the default route A can reach",
    )
    deploy.add_argument("--accept-eula", action="store_true", help="accept the Gemma terms of use")
    deploy.add_argument("--dry-run", action="store_true", help="print the request, call nothing")
    deploy.set_defaults(func=command_deploy)

    describe = sub.add_parser("describe", parents=[common], help="show an endpoint")
    describe.add_argument("--endpoint-id", required=True)
    describe.set_defaults(func=command_describe)

    teardown = sub.add_parser(
        "teardown", parents=[common], help="delete an endpoint (STOPS BILLING)"
    )
    teardown.add_argument("--endpoint-id", required=True)
    teardown.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    teardown.set_defaults(func=command_teardown)

    price = sub.add_parser("price", help="amortised USD per million tokens from YOUR figures")
    price.add_argument(
        "--node-hour-usd", type=float, required=True, help="price of the node per hour"
    )
    price.add_argument(
        "--tokens-per-second", type=float, required=True, help="MEASURED sustained throughput"
    )
    price.add_argument(
        "--utilisation",
        type=float,
        default=1.0,
        help="fraction of billed time the node serves requests, in (0, 1]; idle time bills too",
    )
    price.set_defaults(func=command_price)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
