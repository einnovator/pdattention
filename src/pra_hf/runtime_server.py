"""Managed local Hugging Face runtime used by ``pra runtime serve -e hf``."""

from __future__ import annotations

import argparse

import torch

from .deployment import HuggingFaceEngineAdapter
from .bundle import PRAModelBundle
from .gateway import PRAGateway, serve_gateway
from .hf_storage import HFReferenceHotBridge
from .model import PRAForCausalLM
from .observability import Observability, load_observability_config
from .storage_lifecycle import PRAStorageManager, PRAStoragePolicy


def _apply_runtime_config(gateway: PRAGateway, values):
    """Apply the live-mutable subset owned by the reference runtime."""

    from .management import ManagementAPIError

    unsupported = sorted(set(values) - {"profile"})
    if unsupported:
        raise ManagementAPIError(
            501,
            "CONFIG_FIELD_NOT_SUPPORTED",
            "The reference runtime can update only the default profile live.",
            unsupported_fields=unsupported,
        )
    if "profile" in values:
        gateway.default_profile = str(values["profile"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pra-bundle")
    parser.add_argument("--profile")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--storage", default="balanced")
    parser.add_argument("--storage-config")
    parser.add_argument("--observability")
    parser.add_argument("--otel", action="store_true")
    parser.add_argument("--otel-endpoint")
    parser.add_argument("--prometheus", action="store_true")
    parser.add_argument("--prometheus-port", type=int)
    parser.add_argument("--management-api", action="store_true")
    parser.add_argument("--management-host", default="127.0.0.1")
    parser.add_argument("--management-port", type=int, default=9101)
    parser.add_argument("--management-auth-mode", default="none")
    parser.add_argument("--management-token-env", default="PRA_MANAGEMENT_TOKEN")
    parser.add_argument("--management-metrics-url")
    parser.add_argument("--management-trace-url")
    parser.add_argument("--management-grafana-url")
    args = parser.parse_args()
    storage = PRAStoragePolicy.from_yaml(args.storage_config) if args.storage_config else PRAStoragePolicy.named(args.storage)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model_kwargs = {"revision": args.revision}
    if device == "cuda":
        model_kwargs.update(torch_dtype=torch.float16, device_map="cuda")
    bundle_source = None
    if args.pra_bundle and args.pra_bundle.lower() != "none":
        bundle = PRAModelBundle.from_pretrained(args.pra_bundle)
        bundle_source = bundle.source
        if bundle.base_model.get("id") != args.model:
            raise ValueError("PRA bundle base model does not match --model.")
        if args.revision and bundle.base_model.get("revision") != args.revision:
            raise ValueError("PRA bundle base revision does not match --revision.")
        for name, path in bundle.selected_learned_adapters(args.profile).items():
            adapter_type = str(bundle.learned_adapters[name].get("type", name))
            if adapter_type in {"routing", "router"}:
                model_kwargs["routing_adapter"] = path
            elif adapter_type in {"memory", "memory_adapter", "consumer"}:
                model_kwargs["memory_adapter"] = path
    model = PRAForCausalLM.from_pretrained(args.model, **model_kwargs)
    if device != "cuda":
        model.model.to(device)
    overrides = {}
    if args.otel or args.otel_endpoint or args.prometheus or args.prometheus_port:
        overrides["enabled"] = True
    if args.otel or args.otel_endpoint:
        overrides["otel"] = {
            "enabled": True,
            **({"endpoint": args.otel_endpoint} if args.otel_endpoint else {}),
        }
    if args.prometheus or args.prometheus_port:
        overrides["prometheus"] = {
            "enabled": True,
            **({"port": args.prometheus_port} if args.prometheus_port else {}),
        }
    telemetry = Observability(
        load_observability_config(
            args.observability, overrides=overrides, service="runtime"
        ),
        start_server=True,
    )
    storage_manager = PRAStorageManager(
        storage,
        hot=HFReferenceHotBridge(model),
        observability=telemetry,
        engine="hf",
    )
    storage_manager.start_maintenance()
    gateway = PRAGateway(
        HuggingFaceEngineAdapter(
            model, storage_manager=storage_manager, observability=telemetry
        ),
        mode="G00",
        observability=telemetry,
        bundle_source=bundle_source,
        default_profile=args.profile or "default",
    )
    management_server = None
    if args.management_api:
        from .management import (
            LoadedModel,
            ManagementAPIConfig,
            ManagementAuthConfig,
            ManagementProvider,
            PRAProfileSummary,
            start_management_api,
        )

        management_server = start_management_api(
            ManagementProvider(
                engine="hf",
                capabilities=gateway.engine.capabilities().to_dict(),
                models=[LoadedModel(
                    model_id=args.model,
                    revision=args.revision,
                    pra_bundle_id=bundle_source,
                    profile=args.profile or "default",
                    execution_mode="G00",
                    device_placement=device,
                    loaded_at=storage_manager.started_at if hasattr(storage_manager, "started_at") else None,
                )],
                profiles=[PRAProfileSummary(name=args.profile or "default", source="hf-runtime")],
                effective_config={
                    "engine": "hf", "model": args.model, "revision": args.revision,
                    "profile": args.profile or "default", "device": device,
                    "storage": storage.profile,
                },
                storage_manager=storage_manager,
                session_source=gateway.sessions,
                observability={
                    "otel": {"enabled": bool(args.otel or args.otel_endpoint)},
                    "prometheus": {"enabled": bool(args.prometheus or args.prometheus_port)},
                },
                config_patch_handler=lambda values: _apply_runtime_config(gateway, values),
            ),
            ManagementAPIConfig(
                enabled=True,
                host=args.management_host,
                port=args.management_port,
                auth=ManagementAuthConfig(
                    mode=args.management_auth_mode,
                    token_env=args.management_token_env,
                ),
                metrics_url=args.management_metrics_url,
                trace_backend_url=args.management_trace_url,
                grafana_url=args.management_grafana_url,
            ),
        )
    try:
        serve_gateway(gateway, host=args.host, port=args.port)
    finally:
        if management_server is not None:
            from .management import stop_management_api

            stop_management_api(management_server)
        storage_manager.close()
        telemetry.close()


if __name__ == "__main__":
    main()
