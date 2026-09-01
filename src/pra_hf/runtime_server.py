"""Managed local Hugging Face runtime used by ``pra runtime serve -e hf``."""

from __future__ import annotations

import argparse

from .deployment import HuggingFaceEngineAdapter
from .bundle import PRAModelBundle
from .gateway import PRAGateway, serve_gateway
from .hf_storage import HFReferenceHotBridge
from .model import PRAForCausalLM
from .observability import Observability, load_observability_config
from .storage_lifecycle import PRAStorageManager, PRAStoragePolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
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
    args = parser.parse_args()
    storage = PRAStoragePolicy.from_yaml(args.storage_config) if args.storage_config else PRAStoragePolicy.named(args.storage)
    model_kwargs = {"revision": args.revision}
    if args.pra_bundle and args.pra_bundle.lower() != "none":
        bundle = PRAModelBundle.from_pretrained(args.pra_bundle)
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
    )
    try:
        serve_gateway(gateway, host=args.host, port=args.port)
    finally:
        storage_manager.close()
        telemetry.close()


if __name__ == "__main__":
    main()
