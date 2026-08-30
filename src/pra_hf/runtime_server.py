"""Managed local Hugging Face runtime used by ``pra runtime serve -e hf``."""

from __future__ import annotations

import argparse

from .deployment import HuggingFaceEngineAdapter
from .gateway import PRAGateway, serve_gateway
from .hf_storage import HFReferenceHotBridge
from .model import PRAForCausalLM
from .storage_lifecycle import PRAStorageManager, PRAStoragePolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--storage", default="balanced")
    parser.add_argument("--storage-config")
    args = parser.parse_args()
    storage = PRAStoragePolicy.from_yaml(args.storage_config) if args.storage_config else PRAStoragePolicy.named(args.storage)
    model = PRAForCausalLM.from_pretrained(args.model, revision=args.revision)
    storage_manager = PRAStorageManager(
        storage, hot=HFReferenceHotBridge(model)
    )
    storage_manager.start_maintenance()
    gateway = PRAGateway(
        HuggingFaceEngineAdapter(model, storage_manager=storage_manager), mode="G00"
    )
    try:
        serve_gateway(gateway, host=args.host, port=args.port)
    finally:
        storage_manager.close()


if __name__ == "__main__":
    main()
