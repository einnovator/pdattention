"""Managed local Hugging Face runtime used by ``pra runtime serve -e hf``."""

from __future__ import annotations

import argparse

from .deployment import HuggingFaceEngineAdapter
from .gateway import PRAGateway, serve_gateway
from .model import PRAForCausalLM
from .storage_lifecycle import PRAStoragePolicy


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
    gateway = PRAGateway(HuggingFaceEngineAdapter(model), mode="G00")
    gateway.storage_policy = storage
    serve_gateway(gateway, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
