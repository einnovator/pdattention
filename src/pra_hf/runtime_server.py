"""Managed local Hugging Face runtime used by ``pra runtime serve -e hf``."""

from __future__ import annotations

import argparse

from .deployment import HuggingFaceEngineAdapter
from .gateway import PRAGateway, serve_gateway
from .model import PRAForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    model = PRAForCausalLM.from_pretrained(args.model, revision=args.revision)
    gateway = PRAGateway(HuggingFaceEngineAdapter(model), mode="G00")
    serve_gateway(gateway, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
