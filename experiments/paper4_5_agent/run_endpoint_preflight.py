"""Create a live endpoint qualification receipt for replay after a clean restart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .runners.swebench_verified import gateway_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--served-model", default="qwen3-coder:30b")
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options = SimpleNamespace(
        mode=args.mode,
        base_url=args.base_url,
        served_model=args.served_model,
        prefix_caching=args.prefix_caching,
        require_endpoint_preflight=True,
        endpoint_preflight_receipt=None,
        chat_template_no_thinking=False,
    )
    result = gateway_preflight(options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"gateway_preflight": result}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
